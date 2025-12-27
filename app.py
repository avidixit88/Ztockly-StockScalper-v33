import time
import pandas as pd
import numpy as np
import streamlit as st
from email_utils import send_email_alert, format_alert_email
import plotly.graph_objects as go

from av_client import AlphaVantageClient
from engine import scan_watchlist, fetch_bundle
from indicators import vwap as calc_vwap, session_vwap as calc_session_vwap
from signals import compute_scalp_signal, PRESETS

def load_email_secrets():
    """Load email secrets from either [email] table or flat keys."""
    email_tbl = st.secrets.get("email", {})
    smtp_server = email_tbl.get("smtp_server") or st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(email_tbl.get("smtp_port") or st.secrets.get("SMTP_PORT", 587))
    smtp_user = email_tbl.get("smtp_user") or st.secrets.get("SMTP_USER", "")
    smtp_password = email_tbl.get("smtp_password") or st.secrets.get("SMTP_APP_PASSWORD", "")
    to_email = email_tbl.get("to_email") or st.secrets.get("ALERT_TO", smtp_user)
    return smtp_server, smtp_port, smtp_user, smtp_password, to_email

def send_email_safe(payload: dict, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str, to_email: str):
    """Send an email alert and return (ok, err_msg)."""
    if not (smtp_user and smtp_password and to_email):
        return False, "Missing SMTP secrets"
    try:
        subject = f"Ztockly Alert: {payload.get('Symbol','?')} {payload.get('Bias','')}"
        body = format_alert_email(payload)
        send_email_alert(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_email=to_email,
            subject=subject,
            body=body,
        )
        return True, ""
    except Exception as e:
        return False, str(e)

st.set_page_config(page_title="Ztockly Scalping Scanner", layout="wide")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"]
if "last_alert_ts" not in st.session_state:
    st.session_state.last_alert_ts = {}
if "alerts" not in st.session_state:
    st.session_state.alerts = []

st.sidebar.title("Scalping Scanner")
watchlist_text = st.sidebar.text_area("Watchlist (comma or newline separated)", value="\n".join(st.session_state.watchlist), height=150)

interval = st.sidebar.selectbox("Intraday interval", ["1min", "5min"], index=0)
mode = st.sidebar.selectbox("Signal mode", list(PRESETS.keys()), index=list(PRESETS.keys()).index("Cleaner signals"))

st.sidebar.markdown("### VWAP")
vwap_logic = st.sidebar.selectbox("VWAP logic for signals", ["session", "cumulative"], index=0)
show_dual_vwap = st.sidebar.toggle("Dual VWAP (show both lines)", value=True)

st.sidebar.markdown("### Engine complexity")
pro_mode = st.sidebar.toggle("Pro mode (Liquidity + OB + Breaker + FVG + EMA)", value=False)

st.sidebar.markdown("### Time-of-day filter (ET)")
allow_opening = st.sidebar.checkbox("Opening (09:30–11:00)", value=True)
allow_midday = st.sidebar.checkbox("Midday (11:00–15:00)", value=False)
allow_power = st.sidebar.checkbox("Power hour (15:00–16:00)", value=True)
allow_premarket = st.sidebar.checkbox("Premarket (04:00–09:30)", value=False)
allow_afterhours = st.sidebar.checkbox("Afterhours (16:00+)", value=False)

st.sidebar.markdown("### Higher‑TF bias overlay (optional)")
enable_htf = st.sidebar.toggle("Enable HTF bias", value=False)
htf_interval = st.sidebar.selectbox("HTF interval", ["15min", "30min"], index=0, disabled=not enable_htf)
htf_strict = st.sidebar.checkbox("Strict HTF alignment", value=False, disabled=not enable_htf)

st.sidebar.markdown("### ATR calibration")
target_atr_pct = st.sidebar.slider("Target ATR% (score normalization)", 0.001, 0.010, 0.003, 0.001)

st.sidebar.markdown("### Fib logic")
show_fibs = st.sidebar.checkbox("Show Fibonacci retracement", value=True)
fib_lookback = st.sidebar.slider("Fib lookback bars", 60, 240, 120, 10) if show_fibs else 120

st.sidebar.markdown("### In‑App Alerts")
cooldown_minutes = st.sidebar.slider("Cooldown minutes (per ticker)", 1, 30, 7, 1)
alert_threshold = st.sidebar.slider("Alert score threshold", 60, 100, int(PRESETS[mode]["min_actionable_score"]), 1)
capture_alerts = st.sidebar.checkbox("Capture alerts in-app", value=True)
max_alerts_kept = st.sidebar.slider("Max alerts kept", 10, 300, 60, 10)

st.sidebar.markdown("### API pacing / refresh")
min_between_calls = st.sidebar.slider("Seconds between API calls", 0.5, 8.0, 1.5, 0.5)
auto_refresh = st.sidebar.checkbox("Auto-refresh scanner", value=False)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 10, 180, 30, 5) if auto_refresh else None

st.sidebar.markdown("---")
st.sidebar.caption("Required env var: ALPHAVANTAGE_API_KEY")

symbols = [s.strip().upper() for s in watchlist_text.replace(",", "\n").splitlines() if s.strip()]
st.session_state.watchlist = symbols

st.title("Ztockly — Intraday Reversal Scalping Engine (v7)")
st.caption("Basic: VWAP + RSI‑5 event + MACD histogram turn + volume. Pro adds sweeps/OB/breaker/FVG/EMA. v7 adds fib‑anchored TPs, liquidity‑weighted scoring, ATR score normalization, and optional HTF bias.")

@st.cache_resource
def get_client(min_seconds_between_calls: float):
    client = AlphaVantageClient()
    client.cfg.min_seconds_between_calls = float(min_seconds_between_calls)
    return client

client = get_client(min_between_calls)

def _now_label() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def can_alert(symbol: str, now_ts: float, cooldown_min: int) -> bool:
    last = st.session_state.last_alert_ts.get(symbol)
    if last is None:
        return True
    return (now_ts - float(last)) >= cooldown_min * 60.0

def add_in_app_alert(row: dict) -> None:
    alert = {
        "ts_unix": time.time(),
        "time": _now_label(),
        "symbol": row["Symbol"],
        "bias": row["Bias"],
        "score": int(row["Score"]),
        "session": row.get("Session"),
        "last": row.get("Last"),
        "entry": row.get("Entry"),
        "stop": row.get("Stop"),
        "t1": row.get("Target 1R"),
        "t2": row.get("Target 2R"),
        "why": row.get("Why"),
        "as_of": row.get("AsOf"),
        "mode": mode,
        "interval": interval,
        "pro_mode": pro_mode,
        "extras": row.get("Extras", {}),
    }
    st.session_state.alerts.insert(0, alert)
    st.session_state.alerts = st.session_state.alerts[: int(max_alerts_kept)]

def render_alerts_panel():
    st.subheader("🚨 Live Alerts")
    left, right = st.columns([2, 1])

    with right:
        st.metric("Alerts stored", len(st.session_state.alerts))
        if st.button("Clear alerts", type="secondary"):
            st.session_state.alerts = []
            st.session_state.last_alert_ts = {}
            st.rerun()
        st.markdown("**Filters**")
        f_bias = st.multiselect("Bias", ["LONG", "SHORT"], default=["LONG", "SHORT"])
        min_score = st.slider("Min score", 0, 100, 80, 1)

    with left:
        alerts = [a for a in st.session_state.alerts if a["bias"] in f_bias and a["score"] >= min_score]
        if not alerts:
            st.info("No alerts yet. Turn on auto-refresh + capture alerts, then let it scan.")
            return

        for a in alerts[:30]:
            badge = "🟢" if a["bias"] == "LONG" else "🔴"
            pro_badge = "⚡ Pro" if a.get("pro_mode") else "🧱 Basic"
            title = f"{badge} **{a['symbol']}** — **{a['bias']}** — Score **{a['score']}** ({a.get('session','')}) • {pro_badge}"
            with st.container(border=True):
                st.markdown(title)
                cols = st.columns(6)
                cols[0].metric("Last", f"{a['last']:.4f}" if a["last"] is not None else "N/A")
                cols[1].metric("Entry", f"{a['entry']:.4f}" if a["entry"] is not None else "—")
                cols[2].metric("Stop", f"{a['stop']:.4f}" if a["stop"] is not None else "—")
                cols[3].metric("1R", f"{a['t1']:.4f}" if a["t1"] is not None else "—")
                cols[4].metric("2R", f"{a['t2']:.4f}" if a["t2"] is not None else "—")
                fib_tp1 = (a.get("extras") or {}).get("fib_tp1")
                cols[5].metric("Fib TP1", f"{fib_tp1:.4f}" if isinstance(fib_tp1, (float,int)) else "—")
                st.caption(f"{a['time']} • interval={a['interval']} • mode={a['mode']} • VWAP={a.get('extras',{}).get('vwap_logic')} • liquidity={a.get('extras',{}).get('liquidity_phase')} • as_of={a.get('as_of')}")
                st.write(a.get("why") or "")

                ex = a.get("extras") or {}
                chips = []
                if ex.get("bull_liquidity_sweep"): chips.append("Liquidity sweep (low)")
                if ex.get("bear_liquidity_sweep"): chips.append("Liquidity sweep (high)")
                if ex.get("bull_ob_retest"): chips.append("Bull OB retest")
                if ex.get("bear_ob_retest"): chips.append("Bear OB retest")
                if ex.get("bull_breaker_retest"): chips.append("Bull breaker retest")
                if ex.get("bear_breaker_retest"): chips.append("Bear breaker retest")
                if ex.get("fib_near_long") or ex.get("fib_near_short"): chips.append("Near Fib")
                if ex.get("htf_bias_value") in ("BULL","BEAR"): chips.append(f"HTF {ex.get('htf_bias_value')}")
                if chips:
                    st.markdown("**Chips:** " + " • ".join([f"`{c}`" for c in chips]))
                with st.expander("Raw payload"):
                    st.json(a)

tab_scan, tab_alerts = st.tabs(["📡 Scanner", "🚨 Alerts"])
with tab_alerts:
    render_alerts_panel()

with tab_scan:
    col_a, col_b, col_c, col_d = st.columns([1, 1, 2, 1])
    with col_a:
        scan_now = st.button("Scan Watchlist", type="primary")
with col_b:
    if st.button("Capture test alert"):
        test = {
            "Symbol": "TEST",
            "Bias": "LONG",
            "Score": 95,
            "Session": "TEST",
            "Last": 100.00,
            "Entry": 100.00,
            "Stop": 99.50,
            "TP1": 100.50,
            "TP2": 101.00,
            "Why": "Test alert (wiring check).",
            "AsOf": pd.Timestamp.utcnow().isoformat(),
        }
        add_in_app_alert(test)
        st.success("Test alert captured in-app.")
        if enable_email_alerts:
            ok, err = send_email_safe(test, smtp_server, smtp_port, smtp_user, smtp_password, to_email)
            if ok:
                st.success(f"Test email sent to {to_email}.")
            else:
                st.error(f"Test email failed: {err}")
    with col_c:
        st.write("Tip: Keep watchlist small (5–15) to stay within API limits.")
    with col_d:
        st.write(f"Now: {_now_label()}")

    def run_scan():
        if not symbols:
            st.warning("Add at least one ticker to your watchlist.")
            return []
        with st.spinner("Scanning watchlist..."):
            return scan_watchlist(
                client, symbols,
                interval=interval,
                mode=mode,
                pro_mode=pro_mode,
                allow_opening=allow_opening,
                allow_midday=allow_midday,
                allow_power=allow_power,
                allow_premarket=allow_premarket,
                allow_afterhours=allow_afterhours,
                vwap_logic=vwap_logic,
                fib_lookback_bars=fib_lookback,
                enable_htf_bias=enable_htf,
                htf_interval=htf_interval,
                htf_strict=htf_strict,
                target_atr_pct=target_atr_pct,
            )

    results = []
    if auto_refresh:
        results = run_scan()
        st.info(f"Auto-refresh is ON — rerunning every ~{refresh_seconds}s.")
    else:
        if scan_now:
            results = run_scan()

    if results:
        df = pd.DataFrame([{
            "Symbol": r.symbol,
            "Bias": r.bias,
            "Score": r.setup_score,
            "Session": r.session,
            "Last": r.last_price,
            "Entry": r.entry,
            "Stop": r.stop,
            "Target 1R": r.target_1r,
            "Target 2R": r.target_2r,
            "Why": r.reason,
            "AsOf": str(r.timestamp) if r.timestamp is not None else None,
            "Extras": r.extras,
        } for r in results])

        st.subheader("Ranked Setups")
        st.dataframe(
            df.drop(columns=["Extras"]),
            use_container_width=True,
            hide_index=True,
            column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100)},
        )

        top = results[0]
        pro_badge = "⚡ Pro" if pro_mode else "🧱 Basic"
        st.success(f"Top setup: **{top.symbol}** — **{top.bias}** (Score {top.setup_score}, {top.session}) • {pro_badge}")

        if capture_alerts:
            now = time.time()
            for r in results:
                if r.bias in ["LONG", "SHORT"] and r.setup_score >= alert_threshold:
                    if can_alert(r.symbol, now, cooldown_minutes):
                        row = df.loc[df['Symbol'] == r.symbol].iloc[0].to_dict()
                        add_in_app_alert(row)
                        st.session_state.last_alert_ts[r.symbol] = now

        st.subheader("Chart & Signal Detail")
        pick = st.selectbox("Select ticker", [r.symbol for r in results], index=0)

        with st.spinner(f"Loading chart data for {pick}..."):
            ohlcv, rsi5, rsi14, macd_hist, quote = fetch_bundle(client, pick, interval=interval)

        sig = compute_scalp_signal(
            pick, ohlcv, rsi5, rsi14, macd_hist,
            mode=mode,
            pro_mode=pro_mode,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
            vwap_logic=vwap_logic,
            fib_lookback_bars=fib_lookback,
            target_atr_pct=target_atr_pct,
        )

        plot_df = ohlcv.sort_index().copy().tail(260)
        plot_df["vwap_cum"] = calc_vwap(plot_df)
        plot_df["vwap_sess"] = calc_session_vwap(plot_df)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"], name="Price"))

        if show_dual_vwap:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap_sess"], mode="lines", name="VWAP (Session)"))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap_cum"], mode="lines", name="VWAP (Cumulative)"))
        else:
            key = "vwap_sess" if vwap_logic == "session" else "vwap_cum"
            nm = "VWAP (Session)" if vwap_logic == "session" else "VWAP (Cumulative)"
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[key], mode="lines", name=nm))

        # Fib lines (visual)
        if show_fibs:
            seg = plot_df.tail(int(min(fib_lookback, len(plot_df))))
            hi = float(seg["high"].max())
            lo = float(seg["low"].min())
            if hi > lo:
                for name, level in [("Fib 0.382", hi - 0.382*(hi-lo)), ("Fib 0.5", hi - 0.5*(hi-lo)), ("Fib 0.618", hi - 0.618*(hi-lo)), ("Fib 0.786", hi - 0.786*(hi-lo))]:
                    fig.add_hline(y=level, line_dash="dot", annotation_text=name, annotation_position="top left")

        # Entry/Stop/Targets
        if sig.entry and sig.stop:
            fig.add_hline(y=sig.entry, line_dash="dot", annotation_text="Entry", annotation_position="top left")
            fig.add_hline(y=sig.stop, line_dash="dash", annotation_text="Stop", annotation_position="bottom left")
        if sig.target_1r:
            fig.add_hline(y=sig.target_1r, line_dash="dot", annotation_text="1R", annotation_position="top right")
        if sig.target_2r:
            fig.add_hline(y=sig.target_2r, line_dash="dot", annotation_text="2R", annotation_position="top right")
        fib_tp1 = (sig.extras or {}).get("fib_tp1")
        fib_tp2 = (sig.extras or {}).get("fib_tp2")
        if isinstance(fib_tp1, (float,int)):
            fig.add_hline(y=float(fib_tp1), line_dash="dash", annotation_text="Fib TP1", annotation_position="top right")
        if isinstance(fib_tp2, (float,int)):
            fig.add_hline(y=float(fib_tp2), line_dash="dash", annotation_text="Fib TP2", annotation_position="top right")

        fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("Bias", sig.bias)
        with c2: st.metric("Score", sig.setup_score)
        with c3: st.metric("Session", sig.session)
        with c4: st.metric("Liquidity", (sig.extras or {}).get("liquidity_phase", ""))
        with c5:
            lp = quote if quote is not None else sig.last_price
            st.metric("Last", f"{lp:.4f}" if lp is not None else "N/A")
        with c6:
            sc = (sig.extras or {}).get("atr_score_scale")
            st.metric("ATR scale", f"{sc:.2f}" if isinstance(sc, (float,int)) else "N/A")

        st.write("**Reasoning:**", sig.reason)

        st.markdown("### Trade Plan")
        if sig.bias in ["LONG", "SHORT"] and sig.entry and sig.stop:
            st.write(f"- **Entry:** {sig.entry:.4f}")
            st.write(f"- **Stop:** {sig.stop:.4f}")
            st.write(f"- **Targets (R):** 1R={sig.target_1r:.4f} • 2R={sig.target_2r:.4f}")
            if isinstance(fib_tp1, (float,int)) or isinstance(fib_tp2, (float,int)):
                st.write(f"- **Fib partials:** TP1={fib_tp1 if fib_tp1 is not None else '—'} • TP2={fib_tp2 if fib_tp2 is not None else '—'}")
            st.write("- **Fail-safe exit:** if price loses VWAP and MACD histogram turns against you, flatten remainder.")
            st.warning("Analytics tool only — always position-size and respect stops.")
        else:
            st.info("No clean confluence signal right now (or time-of-day filter blocking).")

        with st.expander("Diagnostics"):
            st.json(sig.extras)

    else:
        st.info("Add your watchlist in the sidebar, then click **Scan Watchlist** or enable auto-refresh.")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
