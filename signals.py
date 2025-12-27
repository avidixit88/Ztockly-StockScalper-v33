from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from indicators import (
    vwap as calc_vwap,
    session_vwap as calc_session_vwap,
    atr as calc_atr,
    ema as calc_ema,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    find_breaker_block,
    in_zone,
)
from sessions import classify_session, classify_liquidity_phase


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100 (calibrated)
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/PREMARKET/AFTERHOURS/OFF
    extras: Dict[str, Any]


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def _fib_retracement_levels(hi: float, lo: float) -> List[Tuple[str, float]]:
    ratios = [0.382, 0.5, 0.618, 0.786]
    rng = hi - lo
    if rng <= 0:
        return []
    # "pullback" levels for an up-move: hi - r*(hi-lo)
    return [(f"Fib {r:g}", hi - r * rng) for r in ratios]


def _fib_extensions(hi: float, lo: float) -> List[Tuple[str, float]]:
    # extensions above hi for longs, below lo for shorts (we'll mirror in logic)
    ratios = [1.0, 1.272, 1.618]
    rng = hi - lo
    if rng <= 0:
        return []
    return [(f"Ext {r:g}", hi + (r - 1.0) * rng) for r in ratios]


def _closest_level(price: float, levels: List[Tuple[str, float]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not levels:
        return None, None, None
    name, lvl = min(levels, key=lambda x: abs(price - x[1]))
    return name, float(lvl), float(abs(price - lvl))


def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    lookback_bars: int = 180,
    vwap_logic: str = "session",
    fib_lookback_bars: int = 120,
    htf_bias: Optional[Dict[str, object]] = None,   # {bias, score, details}
    htf_strict: bool = False,
    target_atr_pct: float = 0.003,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = ohlcv.copy().tail(int(lookback_bars)).copy()
    df["vwap_cum"] = calc_vwap(df)
    df["vwap_sess"] = calc_session_vwap(df)
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap_use = df["vwap_sess"] if vwap_logic == "session" else df["vwap_cum"]

    last_ts = df.index[-1]
    session = classify_session(last_ts)
    phase = classify_liquidity_phase(last_ts)

    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
        or (session == "PREMARKET" and allow_premarket)
        or (session == "AFTERHOURS" and allow_afterhours)
    )
    last_price = float(close.iloc[-1])

    atr_last = float(df["atr14"].iloc[-1]) if np.isfinite(df["atr14"].iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0
    atr_pct = (atr_last / last_price) if last_price else 0.0

    # Liquidity weighting (premarket/afterhours discounted)
    liquidity_mult = 1.0
    if phase in ("PREMARKET", "AFTERHOURS"):
        liquidity_mult = 0.75

    extras: Dict[str, Any] = {
        "vwap_logic": vwap_logic,
        "vwap_session": float(df["vwap_sess"].iloc[-1]) if np.isfinite(df["vwap_sess"].iloc[-1]) else None,
        "vwap_cumulative": float(df["vwap_cum"].iloc[-1]) if np.isfinite(df["vwap_cum"].iloc[-1]) else None,
        "ema20": float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None,
        "ema50": float(df["ema50"].iloc[-1]) if np.isfinite(df["ema50"].iloc[-1]) else None,
        "atr14": atr_last,
        "atr_pct": atr_pct,
        "liquidity_phase": phase,
        "liquidity_mult": liquidity_mult,
        "fib_lookback_bars": int(fib_lookback_bars),
        "htf_bias": htf_bias,
        "htf_strict": bool(htf_strict),
        "target_atr_pct": float(target_atr_pct),
    }

    if not allowed:
        return SignalResult(symbol, "NEUTRAL", 0, f"Filtered by time-of-day ({session})", None, None, None, None, last_price, last_ts, session, extras)

    # VWAP event
    was_below_vwap = (close.shift(3) < vwap_use.shift(3)).iloc[-1] or (close.shift(5) < vwap_use.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap_use.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap_use.shift(3)).iloc[-1] or (close.shift(5) > vwap_use.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap_use.shift(1).iloc[-1])

    # RSI + MACD events
    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    # Volume confirmation (liquidity weighted)
    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Swings
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    # Trend context (EMA)
    trend_long_ok = bool((close.iloc[-1] >= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] >= df["ema50"].iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] <= df["ema50"].iloc[-1]))
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    # Fib context (scoring + fib-anchored take profits)
    seg = df.tail(int(min(max(60, fib_lookback_bars), len(df))))
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    rng = hi - lo

    fib_name = fib_level = fib_dist = None
    fib_near_long = fib_near_short = False
    fib_bias = "range"
    retr = _fib_retracement_levels(hi, lo) if rng > 0 else []
    fib_name, fib_level, fib_dist = _closest_level(last_price, retr)

    if rng > 0:
        pos = (last_price - lo) / rng
        if pos >= 0.60:
            fib_bias = "up"
        elif pos <= 0.40:
            fib_bias = "down"
        else:
            fib_bias = "range"

    if fib_level is not None and fib_dist is not None:
        near = fib_dist <= max(buffer, 0.0) if atr_last else (fib_dist <= (0.002 * last_price))
        if near:
            if fib_bias == "up":
                fib_near_long = True
            elif fib_bias == "down":
                fib_near_short = True

    extras["fib_hi"] = hi if rng > 0 else None
    extras["fib_lo"] = lo if rng > 0 else None
    extras["fib_bias"] = fib_bias
    extras["fib_closest"] = {"name": fib_name, "level": fib_level, "dist": fib_dist}
    extras["fib_near_long"] = fib_near_long
    extras["fib_near_short"] = fib_near_short

    # Liquidity sweeps
    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())
    bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low))
    bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high))
    extras["bull_liquidity_sweep"] = bull_sweep
    extras["bear_liquidity_sweep"] = bear_sweep

    # FVG + OB + Breaker
    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, df["atr14"], side="bull", lookback=35)
    ob_bear = find_order_block(df, df["atr14"], side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear
    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    brk_bull = find_breaker_block(df, df["atr14"], side="bull", lookback=60)
    brk_bear = find_breaker_block(df, df["atr14"], side="bear", lookback=60)
    extras["bull_breaker"] = brk_bull
    extras["bear_breaker"] = brk_bear
    bull_breaker_retest = bool(brk_bull[0] is not None and in_zone(last_price, brk_bull[0], brk_bull[1], buffer=buffer))
    bear_breaker_retest = bool(brk_bear[0] is not None and in_zone(last_price, brk_bear[0], brk_bear[1], buffer=buffer))
    extras["bull_breaker_retest"] = bull_breaker_retest
    extras["bear_breaker_retest"] = bear_breaker_retest

    displacement = bool(atr_last and float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 1.5 * atr_last)
    extras["displacement"] = displacement

    # HTF bias overlay
    htf_b = None
    if isinstance(htf_bias, dict):
        htf_b = htf_bias.get("bias")
    extras["htf_bias_value"] = htf_b

    # --- Scoring (raw) ---
    long_points = 0
    long_reasons: List[str] = []
    if was_below_vwap and reclaim_vwap:
        long_points += 35; long_reasons.append(f"VWAP reclaim ({vwap_logic})")
    if rsi_snap and rsi14 < 60:
        long_points += 20; long_reasons.append("RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        long_points += 20; long_reasons.append("MACD hist turning up")
    if vol_ok:
        long_points += int(round(15 * liquidity_mult)); long_reasons.append("Volume confirmation")
    if df["low"].tail(12).iloc[-1] > df["low"].tail(12).min():
        long_points += 10; long_reasons.append("Higher-low micro structure")

    short_points = 0
    short_reasons: List[str] = []
    if was_above_vwap and reject_vwap:
        short_points += 35; short_reasons.append(f"VWAP rejection ({vwap_logic})")
    if rsi_downshift and rsi14 > 40:
        short_points += 20; short_reasons.append("RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        short_points += 20; short_reasons.append("MACD hist turning down")
    if vol_ok:
        short_points += int(round(15 * liquidity_mult)); short_reasons.append("Volume confirmation")
    if df["high"].tail(12).iloc[-1] < df["high"].tail(12).max():
        short_points += 10; short_reasons.append("Lower-high micro structure")

    # Fib scoring
    if fib_near_long and fib_name is not None:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        long_points += add
        long_reasons.append(f"Near {fib_name}")
    if fib_near_short and fib_name is not None:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        short_points += add
        short_reasons.append(f"Near {fib_name}")

    # Pro structure scoring
    if pro_mode:
        if bull_sweep:
            long_points += int(round(20 * liquidity_mult)); long_reasons.append("Liquidity sweep (low)")
        if bear_sweep:
            short_points += int(round(20 * liquidity_mult)); short_reasons.append("Liquidity sweep (high)")
        if bull_ob_retest:
            long_points += 15; long_reasons.append("Bullish order block retest")
        if bear_ob_retest:
            short_points += 15; short_reasons.append("Bearish order block retest")
        if bull_fvg is not None:
            long_points += 10; long_reasons.append("Bullish FVG present")
        if bear_fvg is not None:
            short_points += 10; short_reasons.append("Bearish FVG present")
        if bull_breaker_retest:
            long_points += 20; long_reasons.append("Bullish breaker retest")
        if bear_breaker_retest:
            short_points += 20; short_reasons.append("Bearish breaker retest")
        if displacement:
            long_points += 5; short_points += 5

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # HTF overlay scoring
    if htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL":
            long_points += 10; long_reasons.append("HTF bias bullish")
            short_points = max(0, short_points - 10)
        elif htf_b == "BEAR":
            short_points += 10; short_reasons.append("HTF bias bearish")
            long_points = max(0, long_points - 10)

    # Requirements
    if int(cfg["require_vwap_event"]) == 1:
        if not ((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap)):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_rsi_event"]) == 1 and not (rsi_snap or rsi_downshift):
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_macd_turn"]) == 1 and not (macd_turn_up or macd_turn_down):
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_volume"]) == 1 and not vol_ok:
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No volume confirmation", None, None, None, None, last_price, last_ts, session, extras)

    if pro_mode:
        if not (bull_sweep or bear_sweep or bull_ob_retest or bear_ob_retest or bull_breaker_retest or bear_breaker_retest):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "Pro mode: no sweep / OB / breaker trigger", None, None, None, None, last_price, last_ts, session, extras)

    # HTF strict filter (optional)
    if htf_strict and htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL" and not (was_below_vwap and reclaim_vwap):
            # only allow longs if bullish HTF and long setup
            pass
        if htf_b == "BEAR" and not (was_above_vwap and reject_vwap):
            pass

    # ATR-normalized score calibration (per ticker)
    # Scale raw points so high-vol tickers don't always dominate.
    scale = 1.0
    if atr_pct and target_atr_pct:
        scale = target_atr_pct / atr_pct
        scale = float(np.clip(scale, 0.75, 1.25))
    extras["atr_score_scale"] = scale

    long_points_cal = int(round(long_points * scale))
    short_points_cal = int(round(short_points * scale))
    extras["long_points_raw"] = long_points
    extras["short_points_raw"] = short_points
    extras["long_points_cal"] = long_points_cal
    extras["short_points_cal"] = short_points_cal

    min_score = int(cfg["min_actionable_score"])

    # Entry/stop + targets
    def _fib_take_profits_long(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        exts = _fib_extensions(hi, lo)
        # Partial at recent high if above entry, else at ext 1.272
        tp1 = hi if entry_px < hi else next((lvl for _, lvl in exts if lvl > entry_px), None)
        tp2 = next((lvl for _, lvl in exts if lvl and tp1 and lvl > tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _fib_take_profits_short(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        # Mirror extensions below lo
        ratios = [1.0, 1.272, 1.618]
        exts_dn = [ (f"Ext -{r:g}", lo - (r - 1.0) * rng) for r in ratios ]
        tp1 = lo if entry_px > lo else next((lvl for _, lvl in exts_dn if lvl < entry_px), None)
        tp2 = next((lvl for _, lvl in exts_dn if lvl and tp1 and lvl < tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _long_entry_stop(entry_px: float):
        stop_px = float(min(recent_swing_low, entry_px - max(atr_last, 0.0) * 0.8))
        if bull_breaker_retest and brk_bull[0] is not None:
            stop_px = float(min(stop_px, brk_bull[0] - buffer))
        if fib_near_long and fib_level is not None:
            stop_px = float(min(stop_px, fib_level - buffer))
        return entry_px, stop_px

    def _short_entry_stop(entry_px: float):
        stop_px = float(max(recent_swing_high, entry_px + max(atr_last, 0.0) * 0.8))
        if bear_breaker_retest and brk_bear[1] is not None:
            stop_px = float(max(stop_px, brk_bear[1] + buffer))
        if fib_near_short and fib_level is not None:
            stop_px = float(max(stop_px, fib_level + buffer))
        return entry_px, stop_px

    # Decide
    if long_points_cal >= min_score and long_points_cal > short_points_cal:
        entry, stop = _long_entry_stop(last_price)
        risk = max(entry - stop, 0.01)
        t1 = entry + risk
        t2 = entry + 2 * risk
        fib_tp1, fib_tp2 = _fib_take_profits_long(entry)
        extras["fib_tp1"] = fib_tp1
        extras["fib_tp2"] = fib_tp2
        return SignalResult(symbol, "LONG", min(100, int(long_points_cal)), ", ".join(long_reasons[:12]), entry, stop, t1, t2, last_price, last_ts, session, extras)

    if short_points_cal >= min_score and short_points_cal > long_points_cal:
        entry, stop = _short_entry_stop(last_price)
        risk = max(stop - entry, 0.01)
        t1 = entry - risk
        t2 = entry - 2 * risk
        fib_tp1, fib_tp2 = _fib_take_profits_short(entry)
        extras["fib_tp1"] = fib_tp1
        extras["fib_tp2"] = fib_tp2
        return SignalResult(symbol, "SHORT", min(100, int(short_points_cal)), ", ".join(short_reasons[:12]), entry, stop, t1, t2, last_price, last_ts, session, extras)

    reason = f"LongScore={long_points_cal} (raw {long_points}); ShortScore={short_points_cal} (raw {short_points})"
    return SignalResult(symbol, "NEUTRAL", int(max(long_points_cal, short_points_cal)), reason, None, None, None, None, last_price, last_ts, session, extras)
