# Ztockly Scalping Scanner v7

v7 adds:
- **Fib‑anchored partial take profit targets** (TP1 at recent swing high/low, TP2 at fib extension)
- **Liquidity‑weighted scoring** (premarket/afterhours discounted vs RTH)
- **ATR‑normalized score calibration per ticker** (keeps scores comparable across very different vol tickers)
- **Optional higher‑TF bias overlay** (15m/30m)

Still included from v6/v5:
- Session VWAP (ET reset) + cumulative VWAP
- VWAP logic selector for scoring + dual VWAP chart toggle
- Pro mode: liquidity sweeps + order blocks + breaker blocks + FVG + EMA context
- Fib retracement confluence scoring (adds points when price is near key retracement levels)
- In‑app alerts with cooldown

## Run
```bash
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
streamlit run app.py
```

## Notes
- Watchlists > ~10 tickers at 1min can hit Alpha Vantage limits. Keep it tight.
