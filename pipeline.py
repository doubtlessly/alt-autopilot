# -*- coding: utf-8 -*-
"""
Generates high-probability ALTCOIN spot buy signals (entry + stop) and writes docs/signals.json
- Exchange: KuCoin via ccxt (public only)
- Cadence: schedule via GitHub Actions (see .github/workflows/scan.yml)
- No liquidity/spread checks. We keep only strong TA setups.
"""
import json, time, math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt

# ---------- Config ----------
EXCHANGE_ID = "kucoin"
QUOTE = "USDT"
TOP_N_BY_VOL = 60                # scan top-N by 24h quote volume
BARS_4H = 150
BARS_1H = 150
BARS_15M = 150
ATR_BAND = (3.0, 15.0)          # 1h ATR%(14) must be inside this band
MAX_SIGNALS = 10                # write at most this many signals
WRITE_CANDLES = False           # set True to also write docs/candles/{symbol}/{tf}.json
OUT_PATH = "docs/signals.json"

# ---------- Helpers ----------
def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def to_df(ohlcv):
    # ohlcv: [ts, o, h, l, c, v]
    cols = ["t","o","h","l","c","v"]
    df = pd.DataFrame(ohlcv, columns=cols)
    # KuCoin returns ms timestamps
    if df["t"].max() > 10**12:
        df["t"] = (df["t"] // 1000).astype(int)
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    # df with columns h,l,c
    high, low, close = df["h"], df["l"], df["c"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def obv_proxy(df):
    # simple OBV proxy on close changes
    dirv = np.sign(df["c"].diff().fillna(0)) * df["v"]
    return dirv.cumsum()

def prior_range_high_1h(df_1h, lookback=48):
    # max high of the last `lookback` bars, excluding the most recent bar
    if len(df_1h) < lookback + 2:
        lookback = max(10, min(lookback, len(df_1h)-2))
    return df_1h["h"].iloc[-(lookback+1):-1].max()

def fetch_ohlcv_safe(ex, symbol, tf, limit):
    for i in range(2):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        except Exception:
            time.sleep(1.2 * (i+1))
    return []

def get_quote_volume_usd(t):
    """
    Try to read 24h quote volume from a ccxt ticker record.
    KuCoin exposes info['volValue'] (string).
    """
    if t is None:
        return 0.0
    if "quoteVolume" in t and t["quoteVolume"]:
        return float(t["quoteVolume"])
    info = t.get("info", {}) if isinstance(t, dict) else {}
    v = info.get("volValue") or info.get("quoteVolume") or info.get("volValue24h")
    try:
        return float(v)
    except Exception:
        return 0.0

# ---------- Exchange ----------
ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
ex.load_markets()

# Symbols ending with /USDT and spot only
symbols = [s for s in ex.symbols if s.endswith(f"/{QUOTE}") and ex.markets.get(s, {}).get("spot")]

# Rank by 24h quote volume
tickers = {}
try:
    tickers = ex.fetch_tickers()
except Exception:
    # fallback: fetch per-symbol (slower); cap to first 120 to be kind
    for s in symbols[:120]:
        try:
            tickers[s] = ex.fetch_ticker(s)
            time.sleep(ex.rateLimit/1000.0)
        except Exception:
            pass

vol_rows = []
for s in symbols:
    qv = get_quote_volume_usd(tickers.get(s))
    if qv > 0:
        vol_rows.append((s, qv))
vol_rows.sort(key=lambda x: x[1], reverse=True)
universe = [s for s,_ in vol_rows[:TOP_N_BY_VOL]]

signals = []
for sym in universe:
    try:
        o4 = fetch_ohlcv_safe(ex, sym, "4h", BARS_4H)
        o1 = fetch_ohlcv_safe(ex, sym, "1h", BARS_1H)
        o15 = fetch_ohlcv_safe(ex, sym, "15m", BARS_15M)
        if len(o4) < 60 or len(o1) < 60 or len(o15) < 40:
            continue

        df4 = to_df(o4)
        df1 = to_df(o1)
        df15 = to_df(o15)

        # Indicators
        df4["ema20"], df4["ema50"] = ema(df4["c"],20), ema(df4["c"],50)
        df1["ema20"], df1["ema50"] = ema(df1["c"],20), ema(df1["c"],50)
        df1["atr14"] = atr(df1,14)
        df1["atr_pct"] = df1["atr14"]/df1["c"]*100
        df1["obv"] = obv_proxy(df1)

        # Conditions
        atr_ok = ATR_BAND[0] <= float(df1["atr_pct"].iloc[-1]) <= ATR_BAND[1]

        ema_up_4h = (df4["ema20"].iloc[-1] > df4["ema50"].iloc[-1]) and (df4["ema20"].iloc[-1] > df4["ema20"].iloc[-5])
        # Range-high reclaim alternative
        prh = prior_range_high_1h(df1, lookback=48)
        reclaim_ok = df1["c"].iloc[-1] > prh and df1["c"].iloc[-2] <= prh

        structure_ok = ema_up_4h or reclaim_ok

        # 1h expansion: close above range high + OBV rising
        obv_up = df1["obv"].iloc[-1] > df1["obv"].iloc[-5]
        expansion_ok = (df1["c"].iloc[-1] > prh) and obv_up

        # 15m trigger: breakout close OR retest-and-hold
        last15_close = df15["c"].iloc[-1]
        last15_low   = df15["l"].iloc[-1]
        breakout = last15_close > prh
        retest_hold = (last15_low <= prh) and (last15_close > prh)
        trigger_ok = breakout or retest_hold

        if not (atr_ok and structure_ok and expansion_ok and trigger_ok):
            continue

        # Invalidation: last swing low of 15m (min of last 8 bars)
        inval = float(df15["l"].iloc[-8:].min())
        entry = float(last15_close)

        # Confidence score (0-100)
        score = 0
        score += 30 if ema_up_4h else 20 if reclaim_ok else 0
        score += 25 if expansion_ok else 0
        score += 25 if trigger_ok else 0
        score += 20 if atr_ok else 0
        score = int(max(0, min(100, score)))

        signals.append({
            "symbol": sym.replace("/", "-"),
            "entry_type": "breakout" if breakout else "retest-hold",
            "entry": round(entry, 8),
            "stop": round(inval, 8),
            "atr_pct_1h": round(float(df1["atr_pct"].iloc[-1]), 3),
            "range_high_1h": round(float(prh), 8),
            "structure": "4h-uptrend" if ema_up_4h else "range-high-reclaim",
            "confidence": score,
            "timeframe": "15m trigger / 1h momentum / 4h structure",
            "updated_at": ts()
        })

        # Optional: write compact candles for transparency
        if WRITE_CANDLES:
            for tf, df in [("4h", df4), ("1h", df1), ("15m", df15)]:
                cpath = f"docs/candles/{sym.replace('/','-')}/{tf}.json"
                df_tail = df[["t","o","h","l","c","v"]].tail(150)
                payload = {"columns":["t","o","h","l","c","v"], "rows": df_tail.values.tolist()}
                Path = __import__("pathlib").Path
                Path(cpath).parent.mkdir(parents=True, exist_ok=True)
                with open(cpath, "w") as f:
                    json.dump(payload, f)

        # be nice to API
        time.sleep(ex.rateLimit/1000.0)
    except Exception:
        # skip symbol on any error
        continue

# pick top signals by confidence (and tiebreak by latest)
signals.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
signals = signals[:MAX_SIGNALS]

out = {"updated_at": ts(), "count": len(signals), "signals": signals}
# ensure /docs exists locally
import os
os.makedirs("docs", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT_PATH} with {len(signals)} signals at {out['updated_at']}")
