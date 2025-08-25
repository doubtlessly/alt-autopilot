# -*- coding: utf-8 -*-
"""
Generates high-probability ALTCOIN spot buy signals (entry + stop) and writes:
  - docs/signals.json  (ready now)
  - docs/watch.json    (near-trigger setups)
Design: KuCoin (public), TA only (no liquidity/spread/alloc), 4h/1h/15m.
"""

import json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt

# ---------------- Config (balanced preset) ----------------
EXCHANGE_ID = "kucoin"
QUOTE = "USDT"
TOP_N_BY_VOL = 120           # scan more names to find candidates
BARS_4H = 150; BARS_1H = 150; BARS_15M = 150
ATR_BAND = (2.0, 20.0)       # wider band = more (still sane) candidates
MIN_CONFIDENCE = 70          # only publish if score >= 70
MAX_SIGNALS = 10             # limit final signals
OUT_SIGNALS = "docs/signals.json"
OUT_WATCH   = "docs/watch.json"

# ---------------- Utils ----------------
def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    if df["t"].max() > 10**12: df["t"] = (df["t"] // 1000).astype(int)
    for c in ["o","h","l","c","v"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(), (df["h"]-pc).abs(), (df["l"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def obv_proxy(df): return (np.sign(df["c"].diff().fillna(0))*df["v"]).cumsum()

def prior_range_high_1h(df_1h, min_look=36, max_look=60):
    look = min(max_look, max(min_look, len(df_1h)-2))
    return df_1h["h"].iloc[-(look+1):-1].max()

def fetch_ohlcv_safe(ex, symbol, tf, limit):
    for i in range(2):
        try: return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        except Exception: time.sleep(1.2*(i+1))
    return []

def get_quote_volume_usd(t):
    if not t: return 0.0
    if "quoteVolume" in t and t["quoteVolume"]: return float(t["quoteVolume"])
    info = t.get("info", {}) if isinstance(t, dict) else {}
    v = info.get("volValue") or info.get("quoteVolume") or info.get("volValue24h")
    try: return float(v)
    except Exception: return 0.0

# ---------------- Exchange + Universe ----------------
ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
ex.load_markets()
symbols = [s for s in ex.symbols if s.endswith(f"/{QUOTE}") and ex.markets.get(s, {}).get("spot")]

tickers = {}
try:
    tickers = ex.fetch_tickers()
except Exception:
    for s in symbols[:150]:
        try:
            tickers[s] = ex.fetch_ticker(s); time.sleep(ex.rateLimit/1000.0)
        except Exception: pass

vol_rows = []
for s in symbols:
    qv = get_quote_volume_usd(tickers.get(s))
    if qv > 0: vol_rows.append((s, qv))
vol_rows.sort(key=lambda x: x[1], reverse=True)
universe = [s for s,_ in vol_rows[:TOP_N_BY_VOL]]

# ---------------- Scan ----------------
signals, watch = [], []

for sym in universe:
    try:
        o4  = fetch_ohlcv_safe(ex, sym, "4h",  BARS_4H)
        o1  = fetch_ohlcv_safe(ex, sym, "1h",  BARS_1H)
        o15 = fetch_ohlcv_safe(ex, sym, "15m", BARS_15M)
        if len(o4)<60 or len(o1)<60 or len(o15)<40: continue
        df4, df1, df15 = to_df(o4), to_df(o1), to_df(o15)

        # indicators
        df4["ema20"], df4["ema50"] = ema(df4["c"],20), ema(df4["c"],50)
        df1["ema20"], df1["ema50"] = ema(df1["c"],20), ema(df1["c"],50)
        df1["atr14"] = atr(df1,14)
        df1["atr_pct"] = df1["atr14"]/df1["c"]*100
        df1["obv"] = obv_proxy(df1)

        atr_now = float(df1["atr_pct"].iloc[-1])
        atr_ok = ATR_BAND[0] <= atr_now <= ATR_BAND[1]

        # structure: 4h uptrend OR clean reclaim
        ema_up_4h = (df4["ema20"].iloc[-1] > df4["ema50"].iloc[-1]) and (df4["ema20"].iloc[-1] > df4["ema20"].iloc[-5])
        prh = prior_range_high_1h(df1, 36, 60)
        reclaim_ok = df1["c"].iloc[-1] > prh and df1["c"].iloc[-2] <= prh
        structure_ok = ema_up_4h or reclaim_ok

        # 1h expansion: above range high and either OBV rising or price > EMA20
        obv_up = df1["obv"].iloc[-1] > df1["obv"].iloc[-5]
        above_ema20 = df1["c"].iloc[-1] > df1["ema20"].iloc[-1]
        expansion_ok = (df1["c"].iloc[-1] > prh) and (obv_up or above_ema20)

        # 15m trigger: (a) two closes above prh OR (b) retest-and-hold
        last2_above = (df15["c"].iloc[-1] > prh) and (df15["c"].iloc[-2] > prh)
        retest_hold = (df15["l"].iloc[-1] <= prh) and (df15["c"].iloc[-1] > prh)
        trigger_ok = last2_above or retest_hold

        # invalidation = min low of last 8 x 15m bars
        inval = float(df15["l"].iloc[-8:].min())
        entry = float(df15["c"].iloc[-1])
        sym_dash = sym.replace("/", "-")

        # confidence score
        score = 0
        score += 30 if ema_up_4h else 20 if reclaim_ok else 0
        score += 25 if expansion_ok else 0
        score += 25 if trigger_ok else 0
        score += 20 if atr_ok else 0
        score = int(max(0, min(100, score)))

        record = {
            "symbol": sym_dash,
            "entry_type": "breakout" if last2_above else ("retest-hold" if retest_hold else "n/a"),
            "entry": round(entry, 8),
            "stop": round(inval, 8),
            "atr_pct_1h": round(atr_now, 3),
            "range_high_1h": round(float(prh), 8),
            "structure": "4h-uptrend" if ema_up_4h else ("range-high-reclaim" if reclaim_ok else "none"),
            "confidence": score,
            "timeframe": "15m trigger / 1h momentum / 4h structure",
            "updated_at": now_iso()
        }

        if atr_ok and structure_ok and expansion_ok and trigger_ok and score >= MIN_CONFIDENCE:
            signals.append(record)
        elif atr_ok and structure_ok and (df1["c"].iloc[-1] > prh):  # near-trigger: above PRH but not confirmed
            # show as watch with arm level = prh
            record["arm_level"] = record["range_high_1h"]
            watch.append(record)

        time.sleep(ex.rateLimit/1000.0)
    except Exception:
        continue

signals.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
watch.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)

signals = signals[:MAX_SIGNALS]
out_sig = {"updated_at": now_iso(), "count": len(signals), "signals": signals}
out_wat = {"updated_at": now_iso(), "count": len(watch),   "watch":   watch[:20]}

import os, json
os.makedirs("docs", exist_ok=True)
with open(OUT_SIGNALS, "w") as f: json.dump(out_sig, f, indent=2)
with open(OUT_WATCH,   "w") as f: json.dump(out_wat, f, indent=2)
print(f"Wrote {OUT_SIGNALS} ({out_sig['count']}) and {OUT_WATCH} ({out_wat['count']})")
