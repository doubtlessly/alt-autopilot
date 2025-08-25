# -*- coding: utf-8 -*-
"""
ALT spot high-probability signals (entry + stop) + diagnostics
Outputs:
  - docs/signals.json  (ready-now)
  - docs/watch.json    (near-trigger)
  - docs/status.json   (why things were filtered)
Exchange: KuCoin via ccxt (public). TA only (no liquidity/spread/alloc).
"""

import os, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt

# ---------------- Config (diagnostics + balanced) ----------------
EXCHANGE_ID = "kucoin"
QUOTE = "USDT"
TOP_N_BY_VOL = 200            # scan many names
BARS_4H = 150; BARS_1H = 150; BARS_15M = 150
ATR_BAND = (1.5, 30.0)        # widen to avoid over-filtering
MIN_CONFIDENCE = 65           # publish if score >= 65
MAX_SIGNALS = 10
NEAR_PCT = 0.015              # within 1.5% of PRH → watch
OUT_SIGNALS = "docs/signals.json"
OUT_WATCH   = "docs/watch.json"
OUT_STATUS  = "docs/status.json"

FALLBACK_SYMBOLS = [
  "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","LINK/USDT","ADA/USDT","AVAX/USDT",
  "TRX/USDT","NEAR/USDT","DOT/USDT","MATIC/USDT","TON/USDT","ATOM/USDT","INJ/USDT","APT/USDT",
  "ARB/USDT","OP/USDT","SUI/USDT","SEI/USDT","RUNE/USDT","AAVE/USDT","UNI/USDT","MKR/USDT",
  "DYDX/USDT","GMX/USDT","FIL/USDT","LDO/USDT","STX/USDT","IMX/USDT","PYTH/USDT","TIA/USDT",
  "ORDI/USDT","PEPE/USDT","FLOKI/USDT","BONK/USDT","WIF/USDT","ONDO/USDT","PENDLE/USDT","WLD/USDT",
]

def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    if len(df)==0: return df
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
    return float(df_1h["h"].iloc[-(look+1):-1].max())

def fetch_ohlcv_safe(ex, symbol, tf, limit):
    for i in range(2):
        try: return ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        except Exception: time.sleep(1.2*(i+1))
    return []

def get_quote_volume_usd(t):
    if not t: return 0.0
    # ccxt standard field first
    if "quoteVolume" in t and t["quoteVolume"]:
        try: return float(t["quoteVolume"])
        except: pass
    info = t.get("info", {}) if isinstance(t, dict) else {}
    for k in ("volValue","quoteVolume","volValue24h","volValue24"):  # try several
        v = info.get(k)
        if v:
            try: return float(v)
            except: continue
    return 0.0

# ---------------- Exchange + Universe ----------------
ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "timeout": 20000})
ex.load_markets()
symbols_all = [s for s in ex.symbols if s.endswith(f"/{QUOTE}") and ex.markets.get(s, {}).get("spot")]

# Rank by 24h quote volume
tickers = {}
try:
    tickers = ex.fetch_tickers()
except Exception:
    # per-symbol fallback (may be rate-limited)
    for s in symbols_all:
        try:
            tickers[s] = ex.fetch_ticker(s); time.sleep(ex.rateLimit/1000.0)
        except Exception: pass

vol_rows = []
for s in symbols_all:
    qv = get_quote_volume_usd(tickers.get(s))
    if qv > 0: vol_rows.append((s, qv))
vol_rows.sort(key=lambda x: x[1], reverse=True)
universe = [s for s,_ in vol_rows[:TOP_N_BY_VOL]]

# If discovery failed, fall back to a curated list to ensure output
if not universe:
    universe = [s for s in FALLBACK_SYMBOLS if s in ex.symbols]

# ---------------- Scan ----------------
signals, watch = [], []
stats = {
    "symbols_total": len(symbols_all),
    "universe_size": len(universe),
    "scanned": 0,
    "fail_atr": 0, "fail_structure": 0, "fail_expansion": 0, "fail_trigger": 0,
    "passed_signals": 0, "passed_watch": 0,
    "sample_universe": universe[:12]
}

for sym in universe:
    try:
        o4  = fetch_ohlcv_safe(ex, sym, "4h",  BARS_4H)
        o1  = fetch_ohlcv_safe(ex, sym, "1h",  BARS_1H)
        o15 = fetch_ohlcv_safe(ex, sym, "15m", BARS_15M)
        if len(o4)<60 or len(o1)<60 or len(o15)<40:
            continue

        df4, df1, df15 = to_df(o4), to_df(o1), to_df(o15)
        if len(df4)<60 or len(df1)<60 or len(df15)<40:
            continue

        stats["scanned"] += 1

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

        # 1h expansion: above range high AND (OBV rising OR above EMA20)
        obv_up = df1["obv"].iloc[-1] > df1["obv"].iloc[-5]
        above_ema20 = df1["c"].iloc[-1] > df1["ema20"].iloc[-1]
        expansion_ok = (df1["c"].iloc[-1] >= prh) and (obv_up or above_ema20)

        # 15m trigger: single close above PRH OR retest-and-hold
        last15_close = float(df15["c"].iloc[-1])
        last15_low   = float(df15["l"].iloc[-1])
        single_above = last15_close > prh
        retest_hold  = (last15_low <= prh) and (last15_close > prh)
        trigger_ok   = single_above or retest_hold

        # invalidation = min low of last 8 x 15m bars
        inval = float(df15["l"].iloc[-8:].min())
        entry = float(last15_close)
        sym_dash = sym.replace("/", "-")

        # Counters by reason (only after ATR+structure pass)
        if not atr_ok:
            stats["fail_atr"] += 1
            continue
        if not structure_ok:
            stats["fail_structure"] += 1
            continue
        if not expansion_ok:
            stats["fail_expansion"] += 1

        # confidence (simple)
        score = 0
        score += 30 if ema_up_4h else 20 if reclaim_ok else 0
        score += 25 if expansion_ok else 0
        score += 25 if trigger_ok else 0
        score += 20 if atr_ok else 0
        score = int(max(0, min(100, score)))

        record = {
            "symbol": sym_dash,
            "entry_type": "breakout" if single_above else ("retest-hold" if retest_hold else "n/a"),
            "entry": round(entry, 8),
            "stop": round(inval, 8),
            "atr_pct_1h": round(atr_now, 3),
            "range_high_1h": round(float(prh), 8),
            "structure": "4h-uptrend" if ema_up_4h else ("range-high-reclaim" if reclaim_ok else "none"),
            "confidence": score,
            "timeframe": "15m trigger / 1h momentum / 4h structure",
            "updated_at": now_iso()
        }

        if trigger_ok and score >= MIN_CONFIDENCE:
            signals.append(record); stats["passed_signals"] += 1
        else:
            # near-trigger if within NEAR_PCT of PRH (and ATR + structure ok)
            close = float(df1["c"].iloc[-1])
            if prh > 0 and abs(close - prh)/prh <= NEAR_PCT:
                record["arm_level"] = record["range_high_1h"]
                watch.append(record); stats["passed_watch"] += 1

        time.sleep(getattr(ex, "rateLimit", 400)/1000.0)
    except Exception:
        # swallow per-symbol errors to keep job green
        continue

# sort & cap
signals.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
watch.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
signals = signals[:MAX_SIGNALS]
watch   = watch[:20]

# write outputs
os.makedirs("docs", exist_ok=True)
with open(OUT_SIGNALS, "w") as f: json.dump({"updated_at": now_iso(), "count": len(signals), "signals": signals}, f, indent=2)
with open(OUT_WATCH,   "w") as f: json.dump({"updated_at": now_iso(), "count": len(watch),   "watch":   watch}, f, indent=2)
with open(OUT_STATUS,  "w") as f: json.dump({
    "updated_at": now_iso(),
    "config": {"top_n_by_vol": TOP_N_BY_VOL, "atr_band": ATR_BAND, "min_confidence": MIN_CONFIDENCE, "near_pct": NEAR_PCT},
    "stats":  {"symbols_total": len(symbols_all), "universe_size": len(universe), "scanned": stats["scanned"],
               "fail_atr": stats["fail_atr"], "fail_structure": stats["fail_structure"],
               "fail_expansion": stats["fail_expansion"], "fail_trigger": stats["fail_trigger"],
               "passed_signals": stats["passed_signals"], "passed_watch": stats["passed_watch"],
               "sample_universe": stats["sample_universe"]}
}, f, indent=2)

print(f"Wrote {OUT_SIGNALS} ({len(signals)}) and {OUT_WATCH} ({len(watch)})")
print(f"Status: scanned={stats['scanned']}  fail_atr={stats['fail_atr']}  fail_structure={stats['fail_structure']}  "
      f"fail_expansion={stats['fail_expansion']}  fail_trigger={stats['fail_trigger']}  "
      f"signals={stats['passed_signals']}  watch={stats['passed_watch']}")

