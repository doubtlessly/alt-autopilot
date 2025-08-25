from __future__ import annotations
import time
from typing import List, Dict
from . import config as C
from .log import get_logger
from .models import Signal
from .writer import write_json, now_iso
from .exchanges import init_exchange, list_spot_usdt, fetch_tickers_safe, quote_volume_usd
from .fetch import fetch_ohlcv_safe
from .filters import TAFeatures
from .scoring import confidence

log = get_logger()

def run() -> None:
    ex = init_exchange(C.EXCHANGE_ID)
    symbols_all = list_spot_usdt(ex, C.QUOTE)

    tickers = fetch_tickers_safe(ex)
    vol_rows = []
    for s in symbols_all:
        qv = quote_volume_usd(tickers.get(s))
        if qv > 0:
            vol_rows.append((s, qv))
    vol_rows.sort(key=lambda x: x[1], reverse=True)
    universe = [s for s,_ in vol_rows[:C.TOP_N_BY_VOL]]

    # BTC 4h for RS baseline
    btc_sym = "BTC/USDT" if "BTC/USDT" in ex.symbols else universe[0]
    df_btc4 = fetch_ohlcv_safe(ex, btc_sym, "4h", C.BARS_4H)

    signals: List[Dict] = []
    watch:   List[Dict] = []
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
            df4  = fetch_ohlcv_safe(ex, sym, "4h",  C.BARS_4H)
            df1  = fetch_ohlcv_safe(ex, sym, "1h",  C.BARS_1H)
            df15 = fetch_ohlcv_safe(ex, sym, "15m", C.BARS_15M)
            df1d = fetch_ohlcv_safe(ex, sym, "1d",  C.BARS_1D)   # daily context
            if min(len(df4), len(df1), len(df15), len(df1d)) < 60:
                continue
            stats["scanned"] += 1

            feats = TAFeatures(df4, df1, df15, df1d, df_btc4)

            atr_ok = feats.atr_ok()
            if not atr_ok:
                stats["fail_atr"] += 1; continue

            structure_ok, structure = feats.structure_ok()
            if not structure_ok:
                stats["fail_structure"] += 1; continue

            expansion_ok = feats.expansion_ok()
            if not expansion_ok:
                stats["fail_expansion"] += 1

            trig_ok, entry_type = feats.trigger_ok()
            if not trig_ok:
                stats["fail_trigger"] += 1

            inval = round(feats.invalidation(), 8)
            entry = round(float(df15["c"].iloc[-1]), 8)
            prh   = round(float(feats.prh), 8)
            atrp  = round(float(df1["atr_pct"].iloc[-1]), 3)

            conf = confidence(structure, expansion_ok, trig_ok, atr_ok)
            record = {
                "symbol": sym.replace("/", "-"),
                "entry_type": entry_type,
                "entry": entry,
                "stop": inval,
                "atr_pct_1h": atrp,
                "range_high_1h": prh,
                "structure": structure,
                "confidence": conf,
                "timeframe": "15m trigger / 1h momentum / 4h+1d context",
                "updated_at": now_iso()
            }

            if trig_ok and conf >= C.MIN_CONFIDENCE:
                signals.append(record); stats["passed_signals"] += 1
            else:
                close_1h = float(df1["c"].iloc[-1])
                near_prh = (prh > 0) and (abs(close_1h - prh) / prh <= C.NEAR_PCT)
                if near_prh or (close_1h >= float(df1["ema20"].iloc[-1])):
                    record["arm_level"] = prh
                    watch.append(record); stats["passed_watch"] += 1

            time.sleep(getattr(ex, "rateLimit", 400)/1000.0)
        except Exception:
            continue

    # Order, cap, write
    signals.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
    watch.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
    signals = signals[:C.MAX_SIGNALS]
    watch   = watch[:20]

    write_json(C.OUT_SIGNALS, {"updated_at": now_iso(), "count": len(signals), "signals": signals})
    write_json(C.OUT_WATCH,   {"updated_at": now_iso(), "count": len(watch),   "watch":   watch})
    write_json(C.OUT_STATUS,  {
        "updated_at": now_iso(),
        "config": {
            "top_n_by_vol": C.TOP_N_BY_VOL, "atr_band": C.ATR_BAND,
            "min_confidence": C.MIN_CONFIDENCE, "near_pct": C.NEAR_PCT,
            "rs_lookback_4h": C.RS_LOOKBACK_4H, "rs_edge": C.RS_EDGE
        },
        "stats": stats
    })
    log.info(f"Wrote {C.OUT_SIGNALS} ({len(signals)})  {C.OUT_WATCH} ({len(watch)})")

if __name__ == "__main__":
    run()
