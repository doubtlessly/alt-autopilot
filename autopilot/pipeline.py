from __future__ import annotations
import time
from typing import List, Dict
from . import config as C
from .log import get_logger
from .models import Signal, EnhancedConfig
from .writer import write_json, now_iso
from .exchanges import init_exchange, list_spot_usdt, fetch_tickers_safe, quote_volume_usd
from .fetch import fetch_ohlcv_safe
from .filters import TAFeatures
from .scoring import confidence, get_signal_quality_tier
import pandas as pd

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
        "fail_market_regime": 0, "fail_volume": 0, "fail_rsi": 0,
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

            # v1.1 Upgrade: Market Regime Gate (FIRST CHECK)
            regime_ok, regime_type = feats.market_regime_ok()
            if not regime_ok:
                stats["fail_market_regime"] += 1
                # In weak regimes, only allow RS leaders to watch, not signals
                if regime_type == "weak_rs_only":
                    # Check if this is a relative strength leader
                    structure_ok, structure = feats.structure_ok()
                    if structure_ok and "rs" in structure.lower():
                        # Allow to watch but not as signal
                        close_1h = float(df1["c"].iloc[-1])
                        near_prh = (feats.prh > 0) and (abs(close_1h - feats.prh) / feats.prh <= C.NEAR_PCT)
                        if near_prh:
                            record = _create_watch_record(sym, feats, df1, df15, regime_type, False, "no_confirmation", True)
                            watch.append(record)
                            stats["passed_watch"] += 1
                continue

            # Standard technical filters
            atr_ok = feats.atr_ok()
            if not atr_ok:
                stats["fail_atr"] += 1; continue

            structure_ok, structure = feats.structure_ok()
            if not structure_ok:
                stats["fail_structure"] += 1; continue

            expansion_ok = feats.expansion_ok()
            if not expansion_ok:
                stats["fail_expansion"] += 1

            # v1.1 Upgrade: Enhanced trigger validation
            trig_ok, entry_type = feats.trigger_ok()
            if not trig_ok:
                stats["fail_trigger"] += 1

            # v1.1 Upgrade: Volume surge detection
            volume_surge = feats.volume_surge_ok()
            if not volume_surge:
                stats["fail_volume"] += 1

            # v1.1 Upgrade: RSI divergence check
            rsi_divergence = feats.rsi_divergence_ok()
            if not rsi_divergence:
                stats["fail_rsi"] += 1

            # v1.1 Upgrade: Structural stop loss
            inval = round(feats.invalidation(), 8)
            entry = round(float(df15["c"].iloc[-1]), 8)
            prh   = round(float(feats.prh), 8)
            atrp  = round(float(df1["atr_pct"].iloc[-1]), 3)

            # v1.1 Upgrade: Enhanced confidence scoring
            conf = confidence(structure, expansion_ok, trig_ok, atr_ok,
                           regime_type, volume_surge, entry_type, rsi_divergence)
            
            # Create enhanced record with v1.1 metadata
            record = _create_signal_record(sym, feats, df1, df15, regime_type, 
                                         volume_surge, entry_type, rsi_divergence, conf)

            if trig_ok and conf >= C.MIN_CONFIDENCE:
                signals.append(record); stats["passed_signals"] += 1
            else:
                # Watch logic: near PRH or above EMA20
                close_1h = float(df1["c"].iloc[-1])
                near_prh = (prh > 0) and (abs(close_1h - prh) / prh <= C.NEAR_PCT)
                above_ema20 = close_1h >= float(df1["ema20"].iloc[-1])
                
                if near_prh or above_ema20:
                    record["arm_level"] = prh
                    watch.append(record); stats["passed_watch"] += 1

            time.sleep(getattr(ex, "rateLimit", 400)/1000.0)
        except Exception as e:
            log.warning(f"Error processing {sym}: {e}")
            continue

    # Order, cap, write
    signals.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
    watch.sort(key=lambda x: (x["confidence"], x["updated_at"]), reverse=True)
    signals = signals[:C.MAX_SIGNALS]
    watch   = watch[:20]

    # v1.1 Upgrade: Enhanced configuration tracking
    enhanced_config: EnhancedConfig = {
        "top_n_by_vol": C.TOP_N_BY_VOL,
        "atr_band": C.ATR_BAND,
        "min_confidence": C.MIN_CONFIDENCE,
        "near_pct": C.NEAR_PCT,
        "rs_lookback_4h": C.RS_LOOKBACK_4H,
        "rs_edge": C.RS_EDGE,
        "donchian_lookback": C.DONCHIAN_LOOKBACK,
        "min_trend_strength": C.MIN_TREND_STRENGTH,
        "volume_surge_threshold": C.VOLUME_SURGE_THRESHOLD,
        "breakout_confirmation_bars": C.BREAKOUT_CONFIRMATION_BARS,
        "retest_threshold": C.RETEST_THRESHOLD,
        "stop_atr_multiplier": C.STOP_ATR_MULTIPLIER,
        "rsi_divergence_lookback": C.RSI_DIVERGENCE_LOOKBACK
    }

    write_json(C.OUT_SIGNALS, {"updated_at": now_iso(), "count": len(signals), "signals": signals})
    write_json(C.OUT_WATCH,   {"updated_at": now_iso(), "count": len(watch),   "watch":   watch})
    write_json(C.OUT_STATUS,  {
        "updated_at": now_iso(),
        "config": enhanced_config,
        "stats": stats
    })
    
    log.info(f"v1.1 Pipeline Complete: {C.OUT_SIGNALS} ({len(signals)})  {C.OUT_WATCH} ({len(watch)})")
    log.info(f"Market Regime Stats: {stats['fail_market_regime']} rejected, {stats['passed_signals']} signals, {stats['passed_watch']} watch")

def _create_signal_record(sym: str, feats: TAFeatures, df1: pd.DataFrame, df15: pd.DataFrame,
                         regime_type: str, volume_surge: bool, entry_type: str, 
                         rsi_divergence: bool, confidence: int) -> Dict:
    """Create enhanced signal record with v1.1 metadata and enhanced features"""
    entry = round(float(df15["c"].iloc[-1]), 8)
    inval = round(feats.invalidation(), 8)
    prh = round(float(feats.prh), 8)
    atrp = round(float(df1["atr_pct"].iloc[-1]), 3)
    
    return {
        "symbol": sym.replace("/", "-"),
        "entry_type": entry_type,
        "entry": entry,
        "stop": inval,
        "atr_pct_1h": atrp,
        "range_high_1h": prh,
        "structure": "4h-uptrend" if "4h-uptrend" in entry_type else "range-high-reclaim" if "reclaim" in entry_type else "flat-accept-rs",
        "confidence": confidence,
        "timeframe": "15m trigger / 1h momentum / 4h+1d context",
        "updated_at": now_iso(),
        # v1.1 Enhanced metadata
        "market_regime": regime_type,
        "volume_surge": volume_surge,
        "breakout_confirmation": entry_type,
        "rsi_divergence": rsi_divergence,
        # Enhanced Feature Engineering for AI Consumption
        "technical_features": feats.get_enhanced_features()
    }

def _create_watch_record(sym: str, feats: TAFeatures, df1: pd.DataFrame, df15: pd.DataFrame,
                        regime_type: str, volume_surge: bool, entry_type: str, 
                        rsi_divergence: bool) -> Dict:
    """Create watch record for near-trigger opportunities with enhanced features"""
    entry = round(float(df15["c"].iloc[-1]), 8)
    inval = round(feats.invalidation(), 8)
    prh = round(float(feats.prh), 8)
    atrp = round(float(df1["atr_pct"].iloc[-1]), 3)
    
    return {
        "symbol": sym.replace("/", "-"),
        "entry_type": entry_type,
        "entry": entry,
        "stop": inval,
        "atr_pct_1h": atrp,
        "range_high_1h": prh,
        "structure": "weak_rs_only",
        "confidence": 15,  # Base score for weak regime
        "timeframe": "15m trigger / 1h momentum / 4h+1d context",
        "updated_at": now_iso(),
        "market_regime": regime_type,
        "volume_surge": volume_surge,
        "breakout_confirmation": entry_type,
        "rsi_divergence": rsi_divergence,
        # Enhanced Feature Engineering for AI Consumption
        "technical_features": feats.get_enhanced_features(),
        "arm_level": prh
    }

if __name__ == "__main__":
    run()
