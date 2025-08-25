from __future__ import annotations
import pandas as pd
from .ta import (ema, atr, obv_proxy, prior_range_high_1h, slope_up, pct_return,
                 donchian_high, donchian_low, volume_surge, rsi, 
                 detect_bearish_rsi_divergence, structural_stop_loss, breakout_confirmation,
                 calculate_price_momentum, calculate_volume_trend, calculate_volatility_regime,
                 calculate_market_strength, calculate_correlation_with_btc, calculate_trend_quality)
from . import config as C

class TAFeatures:
    def __init__(self, df4: pd.DataFrame, df1: pd.DataFrame, df15: pd.DataFrame, df1d: pd.DataFrame, df_btc4: pd.DataFrame):
        # 4h
        self.df4 = df4.copy()
        self.df4["ema20"] = ema(self.df4["c"], 20)
        self.df4["ema50"] = ema(self.df4["c"], 50)
        
        # 1h
        self.df1 = df1.copy()
        self.df1["ema20"] = ema(self.df1["c"], 20)
        self.df1["ema50"] = ema(self.df1["c"], 50)
        self.df1["atr14"] = atr(self.df1, 14)
        self.df1["atr_pct"] = self.df1["atr14"] / self.df1["c"] * 100
        self.df1["obv"] = obv_proxy(self.df1)
        self.df1["rsi"] = rsi(self.df1, C.RSI_PERIOD)
        
        # 15m
        self.df15 = df15.copy()
        
        # daily
        self.df1d = df1d.copy()
        self.df1d["ema20"] = ema(self.df1d["c"], 20)
        self.df1d["ema50"] = ema(self.df1d["c"], 50)
        self.df1d["donchian_high"] = donchian_high(self.df1d, C.DONCHIAN_LOOKBACK)
        self.df1d["donchian_low"] = donchian_low(self.df1d, C.DONCHIAN_LOOKBACK)
        
        # BTC 4h for RS
        self.df_btc4 = df_btc4.copy()
        self.prh = prior_range_high_1h(self.df1)
        
        # Enhanced Features for AI Consumption
        self._calculate_enhanced_features()

    def _calculate_enhanced_features(self):
        """Calculate all enhanced features for AI consumption"""
        # Price momentum (4H timeframe for trend momentum)
        self.price_momentum = calculate_price_momentum(self.df4, lookback=20)
        
        # Volume trend (1H timeframe for recent volume patterns)
        self.volume_trend = calculate_volume_trend(self.df1, lookback=20)
        
        # Volatility regime based on current ATR
        current_atr_pct = float(self.df1["atr_pct"].iloc[-1])
        self.volatility_regime = calculate_volatility_regime(current_atr_pct)
        
        # Trend quality (4H timeframe for trend analysis)
        self.trend_quality = calculate_trend_quality(self.df4, ema_short=20, ema_long=50)
        
        # Correlation with BTC
        self.correlation_with_btc = calculate_correlation_with_btc(self.df4, self.df_btc4, lookback=20)
        
        # Market strength (combined metric)
        self.market_strength = calculate_market_strength(
            self.trend_quality, 
            self.volume_trend, 
            self.price_momentum
        )

    def get_enhanced_features(self) -> dict:
        """
        Return all enhanced features in AI-friendly format
        All values are normalized 0-1 scale or categorical
        """
        return {
            "price_momentum": self.price_momentum,           # 0-1: How strong is recent momentum
            "volume_trend": self.volume_trend,               # 0-1: How strong is volume trend
            "volatility_regime": self.volatility_regime,     # "low"/"medium"/"high"
            "trend_quality": self.trend_quality,             # 0-1: How clean/strong is trend
            "correlation_with_btc": self.correlation_with_btc,  # 0-1: How correlated with BTC
            "market_strength": self.market_strength          # 0-1: Overall market health
        }

    # v1.1 Upgrades: Market Regime Gates
    def market_regime_ok(self) -> tuple[bool, str]:
        """
        Check if market regime allows signals
        Returns (is_allowed, regime_type)
        """
        # Daily trend: EMA20 >= EMA50
        daily_trend = (self.df1d["ema20"].iloc[-1] >= self.df1d["ema50"].iloc[-1] and
                      slope_up(self.df1d["ema20"], look=5, tol=1.0 - C.MIN_TREND_STRENGTH))
        
        # Daily Donchian reclaim: close back above 20-day high after being below
        if len(self.df1d) >= C.DONCHIAN_LOOKBACK:
            donchian_high_val = float(self.df1d["donchian_high"].iloc[-1])
            current_close = float(self.df1d["c"].iloc[-1])
            previous_close = float(self.df1d["c"].iloc[-2])
            
            # Check if we're reclaiming the Donchian high
            reclaim_ok = (current_close > donchian_high_val and 
                         previous_close <= donchian_high_val)
        else:
            reclaim_ok = False
        
        if daily_trend:
            return True, "trending"
        elif reclaim_ok:
            return True, "reclaim"
        else:
            return False, "weak_rs_only"

    # Gates
    def atr_ok(self) -> bool:
        x = float(self.df1["atr_pct"].iloc[-1])
        return C.ATR_BAND[0] <= x <= C.ATR_BAND[1]

    def structure_ok(self) -> tuple[bool, str]:
        ema_up_4h = (self.df4["ema20"].iloc[-1] > self.df4["ema50"].iloc[-1]) and slope_up(self.df4["ema20"])
        reclaim_ok = self.df1["c"].iloc[-1] > self.prh and self.df1["c"].iloc[-2] <= self.prh
        ema20_flat_up = slope_up(self.df4["ema20"], look=5, tol=0.999)
        above_ema20_1h = self.df1["c"].iloc[-1] > self.df1["ema20"].iloc[-1]
        
        # RS vs BTC 4h
        sym_ret = pct_return(self.df4["c"], C.RS_LOOKBACK_4H)
        btc_ret = pct_return(self.df_btc4["c"], C.RS_LOOKBACK_4H)
        rs_ok = (sym_ret - btc_ret) >= C.RS_EDGE
        
        ok = ema_up_4h or reclaim_ok or (ema20_flat_up and above_ema20_1h and rs_ok)
        which = "4h-uptrend" if ema_up_4h else ("range-high-reclaim" if reclaim_ok else "flat-accept-rs" if ok else "none")
        return ok, which

    def expansion_ok(self) -> bool:
        obv_up = self.df1["obv"].iloc[-1] > self.df1["obv"].iloc[-5]
        above_ema20 = self.df1["c"].iloc[-1] > self.df1["ema20"].iloc[-1]
        return (self.df1["c"].iloc[-1] >= max(self.prh * 0.998, self.df1["ema20"].iloc[-1])) and (obv_up or above_ema20)

    # v1.1 Upgrades: Enhanced Breakout Validation
    def trigger_ok(self) -> tuple[bool, str]:
        """
        Enhanced trigger validation with breakout confirmation
        Returns (is_valid, trigger_type)
        """
        # Basic trigger check
        last_close = float(self.df15["c"].iloc[-1])
        last_low = float(self.df15["l"].iloc[-1])
        single_above = last_close > self.prh
        retest_hold = (last_low <= self.prh) and (last_close > self.prh)
        
        if not (single_above or retest_hold):
            return False, "no_trigger"
        
        # v1.1 Upgrade: Breakout confirmation
        confirmation_ok, confirmation_type = breakout_confirmation(
            self.df15, self.prh, C.BREAKOUT_CONFIRMATION_BARS, 
            C.RETEST_THRESHOLD, C.MIN_RETEST_WICK
        )
        
        if confirmation_ok:
            trigger_type = "breakout" if single_above else "retest-hold"
            return True, f"{trigger_type}_{confirmation_type}"
        else:
            return False, "no_confirmation"

    # v1.1 Upgrades: Volume Surge Detection
    def volume_surge_ok(self) -> bool:
        """Check if recent volume shows surge vs historical median"""
        return volume_surge(
            self.df1, C.VOLUME_LOOKBACK, C.VOLUME_MEDIAN_LOOKBACK, C.VOLUME_SURGE_THRESHOLD
        )

    # v1.1 Upgrades: RSI Divergence Detection
    def rsi_divergence_ok(self) -> bool:
        """Check for bearish RSI divergence (returns True if NO divergence)"""
        return not detect_bearish_rsi_divergence(
            self.df1, self.df1["rsi"], C.RSI_DIVERGENCE_LOOKBACK, C.RSI_DIVERGENCE_MIN_BARS
        )

    # v1.1 Upgrades: Structural Stop Loss
    def invalidation(self) -> float:
        """Calculate structural stop loss based on swing low and ATR"""
        entry_price = float(self.df15["c"].iloc[-1])
        atr_1h = float(self.df1["atr14"].iloc[-1])
        
        return structural_stop_loss(
            self.df15, entry_price, atr_1h, C.STOP_SWING_LOOKBACK, C.STOP_ATR_MULTIPLIER
        )
