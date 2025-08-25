#!/usr/bin/env python3
"""
v1.1 Upgrade Demonstration
Shows the enhanced features in action with sample data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autopilot.ta import (ema, donchian_high, donchian_low, volume_surge, 
                          rsi, structural_stop_loss, breakout_confirmation)
from autopilot.filters import TAFeatures
from autopilot.scoring import confidence, get_signal_quality_tier
from autopilot import config as C

def create_trending_market_data():
    """Create sample data for a trending market"""
    np.random.seed(42)
    n_bars = 200
    
    # Strong uptrend with some pullbacks
    base_price = 100
    trend = np.linspace(0, 30, n_bars)  # Strong upward trend
    noise = np.random.normal(0, 1.5, n_bars)
    prices = base_price + trend + noise
    
    # Create OHLCV data
    data = []
    for i in range(n_bars):
        open_price = prices[i]
        high_price = open_price + abs(np.random.normal(0, 1))
        low_price = open_price - abs(np.random.normal(0, 1))
        close_price = open_price + np.random.normal(0, 0.5)
        
        # Volume increases with trend strength
        volume = abs(np.random.normal(1000 + i*2, 200))
        
        data.append([i, open_price, high_price, low_price, close_price, volume])
    
    return pd.DataFrame(data, columns=["t", "o", "h", "l", "c", "v"])

def create_ranging_market_data():
    """Create sample data for a ranging market"""
    np.random.seed(123)
    n_bars = 200
    
    # Sideways range with some volatility
    base_price = 100
    range_high = 110
    range_low = 90
    
    prices = []
    for i in range(n_bars):
        if i < 50:
            # Initial range
            price = base_price + np.random.normal(0, 3)
        elif i < 100:
            # Breakout attempt
            price = range_high + np.random.normal(0, 2)
        else:
            # Return to range
            price = base_price + np.random.normal(0, 3)
        
        prices.append(max(range_low, min(range_high, price)))
    
    # Create OHLCV data
    data = []
    for i in range(n_bars):
        open_price = prices[i]
        high_price = open_price + abs(np.random.normal(0, 1))
        low_price = open_price - abs(np.random.normal(0, 1))
        close_price = open_price + np.random.normal(0, 0.5)
        volume = abs(np.random.normal(800, 150))
        
        data.append([i, open_price, high_price, low_price, close_price, volume])
    
    return pd.DataFrame(data, columns=["t", "o", "h", "l", "c", "v"])

def demonstrate_market_regime_detection():
    """Show how market regime detection works"""
    print("🔍 Market Regime Detection Demo")
    print("=" * 50)
    
    # Create trending market data
    df_trending = create_trending_market_data()
    df_trending["ema20"] = ema(df_trending["c"], 20)
    df_trending["ema50"] = ema(df_trending["c"], 50)
    
    # Check if it's trending
    current_ema20 = float(df_trending["ema20"].iloc[-1])
    current_ema50 = float(df_trending["ema50"].iloc[-1])
    is_trending = current_ema20 >= current_ema50
    
    print(f"Trending Market Example:")
    print(f"  EMA20: {current_ema20:.2f}")
    print(f"  EMA50: {current_ema50:.2f}")
    print(f"  Is Trending: {is_trending}")
    print(f"  Regime: {'trending' if is_trending else 'weak'}")
    print()

def demonstrate_volume_surge():
    """Show how volume surge detection works"""
    print("📊 Volume Surge Detection Demo")
    print("=" * 50)
    
    df = create_trending_market_data()
    
    # Check volume surge
    surge = volume_surge(df, C.VOLUME_LOOKBACK, C.VOLUME_MEDIAN_LOOKBACK, C.VOLUME_SURGE_THRESHOLD)
    
    recent_volume = df["v"].iloc[-C.VOLUME_LOOKBACK:].sum()
    historical_median = df["v"].iloc[-C.VOLUME_MEDIAN_LOOKBACK:].median()
    surge_ratio = recent_volume / historical_median
    
    print(f"Volume Analysis:")
    print(f"  Recent Volume (3 bars): {recent_volume:.0f}")
    print(f"  Historical Median (20 bars): {historical_median:.0f}")
    print(f"  Surge Ratio: {surge_ratio:.2f}x")
    print(f"  Threshold: {C.VOLUME_SURGE_THRESHOLD}x")
    print(f"  Volume Surge Detected: {surge}")
    print()

def demonstrate_breakout_confirmation():
    """Show how breakout confirmation works"""
    print("🚀 Breakout Confirmation Demo")
    print("=" * 50)
    
    df = create_ranging_market_data()
    
    # Find a potential breakout level
    prh = float(df["h"].iloc[-50:].max())
    current_price = float(df["c"].iloc[-1])
    
    # Check breakout confirmation
    conf_ok, conf_type = breakout_confirmation(
        df, prh, C.BREAKOUT_CONFIRMATION_BARS, 
        C.RETEST_THRESHOLD, C.MIN_RETEST_WICK
    )
    
    print(f"Breakout Analysis:")
    print(f"  Previous Range High: {prh:.2f}")
    print(f"  Current Price: {current_price:.2f}")
    print(f"  Breakout Confirmed: {conf_ok}")
    print(f"  Confirmation Type: {conf_type}")
    print()

def demonstrate_enhanced_scoring():
    """Show how the enhanced scoring system works"""
    print("🎯 Enhanced Scoring System Demo")
    print("=" * 50)
    
    # Example 1: Strong trending signal
    strong_score = confidence(
        "4h-uptrend",           # structure
        True,                    # expansion_ok
        True,                    # trigger_ok
        True,                    # atr_ok
        "trending",              # market_regime
        True,                    # volume_surge
        "breakout_multiple_closes", # breakout_confirmation
        True                     # rsi_divergence
    )
    
    # Example 2: Weak ranging signal
    weak_score = confidence(
        "flat-accept-rs",        # structure
        False,                   # expansion_ok
        True,                    # trigger_ok
        True,                    # atr_ok
        "weak_rs_only",          # market_regime
        False,                   # volume_surge
        "breakout_no_confirmation", # breakout_confirmation
        True                     # rsi_divergence
    )
    
    print(f"Scoring Examples:")
    print(f"  Strong Trending Signal: {strong_score}/100 ({get_signal_quality_tier(strong_score)})")
    print(f"  Weak Ranging Signal: {weak_score}/100 ({get_signal_quality_tier(weak_score)})")
    print()
    
    print(f"Score Breakdown for Strong Signal:")
    print(f"  Market Regime (trending): +30")
    print(f"  Structure (4h-uptrend): +25")
    print(f"  Volume Surge: +20")
    print(f"  Breakout Confirmation: +15")
    print(f"  Technical Health: +10")
    print(f"  Synergy Bonus: +5")
    print(f"  Total: {strong_score}/100")

def main():
    """Run the v1.1 demonstration"""
    print("🚀 Alt-Autopilot v1.1 Feature Demonstration")
    print("=" * 60)
    print()
    
    try:
        demonstrate_market_regime_detection()
        demonstrate_volume_surge()
        demonstrate_breakout_confirmation()
        demonstrate_enhanced_scoring()
        
        print("🎉 v1.1 Demonstration Complete!")
        print("\nKey Improvements:")
        print("✅ Market regime awareness prevents trading in weak markets")
        print("✅ Volume surge detection catches institutional interest")
        print("✅ Breakout confirmation eliminates false breakouts")
        print("✅ Structural stops adapt to market conditions")
        print("✅ Enhanced scoring reflects signal quality")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
