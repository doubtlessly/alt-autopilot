#!/usr/bin/env python3
"""
Test script for v1.1 upgrades
Validates that all new features are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autopilot.ta import (ema, atr, obv_proxy, donchian_high, donchian_low, 
                          volume_surge, rsi, detect_bearish_rsi_divergence,
                          structural_stop_loss, breakout_confirmation)
from autopilot.filters import TAFeatures
from autopilot.scoring import confidence, get_signal_quality_tier
from autopilot import config as C

def create_sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n_bars = 200
    
    # Generate sample price data with trend
    base_price = 100
    trend = np.linspace(0, 20, n_bars)  # Upward trend
    noise = np.random.normal(0, 2, n_bars)
    prices = base_price + trend + noise
    
    # Create OHLCV data
    data = []
    for i in range(n_bars):
        open_price = prices[i]
        high_price = open_price + abs(np.random.normal(0, 1))
        low_price = open_price - abs(np.random.normal(0, 1))
        close_price = open_price + np.random.normal(0, 0.5)
        volume = abs(np.random.normal(1000, 200))
        
        data.append([i, open_price, high_price, low_price, close_price, volume])
    
    return pd.DataFrame(data, columns=["t", "o", "h", "l", "c", "v"])

def test_technical_indicators():
    """Test all new technical indicators"""
    print("🧪 Testing Technical Indicators...")
    
    df = create_sample_data()
    
    # Test EMA
    ema20 = ema(df["c"], 20)
    assert len(ema20) == len(df), "EMA length mismatch"
    print("✅ EMA calculation working")
    
    # Test ATR
    atr14 = atr(df, 14)
    assert len(atr14) == len(df), "ATR length mismatch"
    print("✅ ATR calculation working")
    
    # Test Donchian channels
    donch_high = donchian_high(df, 20)
    donch_low = donchian_low(df, 20)
    assert len(donch_high) == len(df), "Donchian high length mismatch"
    assert len(donch_low) == len(df), "Donchian low length mismatch"
    print("✅ Donchian channels working")
    
    # Test RSI
    rsi_series = rsi(df, 14)
    assert len(rsi_series) == len(df), "RSI length mismatch"
    assert all(0 <= r <= 100 for r in rsi_series.dropna()), "RSI out of range"
    print("✅ RSI calculation working")
    
    # Test volume surge
    surge = volume_surge(df, 3, 20, 1.6)
    assert isinstance(surge, bool), "Volume surge should return boolean"
    print("✅ Volume surge detection working")
    
    # Test structural stop loss
    entry_price = float(df["c"].iloc[-1])
    atr_val = float(atr14.iloc[-1])
    stop = structural_stop_loss(df, entry_price, atr_val, 8, 1.2)
    assert isinstance(stop, float), "Stop loss should return float"
    print("✅ Structural stop loss working")
    
    # Test breakout confirmation
    prh = float(df["h"].iloc[-50:].max())
    conf_ok, conf_type = breakout_confirmation(df, prh, 2, 1.002, 0.998)
    assert isinstance(conf_ok, bool), "Breakout confirmation should return boolean"
    assert isinstance(conf_type, str), "Confirmation type should be string"
    print("✅ Breakout confirmation working")

def test_filters():
    """Test the enhanced TAFeatures class"""
    print("\n🧪 Testing Enhanced Filters...")
    
    # Create sample dataframes
    df4 = create_sample_data()
    df1 = create_sample_data()
    df15 = create_sample_data()
    df1d = create_sample_data()
    df_btc = create_sample_data()
    
    feats = TAFeatures(df4, df1, df15, df1d, df_btc)
    
    # Test market regime detection
    regime_ok, regime_type = feats.market_regime_ok()
    assert isinstance(regime_ok, bool), "Market regime should return boolean"
    assert isinstance(regime_type, str), "Regime type should be string"
    print("✅ Market regime detection working")
    
    # Test volume surge
    volume_ok = feats.volume_surge_ok()
    assert isinstance(volume_ok, bool), "Volume surge should return boolean"
    print("✅ Volume surge filter working")
    
    # Test RSI divergence
    rsi_ok = feats.rsi_divergence_ok()
    assert isinstance(rsi_ok, bool), "RSI divergence should return boolean"
    print("✅ RSI divergence filter working")
    
    # Test structural stop loss
    stop = feats.invalidation()
    assert isinstance(stop, float), "Stop loss should return float"
    print("✅ Structural stop loss filter working")

def test_scoring():
    """Test the enhanced scoring system"""
    print("\n🧪 Testing Enhanced Scoring...")
    
    # Test confidence scoring
    score = confidence("4h-uptrend", True, True, True, "trending", True, "breakout_multiple_closes", True)
    assert isinstance(score, int), "Confidence should return integer"
    assert 0 <= score <= 100, "Confidence should be 0-100"
    print("✅ Confidence scoring working")
    
    # Test quality tier
    tier = get_signal_quality_tier(85)
    assert isinstance(tier, str), "Quality tier should return string"
    assert tier in ["excellent", "very_good", "good", "fair", "poor"], "Invalid quality tier"
    print("✅ Quality tier classification working")

def test_configuration():
    """Test that all v1.1 configuration parameters are accessible"""
    print("\n🧪 Testing Configuration...")
    
    # Test new config parameters
    assert hasattr(C, 'DONCHIAN_LOOKBACK'), "DONCHIAN_LOOKBACK not found in config"
    assert hasattr(C, 'VOLUME_SURGE_THRESHOLD'), "VOLUME_SURGE_THRESHOLD not found in config"
    assert hasattr(C, 'BREAKOUT_CONFIRMATION_BARS'), "BREAKOUT_CONFIRMATION_BARS not found in config"
    assert hasattr(C, 'STOP_ATR_MULTIPLIER'), "STOP_ATR_MULTIPLIER not found in config"
    assert hasattr(C, 'RSI_DIVERGENCE_LOOKBACK'), "RSI_DIVERGENCE_LOOKBACK not found in config"
    
    print("✅ All v1.1 configuration parameters accessible")

def main():
    """Run all tests"""
    print("🚀 Starting v1.1 Upgrade Validation Tests...\n")
    
    try:
        test_technical_indicators()
        test_filters()
        test_scoring()
        test_configuration()
        
        print("\n🎉 All v1.1 upgrade tests passed successfully!")
        print("✅ Technical indicators working")
        print("✅ Enhanced filters working")
        print("✅ Improved scoring system working")
        print("✅ Configuration properly updated")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
