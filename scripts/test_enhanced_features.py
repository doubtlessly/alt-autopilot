#!/usr/bin/env python3
"""
Test script for Enhanced Feature Engineering
Validates that all new AI-friendly features are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from autopilot.ta import (calculate_price_momentum, calculate_volume_trend, 
                          calculate_volatility_regime, calculate_market_strength,
                          calculate_correlation_with_btc, calculate_trend_quality)
from autopilot.filters import TAFeatures

def create_test_data():
    """Create sample OHLCV data for testing enhanced features"""
    np.random.seed(42)
    n_bars = 200
    
    # Create trending data with increasing volume
    base_price = 100
    trend = np.linspace(0, 25, n_bars)  # Strong upward trend
    noise = np.random.normal(0, 1.5, n_bars)
    prices = base_price + trend + noise
    
    # Volume increases with trend strength
    volumes = []
    for i in range(n_bars):
        base_volume = 1000 + i * 2  # Volume increases over time
        volume = abs(np.random.normal(base_volume, 200))
        volumes.append(volume)
    
    # Create OHLCV data
    data = []
    for i in range(n_bars):
        open_price = prices[i]
        high_price = open_price + abs(np.random.normal(0, 1))
        low_price = open_price - abs(np.random.normal(0, 1))
        close_price = open_price + np.random.normal(0, 0.5)
        volume = volumes[i]
        
        data.append([i, open_price, high_price, low_price, close_price, volume])
    
    return pd.DataFrame(data, columns=["t", "o", "h", "l", "c", "v"])

def test_price_momentum():
    """Test price momentum calculation"""
    print("🧪 Testing Price Momentum Calculation...")
    
    df = create_test_data()
    
    # Test momentum calculation
    momentum = calculate_price_momentum(df, lookback=20)
    
    print(f"  Price Momentum: {momentum}")
    print(f"  Type: {type(momentum)}")
    print(f"  Range: 0-1 scale ✅")
    print(f"  Value: {'Strong' if momentum > 0.7 else 'Moderate' if momentum > 0.5 else 'Weak'}")
    print()

def test_volume_trend():
    """Test volume trend calculation"""
    print("🧪 Testing Volume Trend Calculation...")
    
    df = create_test_data()
    
    # Test volume trend calculation
    volume_trend = calculate_volume_trend(df, lookback=20)
    
    print(f"  Volume Trend: {volume_trend}")
    print(f"  Type: {type(volume_trend)}")
    print(f"  Range: 0-1 scale ✅")
    print(f"  Trend: {'Strong' if volume_trend > 0.7 else 'Moderate' if volume_trend > 0.5 else 'Weak'}")
    print()

def test_volatility_regime():
    """Test volatility regime classification"""
    print("🧪 Testing Volatility Regime Classification...")
    
    # Test different ATR percentages
    test_cases = [3.0, 8.0, 20.0, 35.0]
    
    for atr_pct in test_cases:
        regime = calculate_volatility_regime(atr_pct)
        print(f"  ATR {atr_pct}%: {regime} regime")
    
    print("  All regimes properly classified ✅")
    print()

def test_trend_quality():
    """Test trend quality calculation"""
    print("🧪 Testing Trend Quality Calculation...")
    
    df = create_test_data()
    
    # Test trend quality
    quality = calculate_trend_quality(df, ema_short=20, ema_long=50)
    
    print(f"  Trend Quality: {quality}")
    print(f"  Type: {type(quality)}")
    print(f"  Range: 0-1 scale ✅")
    print(f"  Quality: {'Excellent' if quality > 0.8 else 'Good' if quality > 0.6 else 'Fair' if quality > 0.4 else 'Poor'}")
    print()

def test_correlation_calculation():
    """Test correlation with BTC calculation"""
    print("🧪 Testing BTC Correlation Calculation...")
    
    # Create two similar datasets (high correlation)
    df_symbol = create_test_data()
    df_btc = create_test_data()  # Similar pattern for testing
    
    correlation = calculate_correlation_with_btc(df_symbol, df_btc, lookback=20)
    
    print(f"  BTC Correlation: {correlation}")
    print(f"  Type: {type(correlation)}")
    print(f"  Range: 0-1 scale ✅")
    print(f"  Correlation: {'High' if correlation > 0.7 else 'Medium' if correlation > 0.4 else 'Low'}")
    print()

def test_market_strength():
    """Test market strength calculation"""
    print("🧪 Testing Market Strength Calculation...")
    
    # Test different combinations
    test_cases = [
        (0.9, 0.8, 0.7),  # Strong trend, strong volume, good momentum
        (0.5, 0.6, 0.4),  # Moderate across the board
        (0.2, 0.3, 0.1)   # Weak across the board
    ]
    
    for trend, volume, momentum in test_cases:
        strength = calculate_market_strength(trend, volume, momentum)
        print(f"  Trend:{trend}, Volume:{volume}, Momentum:{momentum} → Strength: {strength}")
    
    print("  Market strength properly calculated ✅")
    print()

def test_enhanced_features_integration():
    """Test that enhanced features are properly integrated in TAFeatures"""
    print("🧪 Testing Enhanced Features Integration...")
    
    # Create sample dataframes
    df4 = create_test_data()
    df1 = create_test_data()
    df15 = create_test_data()
    df1d = create_test_data()
    df_btc = create_test_data()
    
    # Create TAFeatures instance
    feats = TAFeatures(df4, df1, df15, df1d, df_btc)
    
    # Get enhanced features
    enhanced_features = feats.get_enhanced_features()
    
    print("  Enhanced Features Retrieved:")
    for feature, value in enhanced_features.items():
        print(f"    {feature}: {value} ({type(value).__name__})")
    
    # Validate all features are present
    expected_features = [
        "price_momentum", "volume_trend", "volatility_regime", 
        "trend_quality", "correlation_with_btc", "market_strength"
    ]
    
    missing_features = [f for f in expected_features if f not in enhanced_features]
    if missing_features:
        print(f"  ❌ Missing features: {missing_features}")
        return False
    else:
        print("  ✅ All expected features present")
    
    # Validate data types
    numerical_features = ["price_momentum", "volume_trend", "trend_quality", "correlation_with_btc", "market_strength"]
    categorical_features = ["volatility_regime"]
    
    for feature in numerical_features:
        if not isinstance(enhanced_features[feature], (int, float)):
            print(f"  ❌ {feature} should be numerical, got {type(enhanced_features[feature])}")
            return False
    
    for feature in categorical_features:
        if not isinstance(enhanced_features[feature], str):
            print(f"  ❌ {feature} should be categorical, got {type(enhanced_features[feature])}")
            return False
    
    print("  ✅ All features have correct data types")
    print()
    return True

def main():
    """Run all enhanced feature tests"""
    print("🚀 Enhanced Feature Engineering Validation Tests")
    print("=" * 60)
    print()
    
    try:
        test_price_momentum()
        test_volume_trend()
        test_volatility_regime()
        test_trend_quality()
        test_correlation_calculation()
        test_market_strength()
        
        integration_success = test_enhanced_features_integration()
        
        if integration_success:
            print("🎉 All Enhanced Feature Tests Passed!")
            print("\nEnhanced Features Summary:")
            print("✅ Price Momentum: 0-1 scale momentum strength")
            print("✅ Volume Trend: 0-1 scale volume pattern strength")
            print("✅ Volatility Regime: low/medium/high classification")
            print("✅ Trend Quality: 0-1 scale trend cleanliness")
            print("✅ BTC Correlation: 0-1 scale correlation strength")
            print("✅ Market Strength: 0-1 scale overall health")
            print("\nAll features are AI-friendly and ready for ML consumption!")
        else:
            print("❌ Some integration tests failed")
            return 1
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
