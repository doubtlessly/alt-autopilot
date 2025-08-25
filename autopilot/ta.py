from __future__ import annotations
import numpy as np
import pandas as pd

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["c"].shift(1)
    tr = pd.concat([(df["h"]-df["l"]).abs(),
                    (df["h"]-pc).abs(),
                    (df["l"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def obv_proxy(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["c"].diff().fillna(0)) * df["v"]).cumsum()

def prior_range_high_1h(df_1h: pd.DataFrame, min_look=36, max_look=60) -> float:
    look = min(max_look, max(min_look, len(df_1h)-2))
    return float(df_1h["h"].iloc[-(look+1):-1].max())

def slope_up(series: pd.Series, look: int = 5, tol: float = 0.999) -> bool:
    if len(series) < look + 1: return False
    return series.iloc[-1] >= series.iloc[-look] * tol

def pct_return(series: pd.Series, look: int) -> float:
    if len(series) <= look: return 0.0
    return float(series.iloc[-1] / series.iloc[-look] - 1.0)

# v1.1 Upgrades: Enhanced Technical Analysis

def donchian_high(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Calculate Donchian channel high (rolling maximum of highs)"""
    return df["h"].rolling(lookback).max()

def donchian_low(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Calculate Donchian channel low (rolling minimum of lows)"""
    return df["l"].rolling(lookback).min()

def volume_surge(df: pd.DataFrame, lookback: int, median_lookback: int, threshold: float) -> bool:
    """
    Detect volume surge: recent volume vs historical median
    Returns True if sum of last N bars > median * threshold
    """
    if len(df) < max(lookback, median_lookback):
        return False
    
    recent_volume = df["v"].iloc[-lookback:].sum()
    historical_median = df["v"].iloc[-median_lookback:].median()
    
    if historical_median == 0:
        return False
    
    return bool(recent_volume >= historical_median * threshold)

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = df["c"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_bearish_rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback: int, min_bars: int) -> bool:
    """
    Detect bearish RSI divergence: higher price highs with lower RSI highs
    Only consider divergence if it's not in a strong uptrend
    """
    if len(df) < lookback:
        return False
    
    # Get recent price and RSI data
    recent_prices = df["c"].iloc[-lookback:]
    recent_rsi = rsi_series.iloc[-lookback:]
    
    # Find local highs in price and RSI
    price_highs = []
    rsi_highs = []
    
    for i in range(1, len(recent_prices) - 1):
        if (recent_prices.iloc[i] > recent_prices.iloc[i-1] and 
            recent_prices.iloc[i] > recent_prices.iloc[i+1]):
            price_highs.append((i, recent_prices.iloc[i]))
        
        if (recent_rsi.iloc[i] > recent_rsi.iloc[i-1] and 
            recent_rsi.iloc[i] > recent_rsi.iloc[i+1]):
            rsi_highs.append((i, recent_rsi.iloc[i]))
    
    # Need at least 2 highs to detect divergence
    if len(price_highs) < 2 or len(rsi_highs) < 2:
        return False
    
    # Check for bearish divergence pattern
    for i in range(len(price_highs) - 1):
        for j in range(len(rsi_highs) - 1):
            # Ensure minimum bars between highs
            if (abs(price_highs[i+1][0] - price_highs[i][0]) >= min_bars and
                abs(rsi_highs[j+1][0] - rsi_highs[j][0]) >= min_bars):
                
                # Check if price is making higher highs but RSI is making lower highs
                if (price_highs[i+1][1] > price_highs[i][1] and 
                    rsi_highs[j+1][1] < rsi_highs[j][1]):
                    
                    # Additional filter: only reject if not in strong uptrend
                    # Check if price is above EMA20 and EMA20 is sloping up
                    if len(df) > 20:
                        ema20 = ema(df["c"], 20)
                        if (df["c"].iloc[-1] > ema20.iloc[-1] and 
                            ema20.iloc[-1] > ema20.iloc[-5]):
                            continue  # Strong uptrend, don't reject
                    
                    return True
    
    return False

def structural_stop_loss(df_15m: pd.DataFrame, entry_price: float, atr_1h: float, 
                        swing_lookback: int, atr_multiplier: float) -> float:
    """
    Calculate structural stop loss based on swing low and ATR
    Returns the higher of: swing low or entry - ATR * multiplier
    """
    if len(df_15m) < swing_lookback:
        return entry_price * 0.95  # fallback
    
    swing_low = float(df_15m["l"].iloc[-swing_lookback:].min())
    atr_stop = entry_price - (atr_1h * atr_multiplier)
    
    return max(swing_low, atr_stop)

def breakout_confirmation(df_15m: pd.DataFrame, prh: float, confirmation_bars: int, 
                         retest_threshold: float, min_wick: float) -> tuple[bool, str]:
    """
    Validate breakout with multiple confirmation methods
    Returns (is_valid, confirmation_type)
    """
    if len(df_15m) < confirmation_bars:
        return False, "insufficient_data"
    
    # Method 1: Multiple closes above PRH
    recent_closes = df_15m["c"].iloc[-confirmation_bars:]
    if all(close > prh for close in recent_closes):
        return True, "multiple_closes"
    
    # Method 2: Clean retest with wick below and close above
    last_close = float(df_15m["c"].iloc[-1])
    last_low = float(df_15m["l"].iloc[-1])
    last_high = float(df_15m["h"].iloc[-1])
    
    # Check for retest pattern
    if (last_low <= prh * min_wick and  # wick below PRH
        last_close >= prh * retest_threshold and  # close above with threshold
        last_high > prh):  # high above PRH
        return True, "clean_retest"
    
    return False, "no_confirmation"

# Enhanced Feature Engineering for AI Consumption
def calculate_price_momentum(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate price momentum as a percentile (0-1 scale)
    Higher values = stronger momentum vs historical performance
    """
    if len(df) < lookback * 2:
        return 0.5  # neutral if insufficient data
    
    # Calculate recent return
    recent_return = (df["c"].iloc[-1] / df["c"].iloc[-lookback]) - 1
    
    # Calculate historical returns for comparison
    historical_returns = []
    for i in range(lookback, len(df) - lookback):
        hist_return = (df["c"].iloc[i] / df["c"].iloc[i-lookback]) - 1
        historical_returns.append(hist_return)
    
    if not historical_returns:
        return 0.5
    
    # Calculate percentile (how strong is recent momentum vs history)
    percentile = sum(1 for r in historical_returns if r < recent_return) / len(historical_returns)
    return round(percentile, 3)

def calculate_volume_trend(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate volume trend strength (0-1 scale)
    Higher values = stronger volume trend vs historical patterns
    """
    if len(df) < lookback * 2:
        return 0.5  # neutral if insufficient data
    
    # Recent volume average
    recent_volume = df["v"].iloc[-lookback:].mean()
    
    # Historical volume average
    historical_volume = df["v"].iloc[-lookback*2:-lookback].mean()
    
    if historical_volume == 0:
        return 0.5
    
    # Calculate volume ratio and normalize to 0-1
    ratio = recent_volume / historical_volume
    
    # Normalize: 0.5 = no change, 1.0 = 2x volume, 0.0 = 0.5x volume
    normalized = min(1.0, max(0.0, (ratio - 0.5) * 2))
    return round(normalized, 3)

def calculate_volatility_regime(atr_percent: float) -> str:
    """
    Classify volatility regime based on ATR percentage
    Returns: "low", "medium", "high"
    """
    if atr_percent < 5.0:
        return "low"
    elif atr_percent < 15.0:
        return "medium"
    else:
        return "high"

def calculate_market_strength(trend_score: float, volume_score: float, momentum_score: float) -> float:
    """
    Calculate overall market strength (0-1 scale)
    Combines multiple factors into single strength metric
    """
    # Weighted average of different factors
    weights = [0.4, 0.3, 0.3]  # trend, volume, momentum
    strength = (trend_score * weights[0] + 
               volume_score * weights[1] + 
               momentum_score * weights[2])
    
    return round(max(0.0, min(1.0, strength)), 3)

def calculate_correlation_with_btc(df_symbol: pd.DataFrame, df_btc: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate correlation with BTC (0-1 scale)
    Higher values = more correlated with BTC
    """
    if len(df_symbol) < lookback or len(df_btc) < lookback:
        return 0.5  # neutral if insufficient data
    
    # Calculate returns for both
    symbol_returns = df_symbol["c"].pct_change().iloc[-lookback:].dropna()
    btc_returns = df_btc["c"].pct_change().iloc[-lookback:].dropna()
    
    # Align lengths
    min_len = min(len(symbol_returns), len(btc_returns))
    if min_len < 10:  # need minimum data for correlation
        return 0.5
    
    symbol_returns = symbol_returns.iloc[-min_len:]
    btc_returns = btc_returns.iloc[-min_len:]
    
    # Calculate correlation
    try:
        correlation = symbol_returns.corr(btc_returns)
        if pd.isna(correlation):
            return 0.5
        
        # Convert from -1 to +1 scale to 0 to 1 scale
        # 0 = no correlation, 1 = perfect correlation
        normalized_corr = (correlation + 1) / 2
        return round(normalized_corr, 3)
    except:
        return 0.5

def calculate_trend_quality(df: pd.DataFrame, ema_short: int = 20, ema_long: int = 50) -> float:
    """
    Calculate trend quality score (0-1 scale)
    Higher values = cleaner, stronger trend
    """
    if len(df) < ema_long:
        return 0.5
    
    # Calculate EMAs
    ema_short_series = ema(df["c"], ema_short)
    ema_long_series = ema(df["c"], ema_long)
    
    # Check if short EMA > long EMA (uptrend)
    current_trend = ema_short_series.iloc[-1] > ema_long_series.iloc[-1]
    
    if not current_trend:
        return 0.0  # downtrend = 0 quality
    
    # Calculate trend strength (how far apart are EMAs)
    ema_distance = (ema_short_series.iloc[-1] - ema_long_series.iloc[-1]) / ema_long_series.iloc[-1]
    
    # Normalize to 0-1 scale (0.1 = 10% separation = strong trend)
    trend_strength = min(1.0, ema_distance * 10)
    
    # Check trend consistency (how straight is the trend)
    recent_ema_short = ema_short_series.iloc[-10:]
    trend_consistency = 1.0 - (recent_ema_short.std() / recent_ema_short.mean())
    trend_consistency = max(0.0, min(1.0, trend_consistency))
    
    # Combine trend strength and consistency
    quality = (trend_strength * 0.7 + trend_consistency * 0.3)
    return round(quality, 3)
