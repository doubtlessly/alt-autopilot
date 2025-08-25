from __future__ import annotations

def confidence(structure: str, expansion_ok: bool, trigger_ok: bool, atr_ok: bool,
               market_regime: str, volume_surge: bool, breakout_confirmation: str,
               rsi_divergence: bool) -> int:
    """
    v1.1 Enhanced confidence scoring system
    Returns score 0-100 based on signal quality
    """
    score = 0
    
    # Market Regime Gate (highest weight - foundation)
    if market_regime == "trending":
        score += 30
    elif market_regime == "reclaim":
        score += 25
    elif market_regime == "weak_rs_only":
        score += 15  # Reduced score for weak regimes
    else:
        return 0  # No score if regime doesn't allow signals
    
    # Structure Quality
    if structure == "4h-uptrend":
        score += 25
    elif structure in ("range-high-reclaim", "flat-accept-rs"):
        score += 20
    else:
        score += 10
    
    # Volume Surge (institutional interest)
    if volume_surge:
        score += 20
    else:
        score += 5  # Reduced score without volume confirmation
    
    # Breakout Confirmation Quality
    if "multiple_closes" in breakout_confirmation:
        score += 15
    elif "clean_retest" in breakout_confirmation:
        score += 15
    elif "no_confirmation" in breakout_confirmation:
        score += 5  # Reduced score without confirmation
    else:
        score += 10
    
    # Technical Health
    if atr_ok:
        score += 10
    else:
        score += 0
    
    # RSI Divergence Filter
    if rsi_divergence:
        score += 0  # No bonus, but no penalty
    else:
        score += 0  # No penalty for divergence
    
    # Synergy Bonus: Strong combinations get extra points
    if (market_regime in ("trending", "reclaim") and 
        volume_surge and 
        "multiple_closes" in breakout_confirmation):
        score += 5  # Synergy bonus for strong setups
    
    # Cap at 100
    return max(0, min(100, int(score)))

def get_signal_quality_tier(score: int) -> str:
    """Convert numerical score to quality tier"""
    if score >= 90:
        return "excellent"
    elif score >= 80:
        return "very_good"
    elif score >= 70:
        return "good"
    elif score >= 60:
        return "fair"
    else:
        return "poor"
