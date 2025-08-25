from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Optional

@dataclass
class Signal:
    symbol: str
    entry_type: str
    entry: float
    stop: float
    atr_pct_1h: float
    range_high_1h: float
    structure: str
    confidence: int
    timeframe: str
    updated_at: str
    # v1.1 Upgrades: Enhanced metadata
    market_regime: str                    # "trending", "reclaim", "weak_rs_only"
    volume_surge: bool                    # whether volume surge was detected
    breakout_confirmation: str            # "multiple_closes", "clean_retest", "none"
    rsi_divergence: bool                  # whether bearish RSI divergence detected
    # Enhanced Feature Engineering for AI Consumption
    technical_features: Dict[str, float | str]  # Rich numerical features for ML
    arm_level: Optional[float] = None    # for watch signals

class Feed(TypedDict):
    updated_at: str
    count: int
    signals: List[Dict]

class WatchFeed(TypedDict):
    updated_at: str
    count: int
    watch: List[Dict]

class Status(TypedDict):
    updated_at: str
    config: Dict
    stats: Dict

# v1.1 Upgrades: Enhanced configuration tracking
class EnhancedConfig(TypedDict):
    top_n_by_vol: int
    atr_band: tuple[float, float]
    min_confidence: int
    near_pct: float
    rs_lookback_4h: int
    rs_edge: float
    # New v1.1 parameters
    donchian_lookback: int
    min_trend_strength: float
    volume_surge_threshold: float
    breakout_confirmation_bars: int
    retest_threshold: float
    stop_atr_multiplier: float
    rsi_divergence_lookback: int
