from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict, List, Dict

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
    arm_level: float | None = None  # for watch

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
