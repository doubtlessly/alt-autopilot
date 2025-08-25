from __future__ import annotations

# Exchange + universe
EXCHANGE_ID: str = "kucoin"
QUOTE: str = "USDT"
TOP_N_BY_VOL: int = 200            # scan more symbols to avoid empty days

# Candles
BARS_4H: int = 150
BARS_1H: int = 150
BARS_15M: int = 150
BARS_1D: int = 180                 # extra: daily context (≈ 6 months)

# TA thresholds (bear-resilient preset)
ATR_BAND = (1.0, 40.0)            # 1h ATR%(14) must fall in band
MIN_CONFIDENCE: int = 60
NEAR_PCT: float = 0.02             # within 2% of PRH → watch

# Relative strength (4h lookback ~ 3 days)
RS_LOOKBACK_4H: int = 18
RS_EDGE: float = 0.0               # sym_4h_ret − BTC_4h_ret ≥ 0

# Output
OUT_SIGNALS = "docs/signals.json"
OUT_WATCH   = "docs/watch.json"
OUT_STATUS  = "docs/status.json"
MAX_SIGNALS: int = 10
