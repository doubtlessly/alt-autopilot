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
MIN_CONFIDENCE: int = 70           # upgraded: require higher quality signals
NEAR_PCT: float = 0.02             # within 2% of PRH → watch

# Relative strength (4h lookback ~ 3 days)
RS_LOOKBACK_4H: int = 18
RS_EDGE: float = 0.0               # sym_4h_ret − BTC_4h_ret ≥ 0

# v1.1 Upgrades: Market Regime Gates
DONCHIAN_LOOKBACK: int = 20        # 20-day high/low for regime detection
MIN_TREND_STRENGTH: float = 0.001  # minimum EMA slope for trend confirmation

# v1.1 Upgrades: Volume Validation
VOLUME_SURGE_THRESHOLD: float = 1.6  # 1h volume vs 20-bar median
VOLUME_LOOKBACK: int = 3             # last 3 bars for surge calculation
VOLUME_MEDIAN_LOOKBACK: int = 20     # median calculation period

# v1.1 Upgrades: Breakout Confirmation
BREAKOUT_CONFIRMATION_BARS: int = 2  # two 15m closes above PRH
RETEST_THRESHOLD: float = 1.002     # 20 bps through for clean retest
MIN_RETEST_WICK: float = 0.998      # minimum wick below PRH for retest

# v1.1 Upgrades: Structural Stops
STOP_SWING_LOOKBACK: int = 8        # 15m swing low lookback
STOP_ATR_MULTIPLIER: float = 1.2    # ATR-based stop multiplier

# v1.1 Upgrades: Divergence Filter
RSI_DIVERGENCE_LOOKBACK: int = 20   # bars to check for RSI divergence
RSI_PERIOD: int = 14                # RSI calculation period
RSI_DIVERGENCE_MIN_BARS: int = 5    # minimum bars between highs for divergence

# Output
OUT_SIGNALS = "docs/signals.json"
OUT_WATCH   = "docs/watch.json"
OUT_STATUS  = "docs/status.json"
MAX_SIGNALS: int = 10
