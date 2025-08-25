# Alt-Autopilot v1.1

**Professional-grade cryptocurrency trading signal generator** with institutional-quality risk management and market regime awareness.

## 🚀 v1.1 Major Upgrades

### **Market Regime Gates**
- **Daily trend detection**: Only generate signals in trending markets (EMA20 ≥ EMA50)
- **Donchian channel reclaims**: Catch trend reversals early with 20-day high reclaims
- **Weak regime handling**: In bear markets, only allow relative strength leaders to watch list

### **Enhanced Breakout Validation**
- **Volume surge detection**: 1.6× volume vs 20-bar median for institutional interest
- **Multiple confirmation methods**: Two 15m closes above PRH or clean retest with 20bps through
- **False breakout protection**: Eliminate hairline breakouts with proper thresholds

### **Structural Risk Management**
- **Smart stop losses**: Swing low + ATR-based stops (tighter in chop, looser in trend)
- **Risk-adjusted scoring**: Higher confidence requirements (70+ vs 60+)
- **Market condition awareness**: Adaptive to volatility regimes

### **Advanced Technical Analysis**
- **RSI divergence filtering**: Detect bearish divergences while avoiding false positives
- **Enhanced momentum**: Volume-weighted breakout validation
- **Multi-timeframe confluence**: 4H structure + 1H momentum + 15M triggers + daily context

## 🎯 Features

- **Scans top 200 altcoins** by volume on KuCoin
- **Multi-timeframe analysis**: 4H, 1H, 15M, and daily data
- **Relative strength ranking** vs BTC baseline
- **Professional risk management** with structural stops
- **Market regime awareness** for adaptive filtering
- **Volume surge validation** for institutional interest
- **Enhanced signal quality** with 0-100 confidence scoring
- **AI-Optimized Features**: Rich numerical data for machine learning

## 📊 Output Files

- `docs/signals.json` - High-confidence signals (≥70 score) with enhanced features
- `docs/watch.json` - Near-trigger opportunities with enhanced features
- `docs/status.json` - Diagnostic statistics and filtering results

## 🧠 Enhanced Feature Engineering for AI

The system now generates **AI-friendly numerical features** that enable machine learning models to make better trading decisions:

### **Technical Features (0-1 Scale)**
```json
"technical_features": {
  "price_momentum": 0.85,        // How strong is recent momentum vs history
  "volume_trend": 0.72,          // How strong is volume pattern vs history  
  "trend_quality": 0.68,         // How clean/strong is the trend
  "correlation_with_btc": 0.23,  // How correlated with BTC (0=independent, 1=perfect)
  "market_strength": 0.75        // Overall market health (weighted combination)
}
```

### **Categorical Features**
```json
"volatility_regime": "medium"    // "low", "medium", "high" based on ATR
```

### **Feature Benefits for AI**
- **Normalized Scales**: All numerical features are 0-1, perfect for ML
- **Historical Context**: Features compare current vs historical performance
- **Multi-Dimensional**: Rich context for better decision making
- **Consistent Format**: Structured data that's easy to parse and train on

## 🏗️ Architecture

```
autopilot/
├── config.py          # Centralized configuration
├── ta.py             # Technical analysis + Enhanced feature engineering
├── filters.py        # Signal filtering logic + Feature calculation
├── scoring.py        # Confidence scoring system
├── pipeline.py       # Main signal generation + Enhanced output
├── exchanges.py      # Exchange interface
├── fetch.py          # Data fetching utilities
├── writer.py         # Output generation
└── log.py           # Logging utilities
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python -m scripts.run_pipeline

# Test v1.1 upgrades
python -m scripts.test_v1_1

# Test enhanced features
python -m scripts.test_enhanced_features
```

## ⚙️ Configuration

Key v1.1 parameters in `autopilot/config.py`:

```python
# Market Regime
DONCHIAN_LOOKBACK = 20        # 20-day high/low for regime detection
MIN_TREND_STRENGTH = 0.001    # Minimum EMA slope for trend

# Volume Validation
VOLUME_SURGE_THRESHOLD = 1.6  # Volume surge vs historical median
VOLUME_LOOKBACK = 3           # Bars for surge calculation

# Breakout Confirmation
BREAKOUT_CONFIRMATION_BARS = 2  # Multiple closes above PRH
RETEST_THRESHOLD = 1.002      # 20 bps through for clean retest

# Risk Management
STOP_ATR_MULTIPLIER = 1.2     # ATR-based stop multiplier
MIN_CONFIDENCE = 70           # Higher quality threshold
```

## 🔄 GitHub Actions

Automated signal generation every 15 minutes via GitHub Actions workflow.

## 📈 Signal Quality Tiers

- **90-100**: Excellent - Strong trend + volume + confirmation
- **80-89**: Very Good - Good structure + volume surge
- **70-79**: Good - Meets minimum quality threshold
- **60-69**: Fair - Below threshold, watch only
- **0-59**: Poor - Rejected by filters

## 🎯 Trading Strategy

1. **Market Regime Check**: Only trade in trending or reclaiming markets
2. **Structure Validation**: 4H uptrend or range high reclaim
3. **Volume Confirmation**: Institutional interest via volume surge
4. **Breakout Validation**: Multiple closes or clean retest
5. **Risk Management**: Structural stops based on swing lows + ATR
6. **Quality Scoring**: Multi-factor confidence assessment

## 🔧 Development

The system is built with:
- **Type hints** and modern Python practices
- **Modular architecture** for easy maintenance
- **Comprehensive testing** via `scripts/test_v1_1.py`
- **Clean separation** of concerns
- **Professional logging** and error handling

## 📝 License

MIT License - see LICENSE file for details.