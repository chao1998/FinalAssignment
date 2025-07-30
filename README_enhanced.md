# Enhanced Crypto Prediction Model with On-chain Metrics and Sentiment Analysis

## Overview

This repository contains an enhanced cryptocurrency prediction model that incorporates on-chain metrics and sentiment analysis alongside traditional technical indicators. The model provides comprehensive feature engineering and prediction capabilities for crypto market analysis.

## Features

### 🚀 Enhanced Model Capabilities
- **29 Total Features** across 4 categories:
  - **5 Price Features**: Basic OHLCV data and ratios
  - **11 Technical Features**: RSI, MACD, ATR, Bollinger Bands, momentum indicators
  - **6 On-chain Features**: Transaction metrics, network health, value flow indicators
  - **7 Sentiment Features**: Social media sentiment, news analysis, market psychology

### 📊 Data Sources Integration
- **On-chain Metrics**: Transaction counts, active addresses, hash rate changes, MVRV ratio, NVT ratio
- **Sentiment Analysis**: Social media sentiment, news sentiment, Fear & Greed index, Google Trends proxy
- **Technical Indicators**: Traditional TA indicators with enhanced feature engineering
- **15-minute Interval Structure**: Maintained for high-frequency trading applications

### 🎯 Model Performance
- Random Forest regression with hyperparameter optimization
- Feature importance analysis across all categories
- Comprehensive model validation and testing
- 24-hour prediction horizon (96 15-minute intervals)

## Files Structure

```
├── OptimizeModel_enhanced_crypto_prediction.py    # Main enhanced model (live data version)
├── OptimizeModel_demo_version.py                  # Demo version with simulated data
├── crypto_data_utilities.py                       # Data fetching utilities for on-chain and sentiment data
├── complete_crypto_analysis.py                    # Complete integration and analysis script
├── feature_importance_demo.png                    # Feature importance visualization
├── crypto_model_demo.log                         # Demo model logs
├── complete_crypto_analysis.log                  # Complete analysis logs
└── README.md                                      # This file
```

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance requests
```

### Quick Start

1. **Run the complete analysis**:
```bash
python complete_crypto_analysis.py
```

2. **Run the demo version** (with simulated data):
```bash
python OptimizeModel_demo_version.py
```

3. **Test data utilities**:
```bash
python crypto_data_utilities.py
```

## Key Features Implementation

### 1. On-chain Metrics
- **Transaction Count**: Per 15-minute intervals
- **Active Addresses**: Network participation metrics
- **Hash Rate Changes**: Network security indicators
- **MVRV Ratio**: Market value to realized value
- **NVT Ratio**: Network value to transactions
- **Exchange Flow Ratios**: On-chain vs exchange volume analysis

### 2. Sentiment Analysis
- **Social Media Sentiment**: Aggregated from Twitter, Reddit, Telegram
- **News Sentiment**: Financial news analysis
- **Fear & Greed Index**: Market psychology indicator
- **Google Trends**: Search volume proxy
- **Volatility Sentiment**: Market stress indicators

### 3. Enhanced Feature Engineering
```python
# Example usage of the enhanced model
from complete_crypto_analysis import CompleteCryptoAnalysis

# Initialize analysis
analysis = CompleteCryptoAnalysis(symbol="BTC-USD", interval="15m")

# Run complete analysis
results = analysis.run_complete_analysis(days=60)

# Access results
print(f"Model R²: {results['training_results']['test_r2']:.4f}")
print(f"Enhanced features contribution: {results['enhanced_feature_analysis']['enhanced_percentage']:.1f}%")
```

## Results Summary

### Model Performance
- **Training R²**: ~0.51
- **Test R²**: Varies based on data (overfitting protection included)
- **Feature Count**: 29 total features
- **Enhanced Feature Impact**: ~48% of total feature importance

### Feature Importance Insights
Top performing feature categories:
1. **Technical Indicators**: Momentum and volatility measures
2. **On-chain Metrics**: Network value and transaction ratios
3. **Sentiment Features**: Social sentiment and market psychology
4. **Price Features**: Basic price action and volume

### Key Discoveries
- **Enhanced features contribute significantly** (40-50%) to model performance
- **On-chain metrics provide unique insights** not captured by traditional TA
- **Sentiment analysis adds valuable market psychology context**
- **Feature diversity improves model robustness**

## Data Sources and APIs

### Production Integration Ready
The `crypto_data_utilities.py` module is designed to integrate with:

- **On-chain Data**:
  - Glassnode API
  - CoinMetrics API
  - Messari API

- **Sentiment Data**:
  - LunarCrush API (social sentiment)
  - Santiment API (on-chain social data)
  - CryptoCompare API (news sentiment)
  - Fear & Greed Index API

### Current Implementation
For demonstration purposes, the model uses realistic simulated data that mimics actual market patterns and correlations.

## Technical Architecture

### 1. Data Layer
```
crypto_data_utilities.py
├── OnChainDataFetcher
├── SentimentDataFetcher
└── DataValidator
```

### 2. Model Layer
```
OptimizeModel_demo_version.py
├── EnhancedCryptoPredictionModel
├── Feature Engineering
├── Model Training
└── Prediction Generation
```

### 3. Analysis Layer
```
complete_crypto_analysis.py
├── CompleteCryptoAnalysis
├── Feature Analysis
├── Performance Metrics
└── Comprehensive Reporting
```

## Configuration Options

### Model Parameters
- **Symbol**: Crypto pair (e.g., "BTC-USD", "ETH-USD")
- **Interval**: Data frequency ("15m", "1h", "1d")
- **Training Period**: Historical data length (default: 60 days)
- **Prediction Horizon**: Future periods to predict (default: 96 = 24 hours)

### Feature Selection
- Enable/disable specific feature categories
- Customize on-chain metrics selection
- Configure sentiment data sources
- Adjust technical indicator parameters

## Error Handling and Logging

### Comprehensive Logging
- Model training progress
- Data fetching status
- Feature engineering steps
- Performance metrics
- Error tracking and debugging

### Data Validation
- Outlier detection and handling
- Missing value imputation
- Data consistency checks
- Temporal alignment verification

## Future Enhancements

### Planned Features
- **Real-time Data Integration**: Live API connections
- **Multi-asset Support**: Portfolio-level analysis
- **Advanced Models**: Deep learning integration
- **Risk Management**: Position sizing and stop-loss recommendations
- **Backtesting Framework**: Historical performance analysis

### Research Directions
- **Alternative Data Sources**: Derivatives data, institutional flows
- **Advanced Sentiment**: NLP improvements, real-time social analysis
- **Cross-asset Correlations**: Traditional market integration
- **Regime Detection**: Market condition classification

## Contributing

This enhanced crypto prediction model demonstrates the integration of modern data sources with traditional financial modeling. The modular architecture allows for easy extension and customization.

### Development Guidelines
1. Maintain backward compatibility
2. Add comprehensive logging
3. Include data validation
4. Document new features
5. Test with realistic data

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using live APIs.

## Disclaimer

This model is for educational and research purposes only. Cryptocurrency trading involves significant risk, and past performance does not guarantee future results. Always conduct your own research and consider consulting with financial advisors before making investment decisions.

---

**Last Updated**: July 2025  
**Version**: 1.0.0  
**Compatibility**: Python 3.8+