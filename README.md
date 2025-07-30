This is for Coursera course Python for Data Science.
Model final assignment - Enhanced with Advanced Features

## Enhanced OptimizeModel Implementation

This repository now includes a comprehensive stock prediction model (`OptimizeModel_fixed_id_matching.py`) with advanced features:

### Key Features
- **Traditional Technical Indicators** (backward compatible)
- **On-Chain Metrics** for cryptocurrency analysis
- **Sentiment Analysis** from multiple sources
- **Multiple ML Models** (Random Forest, Gradient Boosting, Linear Regression)
- **Feature Importance Analysis**
- **Robust Error Handling** with fallback data

### New Capabilities
1. **On-Chain Metrics**: Network hash rate, active addresses, transaction volume, NVT/MVRV ratios
2. **Sentiment Index**: Social media sentiment, news analysis, Fear & Greed index, Google Trends
3. **Advanced Feature Engineering**: 70+ engineered features across all data types
4. **Model Comparison**: Multiple algorithms with performance metrics

### Usage
```python
from OptimizeModel_fixed_id_matching import OptimizeModel

# Traditional model (backward compatible)
model = OptimizeModel('TSLA')
results = model.train_and_predict(enable_onchain=False, enable_sentiment=False)

# Enhanced model with all features
results = model.train_and_predict(enable_onchain=True, enable_sentiment=True)
```

### Files
- `OptimizeModel_fixed_id_matching.py` - Main enhanced model implementation
- `test_optimize_model.py` - Comprehensive test suite
- `demo_optimize_model.py` - Feature demonstration script
- `OptimizeModel_Documentation.md` - Detailed documentation
- `Final Assignment.ipynb` - Original Jupyter notebook
