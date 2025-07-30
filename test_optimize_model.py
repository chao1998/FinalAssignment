#!/usr/bin/env python3
"""
Test script for OptimizeModel_fixed_id_matching.py

Tests the functionality of:
1. Data loading (stock, on-chain, sentiment)
2. Feature engineering
3. Model training
4. Prediction generation
5. Backward compatibility
"""

import sys
import os
import logging
from OptimizeModel_fixed_id_matching import OptimizeModel, DataLoader, FeatureEngineer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loader():
    """Test DataLoader functionality."""
    logger.info("Testing DataLoader...")
    
    # Test stock data loading
    loader = DataLoader("TSLA", "1y")
    stock_data = loader.load_stock_data()
    assert not stock_data.empty, "Stock data should not be empty"
    assert 'close' in stock_data.columns, "Stock data should have 'close' column"
    assert 'volume' in stock_data.columns, "Stock data should have 'volume' column"
    logger.info(f"✓ Stock data loaded: {len(stock_data)} rows")
    
    # Test on-chain data loading
    onchain_data = loader.load_onchain_metrics()
    assert not onchain_data.empty, "On-chain data should not be empty"
    assert 'nvt_ratio' in onchain_data.columns, "On-chain data should have NVT ratio"
    assert 'mvrv_ratio' in onchain_data.columns, "On-chain data should have MVRV ratio"
    logger.info(f"✓ On-chain data loaded: {len(onchain_data)} rows")
    
    # Test sentiment data loading
    sentiment_data = loader.load_sentiment_data()
    assert not sentiment_data.empty, "Sentiment data should not be empty"
    assert 'composite_sentiment' in sentiment_data.columns, "Sentiment data should have composite sentiment"
    assert 'fear_greed_index' in sentiment_data.columns, "Sentiment data should have Fear & Greed index"
    logger.info(f"✓ Sentiment data loaded: {len(sentiment_data)} rows")


def test_feature_engineer():
    """Test FeatureEngineer functionality."""
    logger.info("Testing FeatureEngineer...")
    
    # Create sample data
    loader = DataLoader("TSLA", "1y")
    stock_data = loader.load_stock_data()
    onchain_data = loader.load_onchain_metrics()
    sentiment_data = loader.load_sentiment_data()
    
    engineer = FeatureEngineer()
    
    # Test technical indicators
    tech_features = engineer.create_technical_indicators(stock_data)
    original_cols = len(stock_data.columns)
    new_cols = len(tech_features.columns)
    assert new_cols > original_cols, "Technical indicators should add new columns"
    assert 'rsi' in tech_features.columns, "Should have RSI indicator"
    assert 'macd' in tech_features.columns, "Should have MACD indicator"
    logger.info(f"✓ Technical indicators created: {new_cols - original_cols} new features")
    
    # Test on-chain features
    onchain_features = engineer.create_onchain_features(onchain_data)
    original_cols = len(onchain_data.columns)
    new_cols = len(onchain_features.columns)
    assert new_cols > original_cols, "On-chain features should add new columns"
    logger.info(f"✓ On-chain features created: {new_cols - original_cols} new features")
    
    # Test sentiment features
    sentiment_features = engineer.create_sentiment_features(sentiment_data)
    original_cols = len(sentiment_data.columns)
    new_cols = len(sentiment_features.columns)
    assert new_cols > original_cols, "Sentiment features should add new columns"
    logger.info(f"✓ Sentiment features created: {new_cols - original_cols} new features")


def test_optimize_model():
    """Test OptimizeModel functionality."""
    logger.info("Testing OptimizeModel...")
    
    # Initialize model
    model = OptimizeModel("TSLA", prediction_days=1)
    assert model.symbol == "TSLA", "Model should store correct symbol"
    assert model.prediction_days == 1, "Model should store correct prediction days"
    logger.info("✓ Model initialized correctly")
    
    # Test data loading
    data = model.load_all_data()
    assert not data.empty, "Combined data should not be empty"
    logger.info(f"✓ Combined data loaded: {data.shape}")
    
    # Test feature creation
    features_data = model.create_advanced_features(data)
    assert not features_data.empty, "Features data should not be empty"
    assert 'target' in features_data.columns, "Should have target variable"
    logger.info(f"✓ Advanced features created: {features_data.shape}")
    
    # Test feature preparation
    X, y, feature_names = model.prepare_features(features_data)
    assert len(X) > 0, "Feature matrix should not be empty"
    assert len(y) > 0, "Target vector should not be empty"
    assert len(feature_names) > 0, "Feature names should not be empty"
    logger.info(f"✓ Features prepared: {X.shape}, target: {y.shape}")


def test_training_and_prediction():
    """Test model training and prediction."""
    logger.info("Testing model training and prediction...")
    
    # Test with all features
    model = OptimizeModel("TSLA", prediction_days=1)
    results = model.train_and_predict(enable_onchain=True, enable_sentiment=True)
    
    assert results is not None, "Results should not be None"
    assert 'predictions' in results, "Results should contain predictions"
    assert 'model_metrics' in results, "Results should contain model metrics"
    assert len(results['predictions']) > 0, "Should have predictions"
    logger.info(f"✓ Training with all features completed: {len(results['predictions'])} predictions")
    
    # Test feature importance
    importance = model.get_feature_importance()
    assert not importance.empty, "Feature importance should not be empty"
    logger.info(f"✓ Feature importance extracted: {len(importance)} features")


def test_backward_compatibility():
    """Test backward compatibility with traditional features only."""
    logger.info("Testing backward compatibility...")
    
    model = OptimizeModel("TSLA", prediction_days=1)
    results = model.train_and_predict(enable_onchain=False, enable_sentiment=False)
    
    assert results is not None, "Traditional model results should not be None"
    assert 'predictions' in results, "Results should contain predictions"
    assert len(results['predictions']) > 0, "Should have predictions"
    logger.info("✓ Backward compatibility confirmed")


def test_model_save_load():
    """Test model saving and loading functionality."""
    logger.info("Testing model save/load functionality...")
    
    # Train a model
    model1 = OptimizeModel("TSLA", prediction_days=1)
    results1 = model1.train_and_predict(enable_onchain=False, enable_sentiment=False)
    
    # Save the model
    model_path = "/tmp/test_model.pkl"
    model1.save_model(model_path)
    assert os.path.exists(model_path), "Model file should be created"
    logger.info("✓ Model saved successfully")
    
    # Load the model
    model2 = OptimizeModel("AAPL", prediction_days=5)  # Different initial settings
    model2.load_model(model_path)
    
    # Check if settings were loaded correctly
    assert model2.symbol == "TSLA", "Loaded symbol should match saved model"
    assert model2.prediction_days == 1, "Loaded prediction days should match saved model"
    assert model2.is_trained, "Loaded model should be marked as trained"
    logger.info("✓ Model loaded successfully")
    
    # Clean up
    os.remove(model_path)


def run_all_tests():
    """Run all tests."""
    logger.info("Starting comprehensive tests...")
    
    try:
        test_data_loader()
        test_feature_engineer()
        test_optimize_model()
        test_training_and_prediction()
        test_backward_compatibility()
        test_model_save_load()
        
        logger.info("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)