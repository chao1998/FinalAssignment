#!/usr/bin/env python3
"""
Demonstration script for OptimizeModel_fixed_id_matching.py

This script showcases the enhanced capabilities of the stock prediction model
including traditional technical indicators, on-chain metrics, and sentiment analysis.
"""

from OptimizeModel_fixed_id_matching import OptimizeModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_traditional_model():
    """Demonstrate backward compatibility with traditional technical indicators only."""
    print("\n" + "="*60)
    print("TRADITIONAL MODEL DEMONSTRATION (Backward Compatible)")
    print("="*60)
    
    model = OptimizeModel('TSLA', prediction_days=1)
    results = model.train_and_predict(enable_onchain=False, enable_sentiment=False)
    
    if results:
        print(f"Symbol: {results['symbol']}")
        print(f"Model Used: {results['model_used']}")
        print(f"Latest Prediction: ${results['predictions'][-1]:.2f}")
        print(f"Recent Actual Prices: {[f'${p:.2f}' for p in results['recent_prices'][-3:]]}")
        print(f"Model Performance (R²): {results['model_metrics']['r2']:.4f}")
        
        # Show top features
        importance = model.get_feature_importance()
        if not importance.empty:
            print("\nTop 5 Traditional Features:")
            for idx, row in importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")


def demonstrate_enhanced_model():
    """Demonstrate enhanced model with all features."""
    print("\n" + "="*60)
    print("ENHANCED MODEL DEMONSTRATION (All Features)")
    print("="*60)
    
    model = OptimizeModel('BTC-USD', prediction_days=1)
    results = model.train_and_predict(enable_onchain=True, enable_sentiment=True)
    
    if results:
        print(f"Symbol: {results['symbol']}")
        print(f"Model Used: {results['model_used']}")
        print(f"Latest Prediction: ${results['predictions'][-1]:.2f}")
        print(f"Recent Actual Prices: {[f'${p:.2f}' for p in results['recent_prices'][-3:]]}")
        print(f"Model Performance (R²): {results['model_metrics']['r2']:.4f}")
        
        # Show top features
        importance = model.get_feature_importance()
        if not importance.empty:
            print("\nTop 10 Enhanced Features (Technical + On-Chain + Sentiment):")
            for idx, row in importance.head(10).iterrows():
                feature_type = "Unknown"
                if any(x in row['feature'].lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb_']):
                    feature_type = "Technical"
                elif any(x in row['feature'].lower() for x in ['hash_rate', 'nvt', 'mvrv', 'transaction', 'active_addresses']):
                    feature_type = "On-Chain"
                elif any(x in row['feature'].lower() for x in ['sentiment', 'fear_greed', 'social', 'reddit', 'twitter']):
                    feature_type = "Sentiment"
                elif 'lag' in row['feature'].lower():
                    feature_type = "Lagged"
                    
                print(f"  {row['feature']} ({feature_type}): {row['importance']:.4f}")


def demonstrate_multiple_models():
    """Demonstrate different model types."""
    print("\n" + "="*60)
    print("MULTIPLE MODELS COMPARISON")
    print("="*60)
    
    model = OptimizeModel('AAPL', prediction_days=1)
    
    # Train once with all features
    data = model.load_all_data()
    if not data.empty:
        features_data = model.create_advanced_features(data)
        X, y, feature_names = model.prepare_features(features_data)
        model.train_models(X, y, feature_names)
        
        # Compare predictions from different models
        models_to_test = ['random_forest', 'gradient_boosting', 'linear_regression']
        
        print("Model Performance Comparison:")
        for model_name in models_to_test:
            if model_name in model.models:
                metrics = model.models[model_name]
                print(f"\n{model_name.replace('_', ' ').title()}:")
                print(f"  R² Score: {metrics['r2']:.4f}")
                print(f"  MSE: {metrics['mse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                
                # Get prediction
                results = model.predict(features_data, model_name)
                if results:
                    print(f"  Latest Prediction: ${results['predictions'][-1]:.2f}")


def demonstrate_feature_types():
    """Demonstrate the different types of features created."""
    print("\n" + "="*60)
    print("FEATURE TYPES DEMONSTRATION")
    print("="*60)
    
    from OptimizeModel_fixed_id_matching import DataLoader, FeatureEngineer
    
    # Load sample data
    loader = DataLoader('TSLA', '1y')
    stock_data = loader.load_stock_data()
    onchain_data = loader.load_onchain_metrics()
    sentiment_data = loader.load_sentiment_data()
    
    engineer = FeatureEngineer()
    
    # Show original data shapes
    print(f"Original Stock Data: {stock_data.shape}")
    print(f"Original On-Chain Data: {onchain_data.shape}")
    print(f"Original Sentiment Data: {sentiment_data.shape}")
    
    # Create features and show expansion
    tech_data = engineer.create_technical_indicators(stock_data)
    onchain_enhanced = engineer.create_onchain_features(onchain_data)
    sentiment_enhanced = engineer.create_sentiment_features(sentiment_data)
    
    print(f"\nAfter Technical Indicators: {tech_data.shape}")
    print(f"Technical Features Added: {tech_data.shape[1] - stock_data.shape[1]}")
    
    print(f"\nAfter On-Chain Features: {onchain_enhanced.shape}")
    print(f"On-Chain Features Added: {onchain_enhanced.shape[1] - onchain_data.shape[1]}")
    
    print(f"\nAfter Sentiment Features: {sentiment_enhanced.shape}")
    print(f"Sentiment Features Added: {sentiment_enhanced.shape[1] - sentiment_data.shape[1]}")
    
    # Show sample feature names by category
    tech_features = [col for col in tech_data.columns if col not in stock_data.columns]
    onchain_features = [col for col in onchain_enhanced.columns if col not in onchain_data.columns]
    sentiment_features = [col for col in sentiment_enhanced.columns if col not in sentiment_data.columns]
    
    print(f"\nSample Technical Features: {tech_features[:5]}")
    print(f"Sample On-Chain Features: {onchain_features[:5]}")
    print(f"Sample Sentiment Features: {sentiment_features[:5]}")


def main():
    """Run all demonstrations."""
    print("OptimizeModel Enhanced Features Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate traditional model (backward compatibility)
        demonstrate_traditional_model()
        
        # Demonstrate enhanced model with all features
        demonstrate_enhanced_model()
        
        # Demonstrate multiple models
        demonstrate_multiple_models()
        
        # Demonstrate feature types
        demonstrate_feature_types()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Backward compatibility with traditional technical indicators")
        print("✓ Enhanced on-chain metrics for cryptocurrency analysis")
        print("✓ Sentiment analysis integration from multiple sources")
        print("✓ Multiple machine learning models (RF, GB, Linear)")
        print("✓ Feature importance analysis and interpretation")
        print("✓ Robust error handling and data validation")
        print("✓ Comprehensive feature engineering pipeline")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")


if __name__ == "__main__":
    main()