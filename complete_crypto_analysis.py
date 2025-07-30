#!/usr/bin/env python3
"""
Complete Enhanced Crypto Prediction Model Integration
====================================================

This script demonstrates the complete integration of the enhanced crypto prediction model
with on-chain metrics and sentiment analysis using the data utility functions.

Features:
- Integration with crypto_data_utilities module
- Complete feature engineering pipeline
- Model training with 29+ features
- Performance analysis and feature importance
- Prediction generation
- Comprehensive reporting
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from crypto_data_utilities import fetch_enhanced_crypto_data, DataValidator
from OptimizeModel_demo_version import EnhancedCryptoPredictionModel

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_crypto_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteCryptoAnalysis:
    """Complete crypto analysis with enhanced features"""
    
    def __init__(self, symbol: str = "BTC-USD", interval: str = "15m"):
        self.symbol = symbol
        self.crypto_symbol = symbol.split('-')[0]  # Extract crypto part (BTC from BTC-USD)
        self.interval = interval
        self.model = EnhancedCryptoPredictionModel(symbol, interval)
        self.enhanced_data = {}
        
        logger.info(f"Initialized Complete Crypto Analysis for {symbol}")
    
    def fetch_all_data(self, days: int = 60) -> Dict:
        """Fetch all required data for analysis"""
        try:
            logger.info(f"Fetching all data for {self.crypto_symbol}")
            
            # Generate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Generate realistic price data (since we can't fetch real data)
            price_data = self.model.simulate_realistic_price_data(days=days)
            
            # Fetch enhanced data using utilities
            enhanced_data = fetch_enhanced_crypto_data(
                self.crypto_symbol, start_date, end_date, self.interval
            )
            
            # Align data to match price data timestamps
            price_index = price_data.index
            
            # Resample enhanced data to match price data
            aligned_enhanced_data = {}
            for data_type, df in enhanced_data.items():
                if len(df) > 0:
                    # Resample to match price data frequency and index
                    df_resampled = df.reindex(price_index, method='ffill').fillna(method='bfill')
                    aligned_enhanced_data[data_type] = df_resampled
            
            self.enhanced_data = aligned_enhanced_data
            
            return {
                'price_data': price_data,
                'enhanced_data': aligned_enhanced_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching all data: {e}")
            raise
    
    def create_comprehensive_features(self, data_dict: Dict) -> pd.DataFrame:
        """Create comprehensive feature set from all data sources"""
        try:
            logger.info("Creating comprehensive feature set")
            
            price_data = data_dict['price_data']
            enhanced_data = data_dict['enhanced_data']
            
            # Use the model's existing feature computation but with real enhanced data
            # Extract the specific features we need
            onchain_features = enhanced_data['onchain_data'][
                ['transaction_count', 'active_addresses', 'hashrate_change_pct', 
                 'mvrv_ratio', 'nvt_ratio']
            ].copy()
            
            # Add exchange volume ratio calculation
            onchain_features['onchain_exchange_volume_ratio'] = (
                enhanced_data['onchain_data']['transaction_volume_usd'] / 
                (price_data['Volume'] * price_data['Close'] + 1e-8)
            )
            
            sentiment_features = enhanced_data['sentiment_data'][
                ['social_sentiment_score', 'news_sentiment_indicator', 'fear_greed_index',
                 'volatility_sentiment', 'google_trends_proxy', 'twitter_mentions',
                 'reddit_engagement_score']
            ].copy()
            
            # Compute factors using the model's method
            comprehensive_features = self.model.compute_factors_numpy(
                price_data, onchain_features, sentiment_features
            )
            
            logger.info(f"Created comprehensive features with {len(comprehensive_features.columns)-1} features")
            
            return comprehensive_features
            
        except Exception as e:
            logger.error(f"Error creating comprehensive features: {e}")
            raise
    
    def run_complete_analysis(self, days: int = 60) -> Dict:
        """Run complete enhanced crypto analysis"""
        try:
            logger.info("Starting complete enhanced crypto analysis")
            
            # Fetch all data
            all_data = self.fetch_all_data(days)
            
            # Create comprehensive features
            features_df = self.create_comprehensive_features(all_data)
            
            # Train model
            training_results = self.model.train_model(features_df)
            
            # Generate predictions
            predictions = self.model.generate_predictions(features_df)
            
            # Analyze feature categories
            category_analysis = self.model.analyze_feature_categories()
            
            # Create enhanced feature analysis
            enhanced_feature_analysis = self._analyze_enhanced_features(training_results)
            
            # Plot feature importance
            self.model.plot_feature_importance(top_n=25)
            
            # Data quality assessment
            data_quality = self._assess_data_quality(all_data, features_df)
            
            results = {
                'training_results': training_results,
                'predictions': predictions,
                'features_df': features_df,
                'category_analysis': category_analysis,
                'enhanced_feature_analysis': enhanced_feature_analysis,
                'data_quality': data_quality,
                'data_summary': {
                    'total_features': len(self.model.feature_names),
                    'training_samples': training_results['train_samples'],
                    'test_samples': training_results['test_samples'],
                    'price_data_points': len(all_data['price_data']),
                    'prediction_horizon': len(predictions),
                    'enhanced_data_sources': len(self.enhanced_data)
                }
            }
            
            logger.info("Complete enhanced analysis finished successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
    
    def _analyze_enhanced_features(self, training_results: Dict) -> Dict:
        """Analyze the contribution of enhanced features"""
        try:
            feature_importance = training_results['feature_importance']
            
            # Define enhanced features (on-chain + sentiment)
            enhanced_features = [
                'transaction_count', 'active_addresses', 'hashrate_change_pct',
                'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio',
                'social_sentiment_score', 'news_sentiment_indicator', 'fear_greed_index',
                'volatility_sentiment', 'google_trends_proxy', 'twitter_mentions',
                'reddit_engagement_score'
            ]
            
            traditional_features = [f for f in feature_importance.keys() if f not in enhanced_features]
            
            # Calculate importance scores
            enhanced_importance = sum(feature_importance.get(f, 0) for f in enhanced_features)
            traditional_importance = sum(feature_importance.get(f, 0) for f in traditional_features)
            total_importance = enhanced_importance + traditional_importance
            
            # Find top contributors in each category
            top_enhanced = sorted(
                [(f, feature_importance[f]) for f in enhanced_features if f in feature_importance],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            top_traditional = sorted(
                [(f, feature_importance[f]) for f in traditional_features if f in feature_importance],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            return {
                'enhanced_features_count': len([f for f in enhanced_features if f in feature_importance]),
                'traditional_features_count': len(traditional_features),
                'enhanced_importance': enhanced_importance,
                'traditional_importance': traditional_importance,
                'enhanced_percentage': (enhanced_importance / total_importance * 100) if total_importance > 0 else 0,
                'traditional_percentage': (traditional_importance / total_importance * 100) if total_importance > 0 else 0,
                'top_enhanced_features': top_enhanced,
                'top_traditional_features': top_traditional
            }
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced features: {e}")
            return {}
    
    def _assess_data_quality(self, all_data: Dict, features_df: pd.DataFrame) -> Dict:
        """Assess data quality metrics"""
        try:
            price_data = all_data['price_data']
            enhanced_data = all_data['enhanced_data']
            
            quality_metrics = {
                'price_data_completeness': 1 - price_data.isnull().sum().sum() / (len(price_data) * len(price_data.columns)),
                'enhanced_data_completeness': {},
                'feature_completeness': 1 - features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns)),
                'data_consistency': True,  # Would check for realistic ranges
                'temporal_alignment': len(features_df) / len(price_data)  # How well data aligns
            }
            
            for data_type, df in enhanced_data.items():
                quality_metrics['enhanced_data_completeness'][data_type] = (
                    1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
                )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {}

def print_comprehensive_results(results: Dict):
    """Print comprehensive analysis results"""
    try:
        print("\n" + "="*100)
        print("🚀 COMPLETE ENHANCED CRYPTO PREDICTION MODEL - COMPREHENSIVE ANALYSIS")
        print("="*100)
        
        # Data Summary
        data_summary = results['data_summary']
        print(f"\n📊 DATA SUMMARY:")
        print(f"   • Symbol: BTC-USD (15-minute intervals)")
        print(f"   • Total Features: {data_summary['total_features']}")
        print(f"   • Enhanced Data Sources: {data_summary['enhanced_data_sources']}")
        print(f"   • Training Samples: {data_summary['training_samples']:,}")
        print(f"   • Test Samples: {data_summary['test_samples']:,}")
        print(f"   • Price Data Points: {data_summary['price_data_points']:,}")
        print(f"   • Prediction Horizon: {data_summary['prediction_horizon']} periods (24 hours)")
        
        # Model Performance
        training_results = results['training_results']
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   • Training R²: {training_results['train_r2']:.4f}")
        print(f"   • Test R²: {training_results['test_r2']:.4f}")
        print(f"   • Training RMSE: {np.sqrt(training_results['train_mse']):.6f}")
        print(f"   • Test RMSE: {np.sqrt(training_results['test_mse']):.6f}")
        print(f"   • Training MAE: {training_results['train_mae']:.6f}")
        print(f"   • Test MAE: {training_results['test_mae']:.6f}")
        
        # Enhanced Features Analysis
        enhanced_analysis = results['enhanced_feature_analysis']
        if enhanced_analysis:
            print(f"\n🔬 ENHANCED FEATURES ANALYSIS:")
            print(f"   • Enhanced Features: {enhanced_analysis['enhanced_features_count']} (On-chain + Sentiment)")
            print(f"   • Traditional Features: {enhanced_analysis['traditional_features_count']} (Price + Technical)")
            print(f"   • Enhanced Contribution: {enhanced_analysis['enhanced_percentage']:.1f}%")
            print(f"   • Traditional Contribution: {enhanced_analysis['traditional_percentage']:.1f}%")
            
            print(f"\n   🔝 Top Enhanced Features:")
            for i, (feature, importance) in enumerate(enhanced_analysis['top_enhanced_features'], 1):
                print(f"      {i}. {feature:<35} | {importance:.4f}")
        
        # Feature Categories
        category_analysis = results['category_analysis']
        print(f"\n📈 FEATURE CATEGORY BREAKDOWN:")
        for category, stats in category_analysis.items():
            print(f"   • {category:<12} | Features: {stats['feature_count']:2d} | " + 
                  f"Total Importance: {stats['total_importance']:.3f} | " + 
                  f"Average: {stats['average_importance']:.4f}")
        
        # Top Overall Features
        print(f"\n🏆 TOP 20 MOST IMPORTANT FEATURES:")
        sorted_importance = sorted(training_results['feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:20]):
            # Determine feature type
            if feature in ['price', 'volume', 'high_low_ratio', 'open_close_ratio', 'price_change_pct']:
                feature_type = "📈 Price    "
            elif feature in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'bb_width', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20']:
                feature_type = "🔧 Technical"
            elif feature in ['transaction_count', 'active_addresses', 'hashrate_change_pct', 'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio']:
                feature_type = "⛓️  On-chain "
            else:
                feature_type = "😊 Sentiment"
            
            print(f"   {i+1:2d}. {feature_type} | {feature:<35} | {importance:.4f}")
        
        # Predictions
        predictions = results['predictions']
        print(f"\n🔮 PREDICTION ANALYSIS:")
        print(f"   • Next 1h avg return: {predictions['predicted_return'][:4].mean():.4f}")
        print(f"   • Next 4h avg return: {predictions['predicted_return'][:16].mean():.4f}")
        print(f"   • Next 12h avg return: {predictions['predicted_return'][:48].mean():.4f}")
        print(f"   • Next 24h cumulative return: {predictions['cumulative_return'].iloc[-1]:.4f}")
        print(f"   • Prediction volatility: {predictions['predicted_return'].std():.4f}")
        
        # Data Quality
        data_quality = results['data_quality']
        if data_quality:
            print(f"\n📋 DATA QUALITY ASSESSMENT:")
            print(f"   • Price Data Completeness: {data_quality['price_data_completeness']:.1%}")
            print(f"   • Feature Data Completeness: {data_quality['feature_completeness']:.1%}")
            print(f"   • Temporal Alignment: {data_quality['temporal_alignment']:.1%}")
            
            print(f"   • Enhanced Data Completeness:")
            for data_type, completeness in data_quality['enhanced_data_completeness'].items():
                print(f"     - {data_type}: {completeness:.1%}")
        
        # Key Insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Calculate impact of new features
        enhanced_impact = enhanced_analysis.get('enhanced_percentage', 0)
        if enhanced_impact > 40:
            impact_level = "Significant"
        elif enhanced_impact > 25:
            impact_level = "Moderate"
        else:
            impact_level = "Limited"
        
        print(f"   • Enhanced Features Impact: {impact_level} ({enhanced_impact:.1f}% of total importance)")
        
        # Model performance assessment
        test_r2 = training_results['test_r2']
        if test_r2 > 0.3:
            performance_level = "Strong"
        elif test_r2 > 0.1:
            performance_level = "Moderate"
        elif test_r2 > 0:
            performance_level = "Weak"
        else:
            performance_level = "Poor (overfitting likely)"
        
        print(f"   • Model Performance: {performance_level} (R² = {test_r2:.4f})")
        
        # Feature diversity
        feature_categories = len(category_analysis)
        print(f"   • Feature Diversity: {feature_categories} categories providing comprehensive market view")
        
        # Top category by importance
        top_category = max(category_analysis.items(), key=lambda x: x[1]['total_importance'])
        print(f"   • Most Important Category: {top_category[0]} ({top_category[1]['total_importance']:.3f} total importance)")
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"   • Enhanced model with {data_summary['total_features']} features from {data_summary['enhanced_data_sources']} data sources")
        print(f"   • Successfully integrated on-chain metrics and sentiment analysis")
        print(f"   • Feature importance visualization saved as 'feature_importance_demo.png'")
        print(f"   • Complete analysis logs saved in 'complete_crypto_analysis.log'")
        print("="*100)
        
    except Exception as e:
        logger.error(f"Error printing results: {e}")
        print(f"❌ Error displaying results: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize complete analysis
        analysis = CompleteCryptoAnalysis(symbol="BTC-USD", interval="15m")
        
        # Run complete analysis
        results = analysis.run_complete_analysis(days=60)
        
        # Print comprehensive results
        print_comprehensive_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()