#!/usr/bin/env python3
"""
Enhanced Crypto Prediction Model with On-chain Metrics and Sentiment Index
===========================================================================

This enhanced version of the crypto prediction model incorporates:
1. On-chain metrics: Transaction volume, active addresses, network hash rate, etc.
2. Sentiment index: Market sentiment from social media, news, and psychology indicators
3. Technical indicators: RSI, MACD, ATR, Bollinger Bands (existing)
4. Price and volume data (existing)

Features:
- 15+ new features from on-chain and sentiment data
- 15-minute interval structure maintained
- Proper scaling and normalization
- Feature importance analysis
- Comprehensive error handling and logging
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedCryptoPredictionModel:
    """
    Enhanced Crypto Prediction Model with On-chain Metrics and Sentiment Analysis
    """
    
    def __init__(self, symbol: str = "BTC-USD", interval: str = "15m"):
        self.symbol = symbol
        self.interval = interval
        self.scaler = RobustScaler()
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        
        logger.info(f"Initialized Enhanced Crypto Model for {symbol} with {interval} intervals")
    
    def fetch_price_data(self, period: str = "30d") -> pd.DataFrame:
        """Fetch crypto price data using yfinance"""
        try:
            logger.info(f"Fetching price data for {self.symbol}")
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=self.interval)
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            logger.info(f"Fetched {len(data)} price data points")
            return data
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            raise
    
    def simulate_onchain_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate realistic on-chain metrics data for demonstration
        In production, this would connect to blockchain APIs like CoinMetrics, Glassnode, etc.
        """
        try:
            logger.info("Generating simulated on-chain metrics")
            np.random.seed(42)  # For reproducible results
            
            n_points = len(data)
            
            # Base trends that correlate with price movement
            price_trend = data['Close'].pct_change().fillna(0)
            
            # Transaction count per interval (correlates with price activity)
            base_tx_count = 15000  # Average transactions per 15min
            tx_volatility = 0.3
            transaction_count = base_tx_count * (1 + price_trend * 2 + 
                                               np.random.normal(0, tx_volatility, n_points))
            transaction_count = np.maximum(transaction_count, 1000)  # Minimum threshold
            
            # Active addresses (trend following with lag)
            base_addresses = 8000  # Average active addresses per 15min
            addr_volatility = 0.25
            active_addresses = base_addresses * (1 + price_trend.rolling(4).mean().fillna(0) * 1.5 + 
                                               np.random.normal(0, addr_volatility, n_points))
            active_addresses = np.maximum(active_addresses, 2000)
            
            # Network hash rate changes (more stable, weekly patterns)
            base_hashrate = 1e18  # Base hash rate
            hashrate_trend = np.random.normal(0.001, 0.005, n_points)  # Small daily changes
            network_hashrate = base_hashrate * np.cumprod(1 + hashrate_trend)
            hashrate_change = np.gradient(network_hashrate) / network_hashrate
            
            # On-chain volume vs exchange volume ratio
            onchain_volume_base = data['Volume'] * np.random.uniform(0.6, 1.4, n_points)
            exchange_volume = data['Volume']
            onchain_exchange_ratio = onchain_volume_base / (exchange_volume + 1e-8)
            
            # MVRV Ratio (Market Value to Realized Value) - important on-chain metric
            mvrv_base = np.random.uniform(0.8, 3.2, n_points)
            mvrv_ratio = mvrv_base * (1 + price_trend.rolling(12).mean().fillna(0) * 0.8)
            
            # Network Value to Transactions (NVT) ratio
            market_cap_proxy = data['Close'] * 19_000_000  # Approximate circulating supply
            nvt_ratio = market_cap_proxy / (transaction_count * data['Close'] + 1e-8)
            
            onchain_df = pd.DataFrame({
                'transaction_count': transaction_count,
                'active_addresses': active_addresses,
                'hashrate_change_pct': hashrate_change * 100,
                'onchain_exchange_volume_ratio': onchain_exchange_ratio,
                'mvrv_ratio': mvrv_ratio,
                'nvt_ratio': nvt_ratio
            }, index=data.index)
            
            logger.info(f"Generated {len(onchain_df.columns)} on-chain features")
            return onchain_df
            
        except Exception as e:
            logger.error(f"Error generating on-chain data: {e}")
            raise
    
    def simulate_sentiment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate realistic sentiment data for demonstration
        In production, this would connect to APIs like LunarCrush, Santiment, CryptoCompare, etc.
        """
        try:
            logger.info("Generating simulated sentiment metrics")
            np.random.seed(43)  # Different seed for variety
            
            n_points = len(data)
            price_trend = data['Close'].pct_change().fillna(0)
            
            # Social media sentiment score (-1 to 1)
            social_sentiment_base = np.random.normal(0, 0.3, n_points)
            # Sentiment tends to follow price with some noise and lag
            social_sentiment = 0.4 * price_trend.rolling(3).mean().fillna(0) + \
                             0.6 * social_sentiment_base
            social_sentiment = np.clip(social_sentiment, -1, 1)
            
            # News sentiment indicator (-1 to 1)
            news_sentiment_base = np.random.normal(0.05, 0.25, n_points)  # Slightly positive bias
            news_sentiment = 0.3 * price_trend.rolling(6).mean().fillna(0) + \
                           0.7 * news_sentiment_base
            news_sentiment = np.clip(news_sentiment, -1, 1)
            
            # Fear & Greed index (0 to 100)
            fear_greed_base = 50 + np.random.normal(0, 15, n_points)
            # Fear & Greed inversely correlates with volatility and follows price trends
            volatility = data['Close'].rolling(24).std() / data['Close'].rolling(24).mean()
            fear_greed_index = fear_greed_base + price_trend.rolling(12).mean().fillna(0) * 30 - \
                             volatility.fillna(0) * 200
            fear_greed_index = np.clip(fear_greed_index, 0, 100)
            
            # Market volatility sentiment (0 to 1, where 1 is high volatility fear)
            volatility_sentiment = np.clip(volatility.fillna(0) * 5, 0, 1)
            
            # Google Trends proxy (0 to 100)
            google_trends_base = 20 + np.random.normal(0, 10, n_points)
            google_trends = google_trends_base + np.abs(price_trend) * 50
            google_trends = np.clip(google_trends, 0, 100)
            
            # Twitter mention count (proxy for social engagement)
            twitter_mentions_base = 5000 + np.random.normal(0, 1500, n_points)
            twitter_mentions = twitter_mentions_base + np.abs(price_trend) * 8000
            twitter_mentions = np.maximum(twitter_mentions, 500)
            
            # Reddit engagement score
            reddit_engagement_base = np.random.uniform(0.2, 0.8, n_points)
            reddit_engagement = reddit_engagement_base + price_trend.rolling(2).mean().fillna(0) * 0.3
            reddit_engagement = np.clip(reddit_engagement, 0, 1)
            
            sentiment_df = pd.DataFrame({
                'social_sentiment_score': social_sentiment,
                'news_sentiment_indicator': news_sentiment,
                'fear_greed_index': fear_greed_index,
                'volatility_sentiment': volatility_sentiment,
                'google_trends_proxy': google_trends,
                'twitter_mentions': twitter_mentions,
                'reddit_engagement_score': reddit_engagement
            }, index=data.index)
            
            logger.info(f"Generated {len(sentiment_df.columns)} sentiment features")
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Error generating sentiment data: {e}")
            raise
    
    def compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute traditional technical indicators"""
        try:
            logger.info("Computing technical indicators")
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = data['Close'].rolling(window=bb_period).mean()
            bb_std_dev = data['Close'].rolling(window=bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_position = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volume indicators
            volume_sma = data['Volume'].rolling(window=20).mean()
            volume_ratio = data['Volume'] / volume_sma
            
            # Price momentum
            momentum_5 = data['Close'].pct_change(5)
            momentum_10 = data['Close'].pct_change(10)
            momentum_20 = data['Close'].pct_change(20)
            
            technical_df = pd.DataFrame({
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'atr': atr,
                'bb_width': bb_width,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20
            }, index=data.index)
            
            logger.info(f"Generated {len(technical_df.columns)} technical features")
            return technical_df
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")
            raise
    
    def compute_factors_numpy(self, data: pd.DataFrame, onchain_data: pd.DataFrame, 
                            sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced factor computation including on-chain metrics and sentiment data
        """
        try:
            logger.info("Computing enhanced factors with on-chain and sentiment data")
            
            # Get technical indicators
            technical_features = self.compute_technical_indicators(data)
            
            # Basic price features
            price_features = pd.DataFrame({
                'price': data['Close'],
                'volume': data['Volume'],
                'high_low_ratio': data['High'] / data['Low'],
                'open_close_ratio': data['Open'] / data['Close'],
                'price_change_pct': data['Close'].pct_change(),
            }, index=data.index)
            
            # Combine all features
            all_features = pd.concat([
                price_features,
                technical_features,
                onchain_data,
                sentiment_data
            ], axis=1)
            
            # Create target variable (next period return)
            all_features['target'] = data['Close'].pct_change().shift(-1)
            
            # Remove rows with missing values
            all_features = all_features.dropna()
            
            self.feature_names = [col for col in all_features.columns if col != 'target']
            
            logger.info(f"Created {len(self.feature_names)} total features")
            logger.info(f"Feature breakdown: Price(5), Technical(11), On-chain(6), Sentiment(7)")
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error computing factors: {e}")
            raise
    
    def train_model(self, features_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train the enhanced prediction model"""
        try:
            logger.info("Training enhanced prediction model")
            
            X = features_df[self.feature_names]
            y = features_df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model with hyperparameter tuning
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf_model = RandomForestRegressor(random_state=42)
            rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            rf_grid.fit(X_train_scaled, y_train)
            
            self.model = rf_grid.best_estimator_
            
            # Predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Feature importance
            self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            results = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_params': rf_grid.best_params_,
                'feature_importance': self.feature_importance
            }
            
            logger.info(f"Model training completed. Test R²: {test_r2:.4f}, Test MSE: {test_mse:.6f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot feature importance analysis"""
        try:
            if not self.feature_importance:
                logger.warning("No feature importance data available")
                return
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Take top N features
            top_features = sorted_features[:top_n]
            features, importances = zip(*top_features)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(features)), importances)
            
            # Color code by feature type
            colors = []
            for feature in features:
                if feature in ['price', 'volume', 'high_low_ratio', 'open_close_ratio', 'price_change_pct']:
                    colors.append('skyblue')  # Price features
                elif feature in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'bb_width', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20']:
                    colors.append('lightgreen')  # Technical features
                elif feature in ['transaction_count', 'active_addresses', 'hashrate_change_pct', 'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio']:
                    colors.append('salmon')  # On-chain features
                else:
                    colors.append('gold')  # Sentiment features
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance - Enhanced Crypto Prediction Model')
            plt.tight_layout()
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='skyblue', label='Price Features'),
                Patch(facecolor='lightgreen', label='Technical Features'),
                Patch(facecolor='salmon', label='On-chain Features'),
                Patch(facecolor='gold', label='Sentiment Features')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Feature importance plot saved as 'feature_importance.png'")
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def generate_predictions(self, features_df: pd.DataFrame, periods: int = 96) -> pd.DataFrame:
        """Generate predictions for the next periods (96 = 24 hours for 15min intervals)"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train_model() first.")
            
            logger.info(f"Generating predictions for next {periods} periods")
            
            # Get the latest features
            latest_features = features_df[self.feature_names].iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            predictions = []
            current_features = latest_features_scaled.copy()
            
            for i in range(periods):
                pred = self.model.predict(current_features)[0]
                predictions.append(pred)
                
                # For simplicity, we'll use the prediction as feedback for the next prediction
                # In a real scenario, you'd update the features with new data
                current_features = current_features.copy()
                
            # Create prediction DataFrame
            last_timestamp = features_df.index[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(minutes=15),
                periods=periods,
                freq='15T'
            )
            
            predictions_df = pd.DataFrame({
                'predicted_return': predictions,
                'cumulative_return': np.cumprod(1 + np.array(predictions)) - 1
            }, index=future_timestamps)
            
            logger.info(f"Generated {len(predictions)} predictions")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete enhanced crypto prediction analysis"""
        try:
            logger.info("Starting complete enhanced crypto prediction analysis")
            
            # Fetch data
            price_data = self.fetch_price_data(period="60d")  # More data for better training
            
            # Generate enhanced features
            onchain_data = self.simulate_onchain_data(price_data)
            sentiment_data = self.simulate_sentiment_data(price_data)
            
            # Compute all factors
            features_df = self.compute_factors_numpy(price_data, onchain_data, sentiment_data)
            
            # Train model
            training_results = self.train_model(features_df)
            
            # Generate predictions
            predictions = self.generate_predictions(features_df)
            
            # Plot feature importance
            self.plot_feature_importance()
            
            # Compile results
            analysis_results = {
                'training_results': training_results,
                'predictions': predictions,
                'features_df': features_df,
                'data_summary': {
                    'total_features': len(self.feature_names),
                    'training_samples': len(features_df),
                    'price_data_points': len(price_data),
                    'prediction_horizon': len(predictions)
                }
            }
            
            logger.info("Complete analysis finished successfully")
            logger.info(f"Model Performance - R²: {training_results['test_r2']:.4f}, MSE: {training_results['test_mse']:.6f}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize the enhanced crypto prediction model
        model = EnhancedCryptoPredictionModel(symbol="BTC-USD", interval="15m")
        
        # Run complete analysis
        results = model.run_complete_analysis()
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED CRYPTO PREDICTION MODEL - ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n📊 Data Summary:")
        print(f"   • Total Features: {results['data_summary']['total_features']}")
        print(f"   • Training Samples: {results['data_summary']['training_samples']}")
        print(f"   • Price Data Points: {results['data_summary']['price_data_points']}")
        print(f"   • Prediction Horizon: {results['data_summary']['prediction_horizon']} periods (24 hours)")
        
        print(f"\n🎯 Model Performance:")
        train_results = results['training_results']
        print(f"   • Training R²: {train_results['train_r2']:.4f}")
        print(f"   • Test R²: {train_results['test_r2']:.4f}")
        print(f"   • Training MSE: {train_results['train_mse']:.6f}")
        print(f"   • Test MSE: {train_results['test_mse']:.6f}")
        
        print(f"\n🔝 Top 10 Most Important Features:")
        sorted_importance = sorted(train_results['feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:10]):
            feature_type = "📈 Price" if feature in ['price', 'volume', 'high_low_ratio', 'open_close_ratio', 'price_change_pct'] else \
                          "🔧 Technical" if feature in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'bb_width', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20'] else \
                          "⛓️  On-chain" if feature in ['transaction_count', 'active_addresses', 'hashrate_change_pct', 'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio'] else \
                          "😊 Sentiment"
            print(f"   {i+1:2d}. {feature_type} | {feature:<30} | {importance:.4f}")
        
        print(f"\n🔮 Prediction Summary:")
        predictions = results['predictions']
        print(f"   • Next 1h avg return: {predictions['predicted_return'][:4].mean():.4f}")
        print(f"   • Next 4h avg return: {predictions['predicted_return'][:16].mean():.4f}")
        print(f"   • Next 24h cumulative return: {predictions['cumulative_return'].iloc[-1]:.4f}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"   • Model saved in memory")
        print(f"   • Feature importance plot saved as 'feature_importance.png'")
        print(f"   • Logs saved in 'crypto_model.log'")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()