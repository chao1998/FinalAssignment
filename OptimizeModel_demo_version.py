#!/usr/bin/env python3
"""
Enhanced Crypto Prediction Model - Demo Version with Simulated Data
==================================================================

This demo version uses simulated realistic crypto data to demonstrate the enhanced model
with on-chain metrics and sentiment analysis when live data is not available.

Features:
- 29 total features (5 price + 11 technical + 6 on-chain + 7 sentiment)
- Realistic data simulation based on actual crypto market patterns
- Complete feature importance analysis
- 15-minute interval structure
- Comprehensive error handling and logging
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
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
        logging.FileHandler('crypto_model_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedCryptoPredictionModel:
    """
    Enhanced Crypto Prediction Model with On-chain Metrics and Sentiment Analysis - Demo Version
    """
    
    def __init__(self, symbol: str = "BTC-USD", interval: str = "15m"):
        self.symbol = symbol
        self.interval = interval
        self.scaler = RobustScaler()
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        
        logger.info(f"Initialized Enhanced Crypto Model (Demo) for {symbol} with {interval} intervals")
    
    def simulate_realistic_price_data(self, days: int = 60) -> pd.DataFrame:
        """
        Generate realistic crypto price data for demonstration
        """
        try:
            logger.info(f"Generating realistic price data for {days} days")
            
            # Set random seed for reproducible results
            np.random.seed(42)
            
            # Calculate number of 15-minute intervals
            intervals_per_day = 96  # 24 * 4 (15-minute intervals)
            n_points = days * intervals_per_day
            
            # Create timestamp index
            start_date = datetime.now() - timedelta(days=days)
            timestamps = pd.date_range(start=start_date, periods=n_points, freq='15T')
            
            # Generate realistic price movement
            initial_price = 65000  # Starting BTC price
            
            # Daily trend (slight upward bias with cycles)
            daily_trend = 0.0001 + 0.0005 * np.sin(np.arange(n_points) / (intervals_per_day * 7))  # Weekly cycles
            
            # Hourly patterns (lower volatility during certain hours)
            hour_pattern = 0.8 + 0.2 * np.sin(2 * np.pi * np.arange(n_points) / (4 * 24))  # Daily cycle
            
            # Random walk with varying volatility
            base_volatility = 0.008  # Base 15-min volatility
            volatility_cycle = base_volatility * (0.8 + 0.4 * np.sin(np.arange(n_points) / (intervals_per_day * 3)))  # 3-day volatility cycle
            
            returns = daily_trend + volatility_cycle * hour_pattern * np.random.normal(0, 1, n_points)
            
            # Apply returns to generate price series
            price_series = initial_price * np.cumprod(1 + returns)
            
            # Generate OHLCV data
            # Open = previous close (with small gap)
            opens = np.concatenate([[initial_price], price_series[:-1]]) * (1 + np.random.normal(0, 0.001, n_points))
            
            # High and Low based on price with realistic ranges
            range_multiplier = 0.005 + 0.003 * np.abs(np.random.normal(0, 1, n_points))
            highs = price_series * (1 + range_multiplier * np.random.uniform(0.3, 1.0, n_points))
            lows = price_series * (1 - range_multiplier * np.random.uniform(0.3, 1.0, n_points))
            
            # Ensure OHLC consistency
            for i in range(n_points):
                highs[i] = max(highs[i], opens[i], price_series[i])
                lows[i] = min(lows[i], opens[i], price_series[i])
            
            # Volume (correlated with price movements and volatility)
            base_volume = 25000
            volume_volatility = np.abs(returns) * 50000  # Higher volume during big moves
            volume_trend = base_volume * (0.8 + 0.4 * np.random.lognormal(0, 0.5, n_points))
            volumes = volume_trend + volume_volatility
            
            price_data = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': price_series,
                'Volume': volumes
            }, index=timestamps)
            
            logger.info(f"Generated {len(price_data)} realistic price data points")
            logger.info(f"Price range: ${price_data['Close'].min():.0f} - ${price_data['Close'].max():.0f}")
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error generating price data: {e}")
            raise
    
    def simulate_onchain_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate realistic on-chain metrics data for demonstration
        """
        try:
            logger.info("Generating simulated on-chain metrics")
            np.random.seed(43)  # Different seed for variety
            
            n_points = len(data)
            
            # Base trends that correlate with price movement
            price_trend = data['Close'].pct_change().fillna(0)
            price_volatility = data['Close'].rolling(24).std() / data['Close'].rolling(24).mean()
            
            # Transaction count per interval (correlates with price activity)
            base_tx_count = 15000  # Average transactions per 15min
            tx_volatility = 0.3
            volume_factor = (data['Volume'] / data['Volume'].median() - 1) * 0.2
            transaction_count = base_tx_count * (1 + price_trend * 2 + volume_factor + 
                                               np.random.normal(0, tx_volatility, n_points))
            transaction_count = np.maximum(transaction_count, 1000)  # Minimum threshold
            
            # Active addresses (trend following with lag)
            base_addresses = 8000  # Average active addresses per 15min
            addr_volatility = 0.25
            active_addresses = base_addresses * (1 + price_trend.rolling(4).mean().fillna(0) * 1.5 + 
                                               np.random.normal(0, addr_volatility, n_points))
            active_addresses = np.maximum(active_addresses, 2000)
            
            # Network hash rate changes (more stable, weekly patterns)
            base_hashrate_change = 0.001  # Small daily changes
            hashrate_volatility = 0.005
            difficulty_adjustment_cycle = np.sin(np.arange(n_points) / (96 * 14)) * 0.002  # 2-week cycles
            hashrate_change = (base_hashrate_change + difficulty_adjustment_cycle + 
                             np.random.normal(0, hashrate_volatility, n_points))
            
            # On-chain volume vs exchange volume ratio
            onchain_base_ratio = 0.85  # Base ratio
            market_stress_factor = price_volatility.fillna(0) * 2  # Higher on-chain activity during stress
            onchain_exchange_ratio = onchain_base_ratio + market_stress_factor + np.random.normal(0, 0.15, n_points)
            onchain_exchange_ratio = np.clip(onchain_exchange_ratio, 0.2, 2.0)
            
            # MVRV Ratio (Market Value to Realized Value) - important on-chain metric
            mvrv_base = 1.2  # Neutral MVRV
            price_momentum = price_trend.rolling(12).mean().fillna(0)
            mvrv_ratio = mvrv_base + price_momentum * 0.8 + np.random.normal(0, 0.3, n_points)
            mvrv_ratio = np.maximum(mvrv_ratio, 0.3)  # Minimum bound
            
            # Network Value to Transactions (NVT) ratio
            market_cap_proxy = data['Close'] * 19_000_000  # Approximate circulating supply
            nvt_denominator = transaction_count * data['Close'] / 1e6  # Scale down
            nvt_ratio = market_cap_proxy / (nvt_denominator + 1e-8)
            nvt_ratio = np.clip(nvt_ratio, 10, 200)  # Reasonable bounds
            
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
        """
        try:
            logger.info("Generating simulated sentiment metrics")
            np.random.seed(44)  # Different seed for variety
            
            n_points = len(data)
            price_trend = data['Close'].pct_change().fillna(0)
            price_volatility = data['Close'].rolling(24).std() / data['Close'].rolling(24).mean()
            
            # Social media sentiment score (-1 to 1)
            social_sentiment_base = np.random.normal(0.05, 0.3, n_points)  # Slight positive bias
            # Sentiment tends to follow price with lag and amplification
            price_momentum = price_trend.rolling(3).mean().fillna(0)
            social_sentiment = 0.4 * price_momentum + 0.6 * social_sentiment_base
            # Add weekend effect (lower sentiment activity)
            weekend_effect = np.where(pd.to_datetime(data.index).weekday >= 5, -0.1, 0)
            social_sentiment = social_sentiment + weekend_effect
            social_sentiment = np.clip(social_sentiment, -1, 1)
            
            # News sentiment indicator (-1 to 1)
            news_sentiment_base = np.random.normal(0.1, 0.25, n_points)  # Positive bias for news
            long_term_trend = price_trend.rolling(6).mean().fillna(0)
            news_sentiment = 0.3 * long_term_trend + 0.7 * news_sentiment_base
            news_sentiment = np.clip(news_sentiment, -1, 1)
            
            # Fear & Greed index (0 to 100)
            fear_greed_base = 50 + np.random.normal(0, 15, n_points)
            # Fear & Greed inversely correlates with volatility and follows price trends
            volatility_factor = price_volatility.fillna(0) * (-200)  # High volatility = fear
            price_factor = price_trend.rolling(12).mean().fillna(0) * 30  # Price up = greed
            fear_greed_index = fear_greed_base + price_factor + volatility_factor
            fear_greed_index = np.clip(fear_greed_index, 0, 100)
            
            # Market volatility sentiment (0 to 1, where 1 is high volatility fear)
            volatility_sentiment = np.clip(price_volatility.fillna(0) * 5, 0, 1)
            
            # Google Trends proxy (0 to 100)
            google_trends_base = 20 + np.random.normal(0, 10, n_points)
            # Searches increase with absolute price movements
            search_factor = np.abs(price_trend) * 50
            google_trends = google_trends_base + search_factor
            google_trends = np.clip(google_trends, 0, 100)
            
            # Twitter mention count (proxy for social engagement)
            twitter_mentions_base = 5000 + np.random.normal(0, 1500, n_points)
            # Mentions spike with price movements and volatility
            mention_spike = (np.abs(price_trend) + price_volatility.fillna(0)) * 8000
            twitter_mentions = twitter_mentions_base + mention_spike
            twitter_mentions = np.maximum(twitter_mentions, 500)
            
            # Reddit engagement score (0 to 1)
            reddit_engagement_base = np.random.uniform(0.2, 0.8, n_points)
            engagement_boost = price_trend.rolling(2).mean().fillna(0) * 0.3
            reddit_engagement = reddit_engagement_base + engagement_boost
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
            rs = gain / (loss + 1e-8)  # Avoid division by zero
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
            volume_ratio = data['Volume'] / (volume_sma + 1e-8)
            
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
            
            # Split data chronologically (no shuffle for time series)
            split_idx = int(len(features_df) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
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
                'feature_importance': self.feature_importance,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model training completed. Test R²: {test_r2:.4f}, Test MSE: {test_mse:.6f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def plot_feature_importance(self, top_n: int = 20):
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
            plt.figure(figsize=(14, 10))
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
                Patch(facecolor='skyblue', label='Price Features (5)'),
                Patch(facecolor='lightgreen', label='Technical Features (11)'),
                Patch(facecolor='salmon', label='On-chain Features (6)'),
                Patch(facecolor='gold', label='Sentiment Features (7)')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.savefig('feature_importance_demo.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Feature importance plot saved as 'feature_importance_demo.png'")
            
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
            for i in range(periods):
                pred = self.model.predict(latest_features_scaled)[0]
                predictions.append(pred)
            
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
    
    def analyze_feature_categories(self) -> Dict:
        """Analyze feature importance by category"""
        try:
            if not self.feature_importance:
                return {}
            
            categories = {
                'Price': ['price', 'volume', 'high_low_ratio', 'open_close_ratio', 'price_change_pct'],
                'Technical': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'bb_width', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20'],
                'On-chain': ['transaction_count', 'active_addresses', 'hashrate_change_pct', 'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio'],
                'Sentiment': ['social_sentiment_score', 'news_sentiment_indicator', 'fear_greed_index', 'volatility_sentiment', 'google_trends_proxy', 'twitter_mentions', 'reddit_engagement_score']
            }
            
            category_importance = {}
            for category, features in categories.items():
                total_importance = sum(self.feature_importance.get(feature, 0) for feature in features)
                avg_importance = total_importance / len(features)
                category_importance[category] = {
                    'total_importance': total_importance,
                    'average_importance': avg_importance,
                    'feature_count': len(features)
                }
            
            return category_importance
            
        except Exception as e:
            logger.error(f"Error analyzing feature categories: {e}")
            return {}
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete enhanced crypto prediction analysis"""
        try:
            logger.info("Starting complete enhanced crypto prediction analysis (Demo Version)")
            
            # Generate realistic data
            price_data = self.simulate_realistic_price_data(days=60)
            
            # Generate enhanced features
            onchain_data = self.simulate_onchain_data(price_data)
            sentiment_data = self.simulate_sentiment_data(price_data)
            
            # Compute all factors
            features_df = self.compute_factors_numpy(price_data, onchain_data, sentiment_data)
            
            # Train model
            training_results = self.train_model(features_df)
            
            # Generate predictions
            predictions = self.generate_predictions(features_df)
            
            # Analyze feature categories
            category_analysis = self.analyze_feature_categories()
            
            # Plot feature importance
            self.plot_feature_importance()
            
            # Compile results
            analysis_results = {
                'training_results': training_results,
                'predictions': predictions,
                'features_df': features_df,
                'category_analysis': category_analysis,
                'data_summary': {
                    'total_features': len(self.feature_names),
                    'training_samples': training_results['train_samples'],
                    'test_samples': training_results['test_samples'],
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
        
        # Print comprehensive summary
        print("\n" + "="*90)
        print("🚀 ENHANCED CRYPTO PREDICTION MODEL - COMPLETE ANALYSIS RESULTS")
        print("="*90)
        
        print(f"\n📊 Data Summary:")
        data_summary = results['data_summary']
        print(f"   • Symbol: BTC-USD (15-minute intervals)")
        print(f"   • Total Features: {data_summary['total_features']}")
        print(f"   • Training Samples: {data_summary['training_samples']:,}")
        print(f"   • Test Samples: {data_summary['test_samples']:,}")
        print(f"   • Price Data Points: {data_summary['price_data_points']:,}")
        print(f"   • Prediction Horizon: {data_summary['prediction_horizon']} periods (24 hours)")
        
        print(f"\n🎯 Model Performance:")
        train_results = results['training_results']
        print(f"   • Training R²: {train_results['train_r2']:.4f}")
        print(f"   • Test R²: {train_results['test_r2']:.4f}")
        print(f"   • Training RMSE: {np.sqrt(train_results['train_mse']):.6f}")
        print(f"   • Test RMSE: {np.sqrt(train_results['test_mse']):.6f}")
        print(f"   • Training MAE: {train_results['train_mae']:.6f}")
        print(f"   • Test MAE: {train_results['test_mae']:.6f}")
        
        print(f"\n📈 Feature Category Analysis:")
        category_analysis = results['category_analysis']
        for category, stats in category_analysis.items():
            print(f"   • {category:<12} | Features: {stats['feature_count']:2d} | " + 
                  f"Total Importance: {stats['total_importance']:.3f} | " + 
                  f"Avg: {stats['average_importance']:.4f}")
        
        print(f"\n🔝 Top 15 Most Important Features:")
        sorted_importance = sorted(train_results['feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:15]):
            feature_type = "📈 Price    " if feature in ['price', 'volume', 'high_low_ratio', 'open_close_ratio', 'price_change_pct'] else \
                          "🔧 Technical" if feature in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'bb_width', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20'] else \
                          "⛓️  On-chain " if feature in ['transaction_count', 'active_addresses', 'hashrate_change_pct', 'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio'] else \
                          "😊 Sentiment"
            print(f"   {i+1:2d}. {feature_type} | {feature:<35} | {importance:.4f}")
        
        print(f"\n🔮 Prediction Summary:")
        predictions = results['predictions']
        print(f"   • Next 1h predicted return: {predictions['predicted_return'][:4].mean():.4f}")
        print(f"   • Next 4h predicted return: {predictions['predicted_return'][:16].mean():.4f}")
        print(f"   • Next 12h predicted return: {predictions['predicted_return'][:48].mean():.4f}")
        print(f"   • Next 24h cumulative return: {predictions['cumulative_return'].iloc[-1]:.4f}")
        print(f"   • Prediction volatility: {predictions['predicted_return'].std():.4f}")
        
        print(f"\n💡 Key Insights:")
        # Most important features by category
        top_price = max([(f, i) for f, i in train_results['feature_importance'].items() 
                        if f in ['price', 'volume', 'high_low_ratio', 'open_close_ratio', 'price_change_pct']], 
                       key=lambda x: x[1], default=('N/A', 0))
        top_technical = max([(f, i) for f, i in train_results['feature_importance'].items() 
                           if f in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'bb_width', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20']], 
                          key=lambda x: x[1], default=('N/A', 0))
        top_onchain = max([(f, i) for f, i in train_results['feature_importance'].items() 
                          if f in ['transaction_count', 'active_addresses', 'hashrate_change_pct', 'onchain_exchange_volume_ratio', 'mvrv_ratio', 'nvt_ratio']], 
                         key=lambda x: x[1], default=('N/A', 0))
        top_sentiment = max([(f, i) for f, i in train_results['feature_importance'].items() 
                           if f in ['social_sentiment_score', 'news_sentiment_indicator', 'fear_greed_index', 'volatility_sentiment', 'google_trends_proxy', 'twitter_mentions', 'reddit_engagement_score']], 
                          key=lambda x: x[1], default=('N/A', 0))
        
        print(f"   • Most important price feature: {top_price[0]} ({top_price[1]:.4f})")
        print(f"   • Most important technical feature: {top_technical[0]} ({top_technical[1]:.4f})")
        print(f"   • Most important on-chain feature: {top_onchain[0]} ({top_onchain[1]:.4f})")
        print(f"   • Most important sentiment feature: {top_sentiment[0]} ({top_sentiment[1]:.4f})")
        
        total_onchain_sentiment = (category_analysis['On-chain']['total_importance'] + 
                                 category_analysis['Sentiment']['total_importance'])
        total_traditional = (category_analysis['Price']['total_importance'] + 
                           category_analysis['Technical']['total_importance'])
        
        print(f"   • On-chain + Sentiment contribution: {total_onchain_sentiment:.3f} ({total_onchain_sentiment/(total_onchain_sentiment+total_traditional)*100:.1f}%)")
        print(f"   • Traditional features contribution: {total_traditional:.3f} ({total_traditional/(total_onchain_sentiment+total_traditional)*100:.1f}%)")
        
        print(f"\n✅ Enhanced Analysis Completed Successfully!")
        print(f"   • Model trained with {data_summary['total_features']} features across 4 categories")
        print(f"   • Added {category_analysis['On-chain']['feature_count'] + category_analysis['Sentiment']['feature_count']} new features (on-chain + sentiment)")
        print(f"   • Feature importance plot saved as 'feature_importance_demo.png'")
        print(f"   • Detailed logs saved in 'crypto_model_demo.log'")
        print("="*90)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()