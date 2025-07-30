#!/usr/bin/env python3
"""
OptimizeModel_fixed_id_matching.py

Enhanced stock prediction model with support for:
- Traditional technical indicators (backward compatibility)
- On-chain metrics (blockchain activity data)
- Sentiment index (social media and news sentiment)

Features:
- Advanced feature engineering
- Multiple data source integration
- Robust error handling and validation
- Comprehensive logging
- Scalable model training pipeline
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data sources
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Enhanced data loader supporting multiple data sources."""
    
    def __init__(self, symbol: str, period: str = "2y"):
        self.symbol = symbol
        self.period = period
        logger.info(f"Initializing DataLoader for {symbol} with period {period}")
    
    def load_stock_data(self) -> pd.DataFrame:
        """Load traditional stock price data using yfinance."""
        try:
            logger.info(f"Loading stock data for {self.symbol}")
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=self.period)
            
            if data.empty:
                logger.warning("Yahoo Finance returned empty data")
                return self._generate_simulated_stock_data()
            
            data.reset_index(inplace=True)
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            logger.info(f"Successfully loaded {len(data)} rows of stock data")
            return data
        except Exception as e:
            logger.warning(f"Error loading stock data from Yahoo Finance: {e}")
            logger.info("Falling back to simulated stock data for demonstration")
            return self._generate_simulated_stock_data()
    
    def _generate_simulated_stock_data(self) -> pd.DataFrame:
        """Generate simulated stock data for demonstration when real data isn't available."""
        try:
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=730),
                end=datetime.now(),
                freq='D'
            )
            
            # Remove weekends for stock data
            dates = [d for d in dates if d.weekday() < 5]
            
            # Generate realistic stock price simulation
            np.random.seed(42)
            n_days = len(dates)
            
            # Start with a base price and simulate daily returns
            base_price = 200.0  # Starting price for TSLA-like stock
            returns = np.random.normal(0.001, 0.03, n_days)  # Daily returns with slight upward bias
            
            # Create price series
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
            
            # Generate OHLC data
            stock_data = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
                'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
                'close': prices,
                'volume': np.random.randint(10000000, 50000000, n_days),
                'dividends': [0.0] * n_days,
                'stock_splits': [0.0] * n_days
            })
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            for i in range(len(stock_data)):
                stock_data.loc[i, 'high'] = max(stock_data.loc[i, 'open'], 
                                               stock_data.loc[i, 'close'], 
                                               stock_data.loc[i, 'high'])
                stock_data.loc[i, 'low'] = min(stock_data.loc[i, 'open'], 
                                              stock_data.loc[i, 'close'], 
                                              stock_data.loc[i, 'low'])
            
            logger.info(f"Generated {len(stock_data)} rows of simulated stock data")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error generating simulated stock data: {e}")
            return pd.DataFrame()
    
    def load_onchain_metrics(self) -> pd.DataFrame:
        """Load on-chain metrics for cryptocurrency analysis."""
        try:
            logger.info("Loading on-chain metrics")
            
            # Simulated on-chain data (in production, this would connect to blockchain APIs)
            # such as CoinMetrics, Glassnode, or other blockchain data providers
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=730),
                end=datetime.now(),
                freq='D'
            )
            
            np.random.seed(42)  # For reproducible results
            onchain_data = pd.DataFrame({
                'date': dates,
                'network_hash_rate': np.random.normal(150e18, 20e18, len(dates)),  # Hash rate in H/s
                'active_addresses': np.random.normal(900000, 100000, len(dates)),  # Daily active addresses
                'transaction_count': np.random.normal(300000, 50000, len(dates)),  # Daily transactions
                'transaction_volume': np.random.normal(50e9, 10e9, len(dates)),    # Daily volume in USD
                'mining_difficulty': np.random.normal(25e12, 5e12, len(dates)),    # Mining difficulty
                'market_cap': np.random.normal(800e9, 200e9, len(dates)),          # Market cap in USD
                'realized_cap': np.random.normal(400e9, 100e9, len(dates))         # Realized cap in USD
            })
            
            # Calculate derived metrics
            onchain_data['nvt_ratio'] = onchain_data['market_cap'] / onchain_data['transaction_volume']
            onchain_data['mvrv_ratio'] = onchain_data['market_cap'] / onchain_data['realized_cap']
            
            # Ensure positive values
            for col in ['network_hash_rate', 'active_addresses', 'transaction_count', 
                       'transaction_volume', 'mining_difficulty']:
                onchain_data[col] = np.abs(onchain_data[col])
            
            logger.info(f"Successfully loaded {len(onchain_data)} rows of on-chain data")
            return onchain_data
            
        except Exception as e:
            logger.error(f"Error loading on-chain metrics: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """Load sentiment analysis data from multiple sources."""
        try:
            logger.info("Loading sentiment data")
            
            # Simulated sentiment data (in production, this would connect to sentiment APIs)
            # such as LunarCrush, Santiment, or custom sentiment analysis services
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=730),
                end=datetime.now(),
                freq='D'
            )
            
            np.random.seed(123)  # For reproducible results
            sentiment_data = pd.DataFrame({
                'date': dates,
                'social_media_sentiment': np.random.normal(0, 1, len(dates)),      # Normalized sentiment -3 to 3
                'news_sentiment': np.random.normal(0, 1, len(dates)),             # News sentiment score
                'fear_greed_index': np.random.uniform(0, 100, len(dates)),        # Fear & Greed index 0-100
                'google_trends': np.random.uniform(0, 100, len(dates)),           # Google Trends popularity
                'reddit_activity': np.random.poisson(1000, len(dates)),          # Reddit mentions/activity
                'twitter_activity': np.random.poisson(5000, len(dates)),         # Twitter mentions/activity
                'reddit_sentiment': np.random.normal(0, 1, len(dates)),          # Reddit sentiment
                'twitter_sentiment': np.random.normal(0, 1, len(dates))          # Twitter sentiment
            })
            
            # Add some correlation to make data more realistic
            sentiment_data['composite_sentiment'] = (
                sentiment_data['social_media_sentiment'] * 0.3 +
                sentiment_data['news_sentiment'] * 0.3 +
                sentiment_data['reddit_sentiment'] * 0.2 +
                sentiment_data['twitter_sentiment'] * 0.2
            )
            
            logger.info(f"Successfully loaded {len(sentiment_data)} rows of sentiment data")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()


class FeatureEngineer:
    """Advanced feature engineering for all data types."""
    
    def __init__(self):
        logger.info("Initializing FeatureEngineer")
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create traditional technical indicators for backward compatibility."""
        try:
            logger.info("Creating technical indicators")
            df = data.copy()
            
            # Basic price features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            for window in [5, 10, 20]:  # Reduced window sizes
                if len(df) > window:
                    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                    df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                    df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            
            # Volatility indicators
            df['volatility_10'] = df['price_change'].rolling(window=10).std()
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            
            # RSI (Relative Strength Index) - simplified for shorter periods
            if len(df) > 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)  # Add small value to avoid division by zero
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD - simplified
            if len(df) > 26:
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            if len(df) > 20:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                bb_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            
            logger.info(f"Created {len([col for col in df.columns if col not in data.columns])} technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error creating technical indicators: {e}")
            return data
    
    def create_onchain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced on-chain features."""
        try:
            logger.info("Creating on-chain features")
            df = data.copy()
            
            # Network activity trends
            for window in [7, 14, 30]:
                df[f'hash_rate_ma_{window}'] = df['network_hash_rate'].rolling(window=window).mean()
                df[f'active_addresses_ma_{window}'] = df['active_addresses'].rolling(window=window).mean()
                df[f'tx_count_ma_{window}'] = df['transaction_count'].rolling(window=window).mean()
                df[f'tx_volume_ma_{window}'] = df['transaction_volume'].rolling(window=window).mean()
            
            # Network growth rates
            df['hash_rate_growth'] = df['network_hash_rate'].pct_change(periods=7)
            df['address_growth'] = df['active_addresses'].pct_change(periods=7)
            df['tx_growth'] = df['transaction_count'].pct_change(periods=7)
            df['volume_growth'] = df['transaction_volume'].pct_change(periods=7)
            
            # Advanced metrics
            df['avg_tx_value'] = df['transaction_volume'] / df['transaction_count']
            df['network_utilization'] = df['transaction_count'] / df['active_addresses']
            
            # NVT and MVRV derivatives
            df['nvt_ma_30'] = df['nvt_ratio'].rolling(window=30).mean()
            df['nvt_deviation'] = (df['nvt_ratio'] - df['nvt_ma_30']) / df['nvt_ma_30']
            df['mvrv_ma_30'] = df['mvrv_ratio'].rolling(window=30).mean()
            df['mvrv_deviation'] = (df['mvrv_ratio'] - df['mvrv_ma_30']) / df['mvrv_ma_30']
            
            # Difficulty adjustment indicators
            df['difficulty_change'] = df['mining_difficulty'].pct_change()
            df['difficulty_ma_14'] = df['mining_difficulty'].rolling(window=14).mean()
            
            logger.info(f"Created {len([col for col in df.columns if col not in data.columns])} on-chain features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating on-chain features: {e}")
            return data
    
    def create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features."""
        try:
            logger.info("Creating sentiment features")
            df = data.copy()
            
            # Sentiment smoothing and trends
            for window in [3, 7, 14]:
                df[f'sentiment_ma_{window}'] = df['composite_sentiment'].rolling(window=window).mean()
                df[f'news_sentiment_ma_{window}'] = df['news_sentiment'].rolling(window=window).mean()
                df[f'social_sentiment_ma_{window}'] = df['social_media_sentiment'].rolling(window=window).mean()
            
            # Sentiment momentum
            df['sentiment_momentum'] = df['composite_sentiment'].diff(periods=3)
            df['sentiment_volatility'] = df['composite_sentiment'].rolling(window=7).std()
            
            # Fear & Greed analysis
            df['fear_greed_normalized'] = (df['fear_greed_index'] - 50) / 50  # Normalize to -1, 1
            df['fear_greed_ma_7'] = df['fear_greed_index'].rolling(window=7).mean()
            df['fear_greed_extreme'] = np.where(df['fear_greed_index'] < 25, -1,
                                               np.where(df['fear_greed_index'] > 75, 1, 0))
            
            # Social activity metrics
            df['social_activity_total'] = df['reddit_activity'] + df['twitter_activity']
            df['social_activity_ratio'] = df['reddit_activity'] / (df['twitter_activity'] + 1)
            
            for window in [7, 14]:
                df[f'reddit_activity_ma_{window}'] = df['reddit_activity'].rolling(window=window).mean()
                df[f'twitter_activity_ma_{window}'] = df['twitter_activity'].rolling(window=window).mean()
                df[f'google_trends_ma_{window}'] = df['google_trends'].rolling(window=window).mean()
            
            # Sentiment divergence indicators
            df['sentiment_news_divergence'] = df['social_media_sentiment'] - df['news_sentiment']
            df['sentiment_reddit_twitter_divergence'] = df['reddit_sentiment'] - df['twitter_sentiment']
            
            # Advanced sentiment indicators
            df['sentiment_strength'] = np.abs(df['composite_sentiment'])
            df['sentiment_direction'] = np.sign(df['composite_sentiment'])
            
            logger.info(f"Created {len([col for col in df.columns if col not in data.columns])} sentiment features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating sentiment features: {e}")
            return data


class OptimizeModel:
    """Enhanced stock prediction model with multi-source data integration."""
    
    def __init__(self, symbol: str, prediction_days: int = 1):
        self.symbol = symbol
        self.prediction_days = prediction_days
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
        # Initialize components
        self.data_loader = DataLoader(symbol)
        self.feature_engineer = FeatureEngineer()
        
        logger.info(f"Initialized OptimizeModel for {symbol} with {prediction_days} day prediction")
    
    def load_all_data(self) -> pd.DataFrame:
        """Load and merge all data sources."""
        try:
            logger.info("Loading all data sources")
            
            # Load base stock data
            stock_data = self.data_loader.load_stock_data()
            if stock_data.empty:
                raise ValueError("No stock data available")
            
            # Load additional data sources
            onchain_data = self.data_loader.load_onchain_metrics()
            sentiment_data = self.data_loader.load_sentiment_data()
            
            # Merge data on date
            if not onchain_data.empty:
                stock_data = pd.merge(stock_data, onchain_data, on='date', how='left')
                logger.info("Merged on-chain data")
            
            if not sentiment_data.empty:
                stock_data = pd.merge(stock_data, sentiment_data, on='date', how='left')
                logger.info("Merged sentiment data")
            
            # Forward fill missing values for merged data
            stock_data.ffill(inplace=True)
            stock_data.bfill(inplace=True)
            
            logger.info(f"Combined dataset shape: {stock_data.shape}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error loading all data: {e}")
            return pd.DataFrame()
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all advanced features."""
        try:
            logger.info("Creating advanced features")
            
            # Apply all feature engineering
            df = self.feature_engineer.create_technical_indicators(data)
            
            # Only create on-chain features if the data is available
            if any(col in df.columns for col in ['network_hash_rate', 'active_addresses']):
                df = self.feature_engineer.create_onchain_features(df)
                logger.info("Added on-chain features")
            
            # Only create sentiment features if the data is available
            if any(col in df.columns for col in ['social_media_sentiment', 'news_sentiment']):
                df = self.feature_engineer.create_sentiment_features(df)
                logger.info("Added sentiment features")
            
            # Remove rows with NaN values (keep at least 100 rows for training)
            initial_len = len(df)
            df.dropna(inplace=True)
            
            if len(df) < 100:
                logger.warning(f"Only {len(df)} valid rows after feature engineering. Using forward fill to preserve more data.")
                # Reset to original data and use more aggressive filling
                df = data.copy()
                df = self.feature_engineer.create_technical_indicators(df)
                
                # Create target variable first
                df['target'] = df['close'].shift(-self.prediction_days)
                
                # Create lagged features
                for lag in [1, 2, 3]:  # Reduced lag periods
                    df[f'close_lag_{lag}'] = df['close'].shift(lag)
                    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                    if 'price_change' in df.columns:
                        df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
                
                # Fill NaN values more aggressively
                df.ffill(inplace=True)
                df.bfill(inplace=True)
                
                # Remove only the last few rows that can't have targets
                df = df[:-self.prediction_days] if self.prediction_days > 0 else df
            else:
                # Create target variable
                df['target'] = df['close'].shift(-self.prediction_days)
                
                # Create lagged features
                for lag in [1, 2, 3]:
                    df[f'close_lag_{lag}'] = df['close'].shift(lag)
                    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                    if 'price_change' in df.columns:
                        df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
                
                # Final cleanup
                df.dropna(inplace=True)
            
            logger.info(f"Final dataset shape after feature engineering: {df.shape} (started with {initial_len})")
            return df
            
        except Exception as e:
            logger.error(f"Error creating advanced features: {e}")
            return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for model training."""
        try:
            # Select feature columns (exclude date, target, and original price columns)
            exclude_cols = ['date', 'target', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            X = data[feature_cols].values
            y = data['target'].values
            
            # Handle infinite and NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            y = np.nan_to_num(y, nan=0.0)
            
            logger.info(f"Prepared features: {len(feature_cols)} features, {len(X)} samples")
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]), np.array([]), []
    
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train multiple models with the prepared data."""
        try:
            logger.info("Training models")
            
            # Split data for training (keeping time series order)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['standard'] = scaler
            
            # Initialize models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            # Train and evaluate models
            for name, model in models.items():
                logger.info(f"Training {name}")
                
                if name == 'linear_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
                # Store model and metrics
                self.models[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[name] = importance
                    logger.info(f"Top 5 features for {name}:")
                    for idx, row in importance.head().iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def predict(self, data: pd.DataFrame, model_name: str = 'random_forest') -> Dict:
        """Make predictions using the trained model."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            logger.info(f"Making predictions with {model_name}")
            
            # Prepare features
            X, _, feature_names = self.prepare_features(data)
            
            # Scale if needed
            if model_name == 'linear_regression':
                X = self.scalers['standard'].transform(X)
            
            # Make predictions
            model = self.models[model_name]['model']
            predictions = model.predict(X)
            
            # Get the last few actual prices for context
            actual_prices = data['close'].tail(10).tolist()
            
            result = {
                'symbol': self.symbol,
                'prediction_days': self.prediction_days,
                'model_used': model_name,
                'predictions': predictions.tolist(),
                'recent_prices': actual_prices,
                'model_metrics': {
                    'mse': self.models[model_name]['mse'],
                    'mae': self.models[model_name]['mae'],
                    'r2': self.models[model_name]['r2']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {len(predictions)} predictions")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {}
    
    def train_and_predict(self, enable_onchain: bool = True, enable_sentiment: bool = True) -> Dict:
        """Complete pipeline: load data, train models, and make predictions."""
        try:
            logger.info("Starting complete training and prediction pipeline")
            
            # Load all data
            data = self.load_all_data()
            if data.empty:
                raise ValueError("No data available for training")
            
            # Optionally disable certain data sources
            if not enable_onchain:
                onchain_cols = [col for col in data.columns if any(x in col.lower() for x in 
                               ['hash_rate', 'active_addresses', 'transaction', 'mining_difficulty', 'nvt', 'mvrv'])]
                data.drop(columns=onchain_cols, inplace=True)
                logger.info("On-chain features disabled")
            
            if not enable_sentiment:
                sentiment_cols = [col for col in data.columns if any(x in col.lower() for x in 
                                 ['sentiment', 'fear_greed', 'social', 'reddit', 'twitter', 'google_trends'])]
                data.drop(columns=sentiment_cols, inplace=True)
                logger.info("Sentiment features disabled")
            
            # Create advanced features
            data = self.create_advanced_features(data)
            
            # Prepare features for training
            X, y, feature_names = self.prepare_features(data)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid features prepared")
            
            # Train models
            self.train_models(X, y, feature_names)
            
            # Make predictions on the latest data
            predictions = self.predict(data)
            
            logger.info("Pipeline completed successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in training and prediction pipeline: {e}")
            return {}
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """Get feature importance for interpretability."""
        if model_name in self.feature_importance:
            return self.feature_importance[model_name]
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save the trained model and scalers."""
        try:
            import pickle
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'symbol': self.symbol,
                'prediction_days': self.prediction_days,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.symbol = model_data['symbol']
            self.prediction_days = model_data['prediction_days']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")


def main():
    """Example usage of the OptimizeModel."""
    try:
        # Initialize model for Tesla stock
        model = OptimizeModel('TSLA', prediction_days=1)
        
        # Train and predict with all features enabled
        logger.info("Training with all features enabled")
        results_all = model.train_and_predict(enable_onchain=True, enable_sentiment=True)
        
        if results_all:
            print("\n=== Prediction Results (All Features) ===")
            print(f"Symbol: {results_all['symbol']}")
            print(f"Model: {results_all['model_used']}")
            print(f"Latest prediction: ${results_all['predictions'][-1]:.2f}")
            print(f"Recent prices: {[f'${p:.2f}' for p in results_all['recent_prices'][-3:]]}")
            print(f"Model R2 Score: {results_all['model_metrics']['r2']:.4f}")
        
        # Show feature importance
        importance = model.get_feature_importance()
        if not importance.empty:
            print("\n=== Top 10 Most Important Features ===")
            for idx, row in importance.head(10).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Train with only traditional features (backward compatibility)
        logger.info("Training with only traditional features")
        results_traditional = model.train_and_predict(enable_onchain=False, enable_sentiment=False)
        
        if results_traditional:
            print(f"\n=== Traditional Model Prediction ===")
            print(f"Latest prediction: ${results_traditional['predictions'][-1]:.2f}")
            print(f"Model R2 Score: {results_traditional['model_metrics']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()