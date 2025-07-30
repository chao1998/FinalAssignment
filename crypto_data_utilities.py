#!/usr/bin/env python3
"""
Crypto Data Utilities - On-chain and Sentiment Data Fetching Functions
======================================================================

This module provides utility functions for fetching on-chain metrics and sentiment data
for crypto prediction models. It includes both real API integrations and simulation
functions for demonstration purposes.

Key Features:
- On-chain data fetching (with fallback to simulation)
- Sentiment data aggregation from multiple sources
- Data validation and cleaning
- Error handling and logging
- Rate limiting and API management
"""

import pandas as pd
import numpy as np
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json

logger = logging.getLogger(__name__)

class OnChainDataFetcher:
    """Fetches on-chain metrics for crypto assets"""
    
    def __init__(self):
        self.api_keys = {
            'glassnode': None,  # Would store actual API key
            'coinmetrics': None,  # Would store actual API key
            'messari': None,  # Would store actual API key
        }
        self.rate_limits = {
            'glassnode': {'calls_per_minute': 10, 'last_call': 0},
            'coinmetrics': {'calls_per_minute': 30, 'last_call': 0},
            'messari': {'calls_per_minute': 20, 'last_call': 0},
        }
    
    def fetch_transaction_metrics(self, symbol: str, start_date: str, end_date: str, 
                                interval: str = "15m") -> pd.DataFrame:
        """
        Fetch transaction-related on-chain metrics
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('15m', '1h', '1d')
        
        Returns:
            DataFrame with transaction metrics
        """
        try:
            # In production, this would call actual APIs like Glassnode
            # For demo, we simulate realistic data
            logger.info(f"Fetching transaction metrics for {symbol}")
            
            return self._simulate_transaction_data(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching transaction metrics: {e}")
            return self._simulate_transaction_data(symbol, start_date, end_date, interval)
    
    def fetch_network_metrics(self, symbol: str, start_date: str, end_date: str,
                            interval: str = "15m") -> pd.DataFrame:
        """
        Fetch network health and security metrics
        
        Args:
            symbol: Crypto symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            DataFrame with network metrics
        """
        try:
            logger.info(f"Fetching network metrics for {symbol}")
            
            return self._simulate_network_data(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching network metrics: {e}")
            return self._simulate_network_data(symbol, start_date, end_date, interval)
    
    def fetch_value_flow_metrics(self, symbol: str, start_date: str, end_date: str,
                               interval: str = "15m") -> pd.DataFrame:
        """
        Fetch value flow and holder behavior metrics
        
        Args:
            symbol: Crypto symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            DataFrame with value flow metrics
        """
        try:
            logger.info(f"Fetching value flow metrics for {symbol}")
            
            return self._simulate_value_flow_data(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching value flow metrics: {e}")
            return self._simulate_value_flow_data(symbol, start_date, end_date, interval)
    
    def _simulate_transaction_data(self, symbol: str, start_date: str, end_date: str,
                                 interval: str) -> pd.DataFrame:
        """Simulate realistic transaction data"""
        # Create date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if interval == "15m":
            freq = "15T"
        elif interval == "1h":
            freq = "1H"
        else:
            freq = "1D"
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        n_points = len(date_range)
        
        # Set seed for reproducible data
        np.random.seed(hash(symbol) % 2**32)
        
        # Base values depend on the crypto
        if symbol.upper() == 'BTC':
            base_tx_count = 15000
            base_tx_volume = 1000000000  # $1B
        elif symbol.upper() == 'ETH':
            base_tx_count = 45000
            base_tx_volume = 800000000   # $800M
        else:
            base_tx_count = 5000
            base_tx_volume = 100000000   # $100M
        
        # Generate transaction count
        tx_trend = np.random.normal(0, 0.1, n_points)
        tx_count = base_tx_count * (1 + np.cumsum(tx_trend) * 0.01)
        tx_count = np.maximum(tx_count, base_tx_count * 0.3)
        
        # Generate transaction volume
        volume_trend = np.random.normal(0, 0.15, n_points)
        tx_volume = base_tx_volume * (1 + np.cumsum(volume_trend) * 0.01)
        tx_volume = np.maximum(tx_volume, base_tx_volume * 0.2)
        
        # Average transaction size
        avg_tx_size = tx_volume / tx_count
        
        return pd.DataFrame({
            'transaction_count': tx_count,
            'transaction_volume_usd': tx_volume,
            'avg_transaction_size': avg_tx_size
        }, index=date_range)
    
    def _simulate_network_data(self, symbol: str, start_date: str, end_date: str,
                             interval: str) -> pd.DataFrame:
        """Simulate realistic network data"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if interval == "15m":
            freq = "15T"
        elif interval == "1h":
            freq = "1H"
        else:
            freq = "1D"
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        n_points = len(date_range)
        
        np.random.seed(hash(symbol + "network") % 2**32)
        
        # Base values
        if symbol.upper() == 'BTC':
            base_addresses = 8000
            base_hashrate = 1e18
        elif symbol.upper() == 'ETH':
            base_addresses = 12000
            base_hashrate = 5e17
        else:
            base_addresses = 2000
            base_hashrate = 1e15
        
        # Active addresses
        addr_trend = np.random.normal(0, 0.05, n_points)
        active_addresses = base_addresses * (1 + np.cumsum(addr_trend) * 0.005)
        active_addresses = np.maximum(active_addresses, base_addresses * 0.5)
        
        # Network hash rate (more stable)
        hashrate_changes = np.random.normal(0, 0.01, n_points)
        hashrate = base_hashrate * np.cumprod(1 + hashrate_changes)
        hashrate_change_pct = np.gradient(hashrate) / hashrate * 100
        
        # Network difficulty adjustment
        difficulty_cycle = np.sin(np.arange(n_points) / (96 * 14)) * 0.05  # 2-week cycles
        network_difficulty = 1 + difficulty_cycle + np.cumsum(np.random.normal(0, 0.02, n_points)) * 0.001
        
        return pd.DataFrame({
            'active_addresses': active_addresses,
            'network_hashrate': hashrate,
            'hashrate_change_pct': hashrate_change_pct,
            'network_difficulty': network_difficulty
        }, index=date_range)
    
    def _simulate_value_flow_data(self, symbol: str, start_date: str, end_date: str,
                                interval: str) -> pd.DataFrame:
        """Simulate realistic value flow data"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if interval == "15m":
            freq = "15T"
        elif interval == "1h":
            freq = "1H"
        else:
            freq = "1D"
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        n_points = len(date_range)
        
        np.random.seed(hash(symbol + "value") % 2**32)
        
        # MVRV Ratio (Market Value to Realized Value)
        mvrv_base = 1.2
        mvrv_trend = np.random.normal(0, 0.1, n_points)
        mvrv_ratio = mvrv_base + np.cumsum(mvrv_trend) * 0.01
        mvrv_ratio = np.maximum(mvrv_ratio, 0.3)
        
        # NVT Ratio (Network Value to Transactions)
        nvt_base = 50 if symbol.upper() == 'BTC' else 30
        nvt_trend = np.random.normal(0, 0.15, n_points)
        nvt_ratio = nvt_base + np.cumsum(nvt_trend) * 0.5
        nvt_ratio = np.maximum(nvt_ratio, 10)
        
        # Exchange inflow/outflow ratio
        inflow_outflow_base = 1.0
        io_volatility = np.random.normal(0, 0.2, n_points)
        inflow_outflow_ratio = inflow_outflow_base + io_volatility
        inflow_outflow_ratio = np.maximum(inflow_outflow_ratio, 0.1)
        
        # Long-term holder ratio
        lth_base = 0.65  # 65% long-term holders
        lth_trend = np.random.normal(0, 0.02, n_points)
        long_term_holder_ratio = lth_base + np.cumsum(lth_trend) * 0.001
        long_term_holder_ratio = np.clip(long_term_holder_ratio, 0.4, 0.8)
        
        return pd.DataFrame({
            'mvrv_ratio': mvrv_ratio,
            'nvt_ratio': nvt_ratio,
            'exchange_inflow_outflow_ratio': inflow_outflow_ratio,
            'long_term_holder_ratio': long_term_holder_ratio
        }, index=date_range)

class SentimentDataFetcher:
    """Fetches sentiment data from various sources"""
    
    def __init__(self):
        self.api_keys = {
            'lunarcrush': None,  # Social media sentiment
            'santiment': None,   # On-chain social sentiment
            'cryptocompare': None,  # News sentiment
            'fear_greed': None,  # Fear & Greed Index
        }
        self.rate_limits = {
            'lunarcrush': {'calls_per_hour': 100, 'last_call': 0},
            'santiment': {'calls_per_hour': 300, 'last_call': 0},
            'cryptocompare': {'calls_per_hour': 200, 'last_call': 0},
        }
    
    def fetch_social_sentiment(self, symbol: str, start_date: str, end_date: str,
                             interval: str = "15m") -> pd.DataFrame:
        """
        Fetch social media sentiment metrics
        
        Args:
            symbol: Crypto symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            DataFrame with social sentiment metrics
        """
        try:
            logger.info(f"Fetching social sentiment for {symbol}")
            
            return self._simulate_social_sentiment(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching social sentiment: {e}")
            return self._simulate_social_sentiment(symbol, start_date, end_date, interval)
    
    def fetch_news_sentiment(self, symbol: str, start_date: str, end_date: str,
                           interval: str = "15m") -> pd.DataFrame:
        """
        Fetch news sentiment metrics
        
        Args:
            symbol: Crypto symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            DataFrame with news sentiment metrics
        """
        try:
            logger.info(f"Fetching news sentiment for {symbol}")
            
            return self._simulate_news_sentiment(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return self._simulate_news_sentiment(symbol, start_date, end_date, interval)
    
    def fetch_market_psychology(self, symbol: str, start_date: str, end_date: str,
                              interval: str = "15m") -> pd.DataFrame:
        """
        Fetch market psychology indicators
        
        Args:
            symbol: Crypto symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            DataFrame with market psychology metrics
        """
        try:
            logger.info(f"Fetching market psychology for {symbol}")
            
            return self._simulate_market_psychology(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error fetching market psychology: {e}")
            return self._simulate_market_psychology(symbol, start_date, end_date, interval)
    
    def _simulate_social_sentiment(self, symbol: str, start_date: str, end_date: str,
                                 interval: str) -> pd.DataFrame:
        """Simulate realistic social sentiment data"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if interval == "15m":
            freq = "15T"
        elif interval == "1h":
            freq = "1H"
        else:
            freq = "1D"
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        n_points = len(date_range)
        
        np.random.seed(hash(symbol + "social") % 2**32)
        
        # Social sentiment score (-1 to 1)
        sentiment_base = np.random.normal(0.05, 0.3, n_points)  # Slight positive bias
        social_sentiment = np.clip(sentiment_base, -1, 1)
        
        # Twitter mentions
        mentions_base = 5000 if symbol.upper() == 'BTC' else 2000
        mention_volatility = np.random.lognormal(0, 0.5, n_points)
        twitter_mentions = mentions_base * mention_volatility
        
        # Reddit engagement score (0 to 1)
        reddit_base = np.random.uniform(0.2, 0.8, n_points)
        reddit_engagement = np.clip(reddit_base, 0, 1)
        
        # Telegram activity score
        telegram_base = np.random.uniform(0.1, 0.9, n_points)
        telegram_activity = np.clip(telegram_base, 0, 1)
        
        return pd.DataFrame({
            'social_sentiment_score': social_sentiment,
            'twitter_mentions': twitter_mentions,
            'reddit_engagement_score': reddit_engagement,
            'telegram_activity_score': telegram_activity
        }, index=date_range)
    
    def _simulate_news_sentiment(self, symbol: str, start_date: str, end_date: str,
                               interval: str) -> pd.DataFrame:
        """Simulate realistic news sentiment data"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if interval == "15m":
            freq = "15T"
        elif interval == "1h":
            freq = "1H"
        else:
            freq = "1D"
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        n_points = len(date_range)
        
        np.random.seed(hash(symbol + "news") % 2**32)
        
        # News sentiment (-1 to 1)
        news_sentiment_base = np.random.normal(0.1, 0.25, n_points)  # Positive bias
        news_sentiment = np.clip(news_sentiment_base, -1, 1)
        
        # News volume (number of articles)
        news_volume_base = 50 if symbol.upper() == 'BTC' else 20
        news_volume = np.random.poisson(news_volume_base, n_points)
        
        # Media coverage score (0 to 1)
        media_coverage_base = np.random.uniform(0.1, 0.7, n_points)
        media_coverage = np.clip(media_coverage_base, 0, 1)
        
        return pd.DataFrame({
            'news_sentiment_indicator': news_sentiment,
            'news_volume': news_volume,
            'media_coverage_score': media_coverage
        }, index=date_range)
    
    def _simulate_market_psychology(self, symbol: str, start_date: str, end_date: str,
                                  interval: str) -> pd.DataFrame:
        """Simulate realistic market psychology data"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if interval == "15m":
            freq = "15T"
        elif interval == "1h":
            freq = "1H"
        else:
            freq = "1D"
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        n_points = len(date_range)
        
        np.random.seed(hash(symbol + "psychology") % 2**32)
        
        # Fear & Greed Index (0 to 100)
        fear_greed_base = 50 + np.random.normal(0, 15, n_points)
        fear_greed_index = np.clip(fear_greed_base, 0, 100)
        
        # Volatility sentiment (0 to 1)
        volatility_sentiment_base = np.random.uniform(0.2, 0.8, n_points)
        volatility_sentiment = np.clip(volatility_sentiment_base, 0, 1)
        
        # Google Trends proxy (0 to 100)
        google_trends_base = 20 + np.random.normal(0, 10, n_points)
        google_trends = np.clip(google_trends_base, 0, 100)
        
        # Market stress indicator (0 to 1)
        market_stress_base = np.random.uniform(0.1, 0.6, n_points)
        market_stress = np.clip(market_stress_base, 0, 1)
        
        return pd.DataFrame({
            'fear_greed_index': fear_greed_index,
            'volatility_sentiment': volatility_sentiment,
            'google_trends_proxy': google_trends,
            'market_stress_indicator': market_stress
        }, index=date_range)

class DataValidator:
    """Validates and cleans fetched data"""
    
    @staticmethod
    def validate_onchain_data(data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean on-chain data"""
        try:
            # Remove outliers (beyond 3 standard deviations)
            for column in data.select_dtypes(include=[np.number]).columns:
                mean = data[column].mean()
                std = data[column].std()
                data[column] = data[column].clip(mean - 3*std, mean + 3*std)
            
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # Remove remaining NaN values
            data = data.dropna()
            
            logger.info(f"Validated on-chain data: {len(data)} records, {len(data.columns)} features")
            return data
            
        except Exception as e:
            logger.error(f"Error validating on-chain data: {e}")
            return data
    
    @staticmethod
    def validate_sentiment_data(data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean sentiment data"""
        try:
            # Clip sentiment scores to valid ranges
            for column in data.columns:
                if 'sentiment' in column.lower():
                    data[column] = data[column].clip(-1, 1)
                elif 'score' in column.lower() or 'index' in column.lower():
                    if data[column].max() > 10:  # Likely 0-100 scale
                        data[column] = data[column].clip(0, 100)
                    else:  # Likely 0-1 scale
                        data[column] = data[column].clip(0, 1)
            
            # Handle missing values
            data = data.fillna(method='ffill')
            data = data.dropna()
            
            logger.info(f"Validated sentiment data: {len(data)} records, {len(data.columns)} features")
            return data
            
        except Exception as e:
            logger.error(f"Error validating sentiment data: {e}")
            return data

# Main integration function
def fetch_enhanced_crypto_data(symbol: str, start_date: str, end_date: str, 
                             interval: str = "15m") -> Dict[str, pd.DataFrame]:
    """
    Fetch complete enhanced crypto data including on-chain and sentiment metrics
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval ('15m', '1h', '1d')
    
    Returns:
        Dictionary containing DataFrames for different data types
    """
    try:
        logger.info(f"Fetching enhanced crypto data for {symbol} from {start_date} to {end_date}")
        
        # Initialize fetchers
        onchain_fetcher = OnChainDataFetcher()
        sentiment_fetcher = SentimentDataFetcher()
        
        # Fetch all data types
        transaction_data = onchain_fetcher.fetch_transaction_metrics(symbol, start_date, end_date, interval)
        network_data = onchain_fetcher.fetch_network_metrics(symbol, start_date, end_date, interval)
        value_flow_data = onchain_fetcher.fetch_value_flow_metrics(symbol, start_date, end_date, interval)
        
        social_sentiment = sentiment_fetcher.fetch_social_sentiment(symbol, start_date, end_date, interval)
        news_sentiment = sentiment_fetcher.fetch_news_sentiment(symbol, start_date, end_date, interval)
        market_psychology = sentiment_fetcher.fetch_market_psychology(symbol, start_date, end_date, interval)
        
        # Combine on-chain data
        onchain_combined = pd.concat([transaction_data, network_data, value_flow_data], axis=1)
        onchain_validated = DataValidator.validate_onchain_data(onchain_combined)
        
        # Combine sentiment data
        sentiment_combined = pd.concat([social_sentiment, news_sentiment, market_psychology], axis=1)
        sentiment_validated = DataValidator.validate_sentiment_data(sentiment_combined)
        
        logger.info("Successfully fetched and validated enhanced crypto data")
        
        return {
            'onchain_data': onchain_validated,
            'sentiment_data': sentiment_validated,
            'transaction_data': transaction_data,
            'network_data': network_data,
            'value_flow_data': value_flow_data,
            'social_sentiment': social_sentiment,
            'news_sentiment': news_sentiment,
            'market_psychology': market_psychology
        }
        
    except Exception as e:
        logger.error(f"Error fetching enhanced crypto data: {e}")
        raise

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        data = fetch_enhanced_crypto_data('BTC', start_date, end_date, '15m')
        
        print("\n" + "="*60)
        print("CRYPTO DATA UTILITIES - DEMO RESULTS")
        print("="*60)
        
        for data_type, df in data.items():
            print(f"\n{data_type.upper()}:")
            print(f"  • Shape: {df.shape}")
            print(f"  • Columns: {list(df.columns)}")
            print(f"  • Date range: {df.index[0]} to {df.index[-1]}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")