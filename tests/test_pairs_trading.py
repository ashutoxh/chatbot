# test_pairs_trading.py
# Unit tests for pairs trading module

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pairs_trading import (
    _validate_date_range,
    _calculate_spread,
    run_pairs_analysis,
    InvalidDateRangeError,
    InsufficientDataError,
    CointegrationRequirementError,
    PairsTradingError
)


class TestDateValidation:
    """Test date range validation."""
    
    def test_valid_date_range(self):
        """Test that valid date ranges pass validation."""
        end = datetime.now()
        start = end - timedelta(days=365)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        result_start, result_end = _validate_date_range(start_str, end_str)
        assert result_start == start
        assert result_end == end
    
    def test_invalid_date_format(self):
        """Test that invalid date formats raise errors."""
        with pytest.raises(InvalidDateRangeError):
            _validate_date_range("2023-13-01", "2023-12-31")
        
        with pytest.raises(InvalidDateRangeError):
            _validate_date_range("invalid", "2023-12-31")
    
    def test_start_after_end(self):
        """Test that start date after end date raises error."""
        end = datetime.now() - timedelta(days=100)
        start = datetime.now() - timedelta(days=50)
        
        with pytest.raises(InvalidDateRangeError):
            _validate_date_range(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            )
    
    def test_exceeds_five_years(self):
        """Test that date ranges exceeding 5 years raise error."""
        end = datetime.now()
        start = end - timedelta(days=5*365 + 10)
        
        with pytest.raises(InvalidDateRangeError):
            _validate_date_range(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            )
    
    def test_start_too_old(self):
        """Test that start dates older than 5 years raise error."""
        end = datetime.now()
        start = datetime.now() - timedelta(days=6*365)
        
        with pytest.raises(InvalidDateRangeError):
            _validate_date_range(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            )
    
    def test_end_in_future(self):
        """Test that end dates in the future raise error."""
        start = datetime.now() - timedelta(days=100)
        end = datetime.now() + timedelta(days=10)
        
        with pytest.raises(InvalidDateRangeError):
            _validate_date_range(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            )


class TestSpreadCalculation:
    """Test spread calculation logic."""
    
    def test_spread_calculation(self):
        """Test that spread is calculated correctly using OLS."""
        # Create synthetic correlated data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        X = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)
        y = pd.Series(X * 1.2 + 5 + np.random.randn(100) * 0.3, index=dates)
        
        data = pd.DataFrame({'STOCK1': X, 'STOCK2': y})
        
        spread = _calculate_spread(data, 'STOCK1', 'STOCK2')
        
        assert len(spread) == len(data)
        assert isinstance(spread, pd.Series)
        # Spread should be roughly stationary (mean-reverting)
        assert abs(spread.mean()) < 10  # Reasonable threshold


class TestPairsTradingAnalysis:
    """Test full pairs trading analysis."""
    
    @patch('pairs_trading.yf.download')
    @patch('pairs_trading.coint')
    def test_successful_analysis(self, mock_coint, mock_download):
        """Test successful pairs trading analysis with valid data."""
        # Mock yfinance download - returns MultiIndex DataFrame with 'Close' column
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        close_data = pd.DataFrame({
            'XOM': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'CVX': 120 + np.cumsum(np.random.randn(100) * 0.5)
        }, index=dates)
        # Simulate yfinance MultiIndex structure
        mock_download.return_value = pd.DataFrame({
            ('Close', 'XOM'): close_data['XOM'],
            ('Close', 'CVX'): close_data['CVX']
        }, index=dates)
        
        # Mock cointegration test (p-value < 0.05)
        mock_coint.return_value = (-3.5, 0.03, None)
        
        result = run_pairs_analysis(
            stock1='XOM',
            stock2='CVX',
            start_date='2023-01-01',
            end_date='2023-04-10',
            run_cointegration_test=False
        )
        
        assert result.stock1 == 'XOM'
        assert result.stock2 == 'CVX'
        assert result.total_pnl is not None
        assert result.spread_plot_base64 is not None
        assert result.pnl_plot_base64 is not None
        assert result.summary_stats['trading_p_value'] < 0.05
    
    @patch('pairs_trading.yf.download')
    @patch('pairs_trading.coint')
    def test_cointegration_requirement_failure(self, mock_coint, mock_download):
        """Test that analysis fails when p-value >= 0.05."""
        # Mock yfinance download - returns MultiIndex DataFrame with 'Close' column
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        close_data = pd.DataFrame({
            'XOM': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'CVX': 120 + np.cumsum(np.random.randn(100) * 0.5)
        }, index=dates)
        # Simulate yfinance MultiIndex structure
        mock_download.return_value = pd.DataFrame({
            ('Close', 'XOM'): close_data['XOM'],
            ('Close', 'CVX'): close_data['CVX']
        }, index=dates)
        
        # Mock cointegration test (p-value >= 0.05 - should fail)
        mock_coint.return_value = (-1.5, 0.15, None)
        
        with pytest.raises(CointegrationRequirementError):
            run_pairs_analysis(
                stock1='XOM',
                stock2='CVX',
                start_date='2023-01-01',
                end_date='2023-04-10',
                run_cointegration_test=False
            )
    
    @patch('pairs_trading.yf.download')
    def test_insufficient_data(self, mock_download):
        """Test that insufficient data raises appropriate error."""
        # Mock empty data
        mock_download.return_value = pd.DataFrame()
        
        with pytest.raises(InsufficientDataError):
            run_pairs_analysis(
                stock1='INVALID1',
                stock2='INVALID2',
                start_date='2023-01-01',
                end_date='2023-12-31',
                run_cointegration_test=False
            )
    
    def test_same_stock_error(self):
        """Test that same stock ticker raises error."""
        with pytest.raises(PairsTradingError):
            run_pairs_analysis(
                stock1='XOM',
                stock2='XOM',
                start_date='2023-01-01',
                end_date='2023-12-31',
                run_cointegration_test=False
            )
    
    def test_empty_ticker_error(self):
        """Test that empty tickers raise error."""
        with pytest.raises(PairsTradingError):
            run_pairs_analysis(
                stock1='',
                stock2='CVX',
                start_date='2023-01-01',
                end_date='2023-12-31',
                run_cointegration_test=False
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

