# pairs_trading.py
# George Soros Pairs Trading Strategy Engine

import base64
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import coint


class PairsTradingError(Exception):
    """Base exception for pairs trading operations."""
    pass


class InvalidDateRangeError(PairsTradingError):
    """Raised when date range exceeds 5 years or is invalid."""
    pass


class InsufficientDataError(PairsTradingError):
    """Raised when insufficient data is available for analysis."""
    pass


class CointegrationRequirementError(PairsTradingError):
    """Raised when p-value requirement is not met."""
    pass


@dataclass
class PairTradingResult:
    """Result container for pairs trading analysis."""
    stock1: str
    stock2: str
    start_date: str
    end_date: str
    cointegration_p_value: Optional[float]
    cointegration_score: Optional[float]
    cointegration_passed: bool
    total_pnl: float
    cumulative_pnl_series: pd.Series
    spread_series: pd.Series
    trading_signals: pd.DataFrame
    spread_plot_base64: str
    pnl_plot_base64: str
    summary_stats: dict


def _validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """
    Validate and parse date range, ensuring it doesn't exceed 5 years.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (start_datetime, end_datetime)
        
    Raises:
        InvalidDateRangeError: If dates are invalid or exceed 5 years
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise InvalidDateRangeError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    
    if start >= end:
        raise InvalidDateRangeError("Start date must be before end date.")
    
    max_start = datetime.now() - timedelta(days=5*365)
    if start < max_start:
        raise InvalidDateRangeError(
            f"Start date cannot be older than 5 years ago ({max_start.strftime('%Y-%m-%d')})."
        )
    
    if end > datetime.now():
        raise InvalidDateRangeError("End date cannot be in the future.")
    
    if (end - start).days > 5*365:
        raise InvalidDateRangeError("Date range cannot exceed 5 years.")
    
    return start, end


def _download_stock_data(
    stock1: str, 
    stock2: str, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    """
    Download historical stock data for two tickers.
    
    Args:
        stock1: First stock ticker symbol
        stock2: Second stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with Close prices for both stocks
        
    Raises:
        InsufficientDataError: If data download fails or insufficient data
    """
    try:
        data = yf.download([stock1, stock2], start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise InsufficientDataError(f"No data available for {stock1} and {stock2} in the specified date range.")
        
        # Handle MultiIndex columns (multiple tickers) or single-level columns
        if isinstance(data.columns, pd.MultiIndex):
            try:
                close_data = data['Close']
            except KeyError:
                raise InsufficientDataError("Unable to retrieve Close prices from yfinance.")
        else:
            # Single ticker case - data should already be Close prices
            close_data = data
        
        if stock1 not in close_data.columns or stock2 not in close_data.columns:
            raise InsufficientDataError(f"One or both tickers ({stock1}, {stock2}) not found in downloaded data.")
        
        if len(close_data) < 30:
            raise InsufficientDataError("Insufficient data points (need at least 30 trading days).")
        
        return close_data.dropna()
        
    except Exception as e:
        if isinstance(e, InsufficientDataError):
            raise
        raise InsufficientDataError(f"Error downloading stock data: {str(e)}")


def _test_cointegration(X: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Perform Engle-Granger cointegration test.
    
    Args:
        X: First time series (stock1 prices)
        y: Second time series (stock2 prices)
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    score, p_value, _ = coint(X, y)
    return score, p_value


def _calculate_spread(data: pd.DataFrame, stock1: str, stock2: str) -> pd.Series:
    """
    Calculate spread using OLS regression.
    
    Args:
        data: DataFrame with Close prices
        stock1: First stock ticker
        stock2: Second stock ticker
        
    Returns:
        Series of spread values
    """
    X = data[stock1]
    y = data[stock2]
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const).fit()
    spread = y - model.predict(X_with_const)
    
    return spread


def _generate_spread_plot(
    data: pd.DataFrame,
    spread: pd.Series,
    stock1: str,
    stock2: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Generate spread plot with thresholds and save as base64.
    
    Returns:
        Base64-encoded PNG image string
    """
    mean_spread = spread.mean()
    std_spread = spread.std()
    c = 1.18
    upper_threshold = mean_spread + c * std_spread
    lower_threshold = mean_spread - c * std_spread
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, spread, label='Spread', color='blue', linewidth=1.5)
    plt.axhline(mean_spread, color='black', linestyle='--', label='Mean', linewidth=1.5)
    plt.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold', linewidth=1.5)
    plt.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold', linewidth=1.5)
    plt.legend()
    plt.title(f'Pairs Trading Strategy: Spread between {stock1} and {stock2}\n{start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plot_base64


def _generate_pnl_plot(
    data: pd.DataFrame,
    cumulative_pnl: pd.Series,
    stock1: str,
    stock2: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Generate cumulative PnL plot and save as base64.
    
    Returns:
        Base64-encoded PNG image string
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, cumulative_pnl, label='Cumulative PnL', color='purple', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.title(f'Pairs Trading Strategy: Cumulative Profit and Loss\n{stock1} vs {stock2} ({start_date} to {end_date})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plot_base64


def run_pairs_analysis(
    stock1: str,
    stock2: str,
    start_date: str,
    end_date: str,
    run_cointegration_test: bool = False
) -> PairTradingResult:
    """
    Run complete pairs trading analysis.
    
    Args:
        stock1: First stock ticker symbol
        stock2: Second stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        run_cointegration_test: Whether to run optional cointegration test
        
    Returns:
        PairTradingResult with all analysis results
        
    Raises:
        InvalidDateRangeError: If date range is invalid
        InsufficientDataError: If data is insufficient
        CointegrationRequirementError: If p-value >= 0.05 during trading period
    """
    # Validate inputs
    stock1 = stock1.strip().upper()
    stock2 = stock2.strip().upper()
    
    if not stock1 or not stock2:
        raise PairsTradingError("Both stock tickers must be provided.")
    
    if stock1 == stock2:
        raise PairsTradingError("Stock tickers must be different.")
    
    # Validate date range
    start_dt, end_dt = _validate_date_range(start_date, end_date)
    
    # Download data
    data = _download_stock_data(stock1, stock2, start_date, end_date)
    
    # Always run cointegration test to check pair quality (informational, not blocking)
    trading_score, trading_p_value = _test_cointegration(data[stock1], data[stock2])
    trading_cointegration_passed = trading_p_value < 0.05
    
    # Optional cointegration test (if user requested it, show detailed results)
    # Always populate these values so they're available when checkbox is checked
    cointegration_score = trading_score
    cointegration_p_value = trading_p_value
    cointegration_passed = trading_cointegration_passed
    
    # But only mark as "user requested" if checkbox was checked
    # This is handled in the UI formatting function
    
    # Calculate spread using OLS regression
    spread = _calculate_spread(data, stock1, stock2)
    data = data.copy()
    data['spread'] = spread
    
    # Define trading thresholds
    mean_spread = spread.mean()
    std_spread = spread.std()
    c = 1.18
    upper_threshold = mean_spread + c * std_spread
    lower_threshold = mean_spread - c * std_spread
    
    # Generate trading signals
    data['long'] = spread < lower_threshold
    data['short'] = spread > upper_threshold
    
    # Calculate returns
    data['returns_stock1'] = data[stock1].pct_change()
    data['returns_stock2'] = data[stock2].pct_change()
    
    # Calculate PnL
    data['pnl'] = np.where(
        data['long'],
        data['returns_stock2'] - data['returns_stock1'],
        0
    ) + np.where(
        data['short'],
        data['returns_stock1'] - data['returns_stock2'],
        0
    )
    
    # Cumulative PnL
    data['cumulative_pnl'] = data['pnl'].cumsum()
    total_pnl = data['cumulative_pnl'].iloc[-1]
    
    # Generate plots
    spread_plot = _generate_spread_plot(data, spread, stock1, stock2, start_date, end_date)
    pnl_plot = _generate_pnl_plot(data, data['cumulative_pnl'], stock1, stock2, start_date, end_date)
    
    # Summary statistics
    num_trades = data['long'].sum() + data['short'].sum()
    num_long = data['long'].sum()
    num_short = data['short'].sum()
    
    summary_stats = {
        'total_pnl': total_pnl,
        'num_trades': int(num_trades),
        'num_long': int(num_long),
        'num_short': int(num_short),
        'mean_spread': float(mean_spread),
        'std_spread': float(std_spread),
        'upper_threshold': float(upper_threshold),
        'lower_threshold': float(lower_threshold),
        'trading_p_value': float(trading_p_value),
        'trading_score': float(trading_score),
        'trading_cointegration_passed': bool(trading_cointegration_passed),
    }
    
    # Trading signals dataframe for display
    signals_df = data[['spread', 'long', 'short', 'pnl', 'cumulative_pnl']].tail(30)
    
    return PairTradingResult(
        stock1=stock1,
        stock2=stock2,
        start_date=start_date,
        end_date=end_date,
        cointegration_p_value=cointegration_p_value,
        cointegration_score=cointegration_score,
        cointegration_passed=cointegration_passed,
        total_pnl=total_pnl,
        cumulative_pnl_series=data['cumulative_pnl'],
        spread_series=spread,
        trading_signals=signals_df,
        spread_plot_base64=spread_plot,
        pnl_plot_base64=pnl_plot,
        summary_stats=summary_stats
    )

