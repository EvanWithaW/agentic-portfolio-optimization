import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import datetime as dt

def download_data(tickers, start_date, end_date):
    """
    Download historical stock price data for the given tickers.

    Args:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pandas.DataFrame: DataFrame containing the adjusted close prices for each ticker
    """
    # Set auto_adjust=False to get the 'Adj Close' column
    # Note: yfinance changed the default of auto_adjust to True, which removes 'Adj Close'
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    # Use only adjusted close prices
    data = data['Adj Close']
    return data

def optimize_portfolio(prices, optimization_method='efficient_risk', risk_free_rate=0.02, target_volatility=0.15):
    """
    Optimize a portfolio using the specified method.

    Args:
        prices (pandas.DataFrame): DataFrame containing historical prices
        optimization_method (str): Method to use for optimization ('efficient_risk', 'efficient_return', or 'max_sharpe')
        risk_free_rate (float): Risk-free rate used for Sharpe ratio calculation
        target_volatility (float): Target volatility for 'efficient_risk' method

    Returns:
        tuple: (weights, performance) where weights is a dict of optimal weights and performance is a dict of metrics
    """
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # Optimize for the given method
    ef = EfficientFrontier(mu, S)

    if optimization_method == 'efficient_risk':
        # First, calculate the minimum volatility portfolio to get the minimum possible volatility
        min_vol_ef = EfficientFrontier(mu, S)
        min_vol_ef.min_volatility()
        _, min_volatility, _ = min_vol_ef.portfolio_performance()

        # Ensure target_volatility is at least the minimum volatility
        adjusted_target_volatility = max(target_volatility, min_volatility + 0.001)  # Add a small buffer

        # If target_volatility was adjusted, inform the user
        if adjusted_target_volatility > target_volatility:
            print(f"Target volatility {target_volatility:.3f} is below the minimum possible volatility {min_volatility:.3f}.")
            print(f"Adjusting target volatility to {adjusted_target_volatility:.3f}.")

        weights = ef.efficient_risk(adjusted_target_volatility)
    elif optimization_method == 'efficient_return':
        weights = ef.efficient_return(target_return=mu.mean(), risk_free_rate=risk_free_rate)
    elif optimization_method == 'max_sharpe':
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")

    # Get the optimized weights
    cleaned_weights = ef.clean_weights()

    # Calculate performance metrics
    expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    performance = {
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'sharpe_ratio': sharpe_ratio
    }

    return cleaned_weights, performance

def allocate_portfolio(weights, latest_prices, total_portfolio_value=10000):
    """
    Allocate portfolio to discrete number of shares.

    Args:
        weights (dict): Portfolio weights
        latest_prices (pandas.Series): Latest prices for each asset
        total_portfolio_value (float): Total portfolio value in currency units

    Returns:
        dict: Allocation of shares for each asset
    """
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.greedy_portfolio()
    return allocation, leftover

def plot_efficient_frontier(prices, points=100, risk_free_rate=0.02):
    """
    Plot the efficient frontier.

    Args:
        prices (pandas.DataFrame): DataFrame containing historical prices
        points (int): Number of points to plot
        risk_free_rate (float): Risk-free rate
    """
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # Plot the efficient frontier
    plotting.plot_efficient_frontier(EfficientFrontier(mu, S, weight_bounds=(0, 1)), points=points)

    # Find and plot the tangency portfolio (max Sharpe ratio)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    plt.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Plot the Capital Market Line (CML)
    plt.axline((0, risk_free_rate), (std_tangent, ret_tangent), color="black", linestyle="--", label="Capital Market Line")

    # Format and show the plot
    plt.title("Efficient Frontier")
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.show()

def main():
    """
    Main function to demonstrate portfolio optimization.
    """
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    start_date = '2018-01-01'
    end_date = dt.datetime.now().strftime('%Y-%m-%d')

    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    prices = download_data(tickers, start_date, end_date)

    # Check for missing data
    missing_data = prices.isna().sum()
    if missing_data.sum() > 0:
        print("Warning: Missing data detected:")
        print(missing_data[missing_data > 0])
        # Fill missing values with forward fill method
        prices = prices.fillna(method='ffill')

    print("\nOptimizing portfolio...")
    # Optimize using different methods
    methods = ['max_sharpe', 'efficient_risk']
    for method in methods:
        print(f"\nMethod: {method}")
        weights, performance = optimize_portfolio(prices, optimization_method=method)

        # Print results
        print("\nOptimal weights:")
        for ticker, weight in weights.items():
            if weight > 0.01:  # Only show significant allocations
                print(f"{ticker}: {weight:.4f}")

        print("\nPortfolio Performance:")
        print(f"Expected annual return: {performance['expected_return']:.4f}")
        print(f"Expected annual volatility: {performance['expected_volatility']:.4f}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.4f}")

        # Allocate portfolio
        latest_prices = get_latest_prices(prices)
        allocation, leftover = allocate_portfolio(weights, latest_prices)

        print("\nDiscrete Allocation:")
        for ticker, shares in allocation.items():
            print(f"{ticker}: {shares} shares (${shares * latest_prices[ticker]:.2f})")
        print(f"Funds remaining: ${leftover:.2f}")

    # Plot the efficient frontier
    print("\nPlotting the efficient frontier...")
    plot_efficient_frontier(prices)

if __name__ == "__main__":
    main()
