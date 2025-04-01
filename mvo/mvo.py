import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import datetime as dt
import backtrader as bt
import backtrader.analyzers as btanalyzers
import io
import os

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

class OptimizedPortfolioStrategy(bt.Strategy):
    """
    Backtrader strategy that implements a portfolio with optimized weights.
    """
    params = (
        ('weights', {}),  # Dictionary of weights for each asset
        ('rebalance_interval', 'monthly'),  # How often to rebalance: 'daily', 'weekly', 'monthly'
    )

    def __init__(self):
        # Store references to all data feeds
        self.datas_dict = {data._name: data for data in self.datas}

        # Initialize portfolio value for tracking
        self.portfolio_value = []
        self.asset_values = {name: [] for name in self.datas_dict.keys()}

        # Set rebalancing schedule
        if self.p.rebalance_interval == 'daily':
            self.rebalance_check = self.check_daily
        elif self.p.rebalance_interval == 'weekly':
            self.rebalance_check = self.check_weekly
        else:  # default to monthly
            self.rebalance_check = self.check_monthly

        # Flag to track if we've invested
        self.invested = False

    def check_daily(self):
        return True

    def check_weekly(self):
        # Rebalance on Mondays (weekday=0)
        return self.data.datetime.date(0).weekday() == 0

    def check_monthly(self):
        # Rebalance on first trading day of the month
        current_date = self.data.datetime.date(0)
        previous_date = self.data.datetime.date(-1) if len(self) > 1 else None

        if previous_date is None:
            return True

        return current_date.month != previous_date.month

    def next(self):
        # Record portfolio and asset values
        self.portfolio_value.append(self.broker.getvalue())
        for name, data in self.datas_dict.items():
            position = self.getposition(data).size
            price = data.close[0]
            self.asset_values[name].append(position * price)

        # Check if we need to rebalance
        if not self.invested or self.rebalance_check():
            self.rebalance_portfolio()
            self.invested = True

    def rebalance_portfolio(self):
        # Close all existing positions
        for data in self.datas:
            self.close(data)

        # Calculate the current portfolio value
        portfolio_value = self.broker.getvalue()

        # Open new positions based on target weights
        for name, data in self.datas_dict.items():
            if name in self.p.weights and self.p.weights[name] > 0:
                target_value = portfolio_value * self.p.weights[name]
                price = data.close[0]
                size = target_value / price
                self.buy(data, size=size)

def backtest_portfolio(prices, test_ratio=0.3, optimization_method='max_sharpe', risk_free_rate=0.02, target_volatility=0.15, rebalance_interval='monthly'):
    """
    Backtest a portfolio optimization strategy using Backtrader.

    Args:
        prices (pandas.DataFrame): DataFrame containing historical prices
        test_ratio (float): Ratio of data to use for testing (e.g., 0.3 means 30% of data for testing)
        optimization_method (str): Method to use for optimization ('efficient_risk', 'efficient_return', or 'max_sharpe')
        risk_free_rate (float): Risk-free rate used for Sharpe ratio calculation
        target_volatility (float): Target volatility for 'efficient_risk' method
        rebalance_interval (str): How often to rebalance the portfolio ('daily', 'weekly', 'monthly')

    Returns:
        tuple: (results_dict, cerebro) where:
            - results_dict: Dictionary containing backtest results including:
                - training_weights: Optimized weights from training period
                - training_performance: Performance metrics from training period
                - test_performance: Performance metrics from test period
                - cumulative_returns: DataFrame of cumulative returns for the portfolio and individual assets
            - cerebro: Backtrader cerebro instance used for backtesting (can be used for plotting)
    """
    # Split data into training and testing periods
    split_idx = int(len(prices) * (1 - test_ratio))
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]

    print(f"Training period: {train_prices.index[0]} to {train_prices.index[-1]}")
    print(f"Testing period: {test_prices.index[0]} to {test_prices.index[-1]}")

    # Optimize portfolio using training data
    weights, training_performance = optimize_portfolio(
        train_prices, 
        optimization_method=optimization_method,
        risk_free_rate=risk_free_rate,
        target_volatility=target_volatility
    )

    # Initialize Backtrader cerebro engine
    cerebro = bt.Cerebro()

    # Set initial cash
    initial_cash = 100000.0
    cerebro.broker.setcash(initial_cash)

    # Add commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Add data feeds for each ticker
    for ticker in test_prices.columns:
        # Convert price data to Backtrader format
        ticker_data = test_prices[ticker].dropna()

        # Create a proper DataFrame for Backtrader
        df = pd.DataFrame(index=ticker_data.index)
        df['open'] = ticker_data.values
        df['high'] = ticker_data.values
        df['low'] = ticker_data.values
        df['close'] = ticker_data.values
        df['volume'] = 0
        df['openinterest'] = 0

        # Create a data feed
        data = bt.feeds.PandasData(
            dataname=df,
            name=ticker
        )

        # Add the data feed to cerebro
        cerebro.adddata(data)

    # Add the strategy
    cerebro.addstrategy(OptimizedPortfolioStrategy, weights=weights, rebalance_interval=rebalance_interval)

    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=risk_free_rate, annualize=True)
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)

    # Run the backtest
    results = cerebro.run()
    strategy = results[0]

    # Get the analyzers
    sharpe_ratio = strategy.analyzers.sharpe.get_analysis()['sharperatio']
    returns = strategy.analyzers.returns.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    time_return = strategy.analyzers.time_return.get_analysis()

    # Calculate performance metrics
    total_return = cerebro.broker.getvalue() / initial_cash - 1

    # Convert time_return to a pandas Series for further calculations
    daily_returns = pd.Series({dt.datetime.combine(k, dt.time()): v for k, v in time_return.items()})
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    annual_volatility = daily_returns.std() * np.sqrt(252)
    max_drawdown = drawdown['max']['drawdown'] / 100 if 'max' in drawdown else 0

    # Create test performance dictionary
    test_performance = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

    # Create cumulative returns DataFrame
    portfolio_values = pd.Series(strategy.portfolio_value, index=test_prices.index[-(len(strategy.portfolio_value)):])
    cumulative_returns = portfolio_values / portfolio_values.iloc[0]

    # Create asset cumulative returns
    asset_values = {}
    for ticker, values in strategy.asset_values.items():
        if values:  # Check if the list is not empty
            asset_values[ticker] = pd.Series(values, index=test_prices.index[-(len(values)):])

    # Normalize asset values to get cumulative returns
    asset_cumulative_returns = {}
    for ticker, values in asset_values.items():
        if not values.empty and values.iloc[0] != 0:
            asset_cumulative_returns[ticker] = values / values.iloc[0]

    # Combine portfolio and asset cumulative returns for visualization
    all_returns = pd.DataFrame({'Portfolio': cumulative_returns})
    for ticker, returns in asset_cumulative_returns.items():
        all_returns[ticker] = returns

    results_dict = {
        'training_weights': weights,
        'training_performance': training_performance,
        'test_performance': test_performance,
        'cumulative_returns': all_returns
    }

    return results_dict, cerebro

def plot_backtest_results(backtest_results, use_backtrader_plot=False, cerebro=None):
    """
    Plot the results of a portfolio backtest.

    Args:
        backtest_results (dict): Dictionary containing backtest results from backtest_portfolio function
        use_backtrader_plot (bool): Whether to use Backtrader's built-in plotting (requires cerebro)
        cerebro (bt.Cerebro): Cerebro instance used for backtesting (required if use_backtrader_plot=True)
    """
    if use_backtrader_plot and cerebro:
        # Use Backtrader's built-in plotting
        cerebro.plot(style='candlestick', barup='green', bardown='red',
                    plotdist=0.1, header=True, volume=False)
    else:
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        backtest_results['cumulative_returns'].plot()
        plt.title('Cumulative Returns during Test Period')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot portfolio composition
        weights = backtest_results['training_weights']
        significant_weights = {k: v for k, v in weights.items() if v > 0.01}

        plt.figure(figsize=(10, 6))
        plt.pie(significant_weights.values(), labels=significant_weights.keys(), autopct='%1.1f%%')
        plt.title('Portfolio Composition')
        plt.axis('equal')
        plt.show()

        # Plot performance metrics
        test_perf = backtest_results['test_performance']
        metrics = ['total_return', 'annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
        values = [test_perf[m] for m in metrics]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)

        plt.title('Performance Metrics')
        plt.ylabel('Value')
        plt.grid(axis='y')
        plt.show()

def main():
    """
    Main function to demonstrate portfolio optimization and backtesting.
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

    # Perform backtesting with Backtrader
    print("\nPerforming portfolio backtesting with Backtrader...")

    # Set backtest parameters
    rebalance_interval = 'monthly'  # Options: 'daily', 'weekly', 'monthly'
    use_backtrader_plot = False     # Whether to use Backtrader's built-in plotting

    # Run backtest
    backtest_results, cerebro = backtest_portfolio(
        prices, 
        test_ratio=0.3, 
        optimization_method='max_sharpe',
        rebalance_interval=rebalance_interval
    )

    # Print backtest results
    print("\nBacktest Results:")
    print("\nTraining Period Performance:")
    print(f"Expected annual return: {backtest_results['training_performance']['expected_return']:.4f}")
    print(f"Expected annual volatility: {backtest_results['training_performance']['expected_volatility']:.4f}")
    print(f"Sharpe Ratio: {backtest_results['training_performance']['sharpe_ratio']:.4f}")

    print("\nTest Period Performance:")
    print(f"Total return: {backtest_results['test_performance']['total_return']:.4f}")
    print(f"Annual return: {backtest_results['test_performance']['annual_return']:.4f}")
    print(f"Annual volatility: {backtest_results['test_performance']['annual_volatility']:.4f}")
    print(f"Sharpe Ratio: {backtest_results['test_performance']['sharpe_ratio']:.4f}")
    print(f"Maximum drawdown: {backtest_results['test_performance']['max_drawdown']:.4f}")

    # Plot backtest results
    print("\nPlotting backtest results...")
    plot_backtest_results(backtest_results, use_backtrader_plot=use_backtrader_plot, cerebro=cerebro)

if __name__ == "__main__":
    main()
