# This file will contain the core logic for our vectorized backtesting engine.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the new StrategyOverlay
from .strategy_overlay import StrategyOverlay

class VectorizedBacktester:
    """
    A class to perform a vectorized backtest of a trading strategy.
    """

    def __init__(self, price_data, signals, initial_capital=100000, strategy_overlay=None):
        """
        Args:
            price_data (pd.Series): Series of prices for the asset.
            signals (pd.Series): Series of trading signals (-1, 0, 1).
            initial_capital (float): The starting capital for the backtest.
            strategy_overlay (StrategyOverlay, optional): An object to apply
                                                       strategy overlays.
        """
        self.price_data = price_data
        self.signals = signals
        self.initial_capital = initial_capital
        self.strategy_overlay = strategy_overlay # Store the overlay object
        self.positions = self.generate_positions()
        self.portfolio = self.backtest_portfolio()

    def generate_positions(self):
        """
        Generates a series of positions based on the signals.
        If a strategy_overlay is provided, it uses it for position sizing.
        """
        if self.strategy_overlay:
            # Use the overlay to get volatility-adjusted positions
            positions = self.strategy_overlay.get_volatility_adjusted_positions(self.signals)
        else:
            # Default to 1 unit of asset per trade
            positions = self.signals

        # Shift positions to avoid lookahead bias in all cases
        return positions.shift(1).fillna(0)

    def backtest_portfolio(self):
        """
        Calculates the portfolio value over time.
        """
        portfolio = pd.DataFrame(index=self.price_data.index)
        portfolio['price'] = self.price_data
        portfolio['signal'] = self.signals
        portfolio['position'] = self.positions

        # Calculate returns of the asset
        portfolio['market_returns'] = self.price_data.pct_change()

        # Calculate returns of the strategy
        portfolio['strategy_returns'] = portfolio['market_returns'] * portfolio['position']

        # Calculate cumulative returns
        portfolio['cumulative_market_returns'] = (1 + portfolio['market_returns']).cumprod()
        portfolio['cumulative_strategy_returns'] = (1 + portfolio['strategy_returns']).cumprod()

        # Calculate total portfolio value
        portfolio['total_value'] = self.initial_capital * portfolio['cumulative_strategy_returns']

        return portfolio

    def get_performance_metrics(self):
        """
        Calculates and returns key performance metrics.
        """
        total_return = self.portfolio['cumulative_strategy_returns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(self.portfolio)) - 1 # Assuming 252 trading days

        # Sharpe Ratio
        sharpe_ratio = self.calculate_sharpe_ratio()

        # Max Drawdown
        max_drawdown = self.calculate_max_drawdown()

        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}"
        }
        return metrics

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """
        Calculates the Sharpe Ratio.
        """
        excess_returns = self.portfolio['strategy_returns'] - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        return sharpe_ratio

    def calculate_max_drawdown(self):
        """
        Calculates the Maximum Drawdown.
        """
        cumulative_returns = self.portfolio['cumulative_strategy_returns']
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def plot_equity_curve(self, output_path):
        """
        Plots the equity curve of the strategy vs. a buy-and-hold approach.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(self.portfolio['cumulative_strategy_returns'], label='Strategy')
        plt.plot(self.portfolio['cumulative_market_returns'], label='Buy & Hold (Market)')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        print(f"Equity curve plot saved to {output_path}")

if __name__ == '__main__':
    # This is a placeholder for a test run of the backtester.
    # We will replace this with actual model predictions later.
    print("Backtester class defined. Ready to be integrated with model predictions.")
