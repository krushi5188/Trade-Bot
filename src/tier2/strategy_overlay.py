# This module will contain the logic for the strategy overlay, including
# dynamic position sizing and risk management rules.

import pandas as pd
import numpy as np

class StrategyOverlay:
    """
    A class to apply strategy overlays, such as dynamic position sizing
    and risk management, to a set of trading signals.
    """

    def __init__(self, price_data, volatility_lookback=21, volatility_target=0.02):
        """
        Args:
            price_data (pd.Series): Series of prices for the asset.
            volatility_lookback (int): The lookback period for calculating volatility.
            volatility_target (float): The target volatility for position sizing.
        """
        self.price_data = price_data
        self.volatility_lookback = volatility_lookback
        self.volatility_target = volatility_target
        self.volatility = self._calculate_volatility()

    def _calculate_volatility(self):
        """
        Calculates the rolling annualized volatility of the asset.
        """
        returns = self.price_data.pct_change()
        # Calculate rolling standard deviation and annualize it for hourly data
        rolling_volatility = returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252 * 24)
        return rolling_volatility

    def get_volatility_adjusted_positions(self, signals):
        """
        Adjusts position sizes based on market volatility.
        Takes smaller positions in high-volatility periods and vice-versa.
        """
        # Inverse volatility scaling: position size is inversely proportional to volatility
        with np.errstate(divide='ignore', invalid='ignore'):
            position_sizing = self.volatility_target / self.volatility

        # Where volatility was 0, division results in inf. This represents max position size.
        # Any other NaNs (e.g., at the start) should be forward-filled then filled with 0.
        position_sizing = position_sizing.reindex(signals.index, method='ffill').fillna(0)

        # Cap leverage to a maximum of 2x the base signal
        position_sizing = position_sizing.clip(0, 2)

        # Apply sizing to the original signals
        adjusted_positions = signals.astype(float) * position_sizing

        return adjusted_positions.fillna(0)

    def apply_risk_management(self, signals):
        """
        Placeholder for applying risk management rules like stop-loss or take-profit.
        For a vectorized backtester, this is complex. For now, it will just pass
        the signals through.

        Args:
            signals (pd.Series): The trading signals.

        Returns:
            pd.Series: The signals after applying risk management rules.
        """
        # In a more advanced, event-driven backtester, we would implement
        # logic here to exit positions based on price movements.
        return signals

if __name__ == '__main__':
    # Placeholder for a test run of the StrategyOverlay.
    print("StrategyOverlay class defined. Ready for integration.")
