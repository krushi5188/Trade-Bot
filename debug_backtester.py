# This script is for debugging the VectorizedBacktester and StrategyOverlay interaction.

import os
import sys
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

def main():
    """
    Main function to run the debugging script.
    """
    print("--- Starting Backtester Debugging Script ---")

    # 1. Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    df = pd.read_parquet(FEATURE_PATH)

    # 2. Create a Simple Signal Series (buy every hour)
    signals = pd.Series(1, index=df.index, name="final_signal")

    # Use the last 20% of the data as the test set
    test_size = 0.2
    split_index = int(len(df) * (1 - test_size))
    df_test = df.iloc[split_index:].copy()
    signals_test = signals.iloc[split_index:]
    price_data_test = df_test['btc_close']

    # 3. Initialize Strategy Overlay
    volatility_lookback = 21
    overlay_price_data = df['btc_close'].loc[:price_data_test.index[-1]]

    print("\n--- Initializing Strategy Overlay ---")
    strategy_overlay = StrategyOverlay(price_data=overlay_price_data, volatility_lookback=volatility_lookback, volatility_target=0.02)

    # --- INSPECTION ---
    print("\n--- Inspecting Strategy Overlay ---")
    volatility = strategy_overlay.volatility.loc[signals_test.index]
    print("Volatility (head):")
    print(volatility.head())
    print(f"Volatility NaN count: {volatility.isna().sum()}")

    position_sizing = (strategy_overlay.volatility_target / volatility).clip(0, 2)
    print("\nPosition Sizing (head):")
    print(position_sizing.head())
    print(f"Position Sizing NaN count: {position_sizing.isna().sum()}")

    adjusted_positions = strategy_overlay.get_volatility_adjusted_positions(signals_test)
    print("\nAdjusted Positions from Overlay (head):")
    print(adjusted_positions.head())
    print(f"Adjusted Positions NaN count: {adjusted_positions.isna().sum()}")
    print(f"Are all adjusted positions zero? { (adjusted_positions == 0).all() }")

    # 4. Initialize Backtester
    print("\n--- Initializing Vectorized Backtester ---")
    backtester = VectorizedBacktester(price_data=price_data_test,
                                      signals=signals_test,
                                      initial_capital=100000,
                                      strategy_overlay=strategy_overlay,
                                      transaction_cost=0.001)

    # --- INSPECTION ---
    print("\n--- Inspecting Backtester ---")
    print("Final Positions in Backtester (head):")
    print(backtester.positions.head())
    print(f"Final Positions NaN count: {backtester.positions.isna().sum()}")
    print(f"Are all final positions zero? { (backtester.positions == 0).all() }")

    # 5. Run Backtest and Print Metrics
    print("\n--- Running Backtest ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n--- Debugging Script Finished ---")


if __name__ == '__main__':
    main()
