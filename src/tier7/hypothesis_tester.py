
import pandas as pd
import numpy as np
import os
import sys
import json

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

def generate_london_pullback_signals(df):
    """
    Generates trading signals based on the London Lunch Pullback strategy.
    - Morning session: 08:00-12:00 GMT
    - Lunch session: 12:00-14:00 GMT
    """
    # Ensure the dataframe has a datetime index in UTC
    df.index = pd.to_datetime(df.index, utc=True)

    # Resample to daily to find the morning range for each day
    daily_data = df.between_time('08:00', '12:00').resample('D')['Close']

    morning_high = daily_data.max()
    morning_low = daily_data.min()
    morning_midpoint = (morning_high + morning_low) / 2

    # Map the daily midpoint to the full 1-minute index
    df['midpoint'] = morning_midpoint.reindex(df.index, method='ffill')
    df = df.dropna(subset=['midpoint'])

    # Isolate the lunch session where trades can happen
    lunch_df = df.between_time('12:01', '14:00').copy()

    # Determine position relative to midpoint
    lunch_df['above_mid'] = lunch_df['Close'] > lunch_df['midpoint']
    lunch_df['below_mid'] = lunch_df['Close'] < lunch_df['midpoint']

    # Identify crosses of the midpoint
    lunch_df['buy_signal'] = (lunch_df['below_mid'].shift(1)) & (lunch_df['above_mid'])
    lunch_df['sell_signal'] = (lunch_df['above_mid'].shift(1)) & (lunch_df['below_mid'])

    # Combine signals into one series
    signals = pd.Series(0, index=lunch_df.index)
    signals[lunch_df['buy_signal']] = 1
    signals[lunch_df['sell_signal']] = -1

    # Ensure only one signal per day (take the first one)
    daily_signals = signals.groupby(signals.index.date).apply(lambda x: x[x != 0].first_valid_index())

    final_signals = pd.Series(0, index=df.index)
    for date, signal_idx in daily_signals.dropna().items():
        final_signals.loc[signal_idx] = signals.loc[signal_idx]

    return final_signals

def test_london_lunch_pullback():
    """
    A specific implementation to test the 'London Lunch Pullback' hypothesis.
    """
    # 1. Load Data
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'XAUUSD_1m.parquet')
    print(f"Loading data from {processed_data_path}...")
    df = pd.read_parquet(processed_data_path)
    df = df[df.index.year >= 2020].copy() # Use recent data for efficiency
    print("Data loaded.")

    print("Generating signals for London Lunch Pullback...")
    signals = generate_london_pullback_signals(df)
    print(f"Signal generation complete. Found {len(signals[signals != 0])} trading signals.")

    # 2. Backtest the Strategy
    print("Backtesting the strategy...")
    backtester = VectorizedBacktester(
        price_data=df['Close'],
        signals=signals,
        initial_capital=100000,
        transaction_cost=0.0002
    )

    metrics = backtester.get_performance_metrics()

    print("\n--- Backtest Performance: London Lunch Pullback ---")
    print(json.dumps(metrics, indent=4))
    print("---------------------------------------------------\n")

    # 3. Save Equity Curve
    output_dir = os.path.join(project_root, 'analysis/hypothesis_reports')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "london_lunch_pullback_equity_curve.png")
    backtester.plot_equity_curve(plot_path)

    return metrics

def main():
    """
    Main function to orchestrate hypothesis tests.
    """
    print("--- Starting Hypothesis Testing Framework ---")
    test_london_lunch_pullback()
    print("--- Hypothesis Testing Framework Finished ---")

if __name__ == "__main__":
    main()
