# This script will test the "London Lunch Pullback" hypothesis.

import os
import sys
import pandas as pd
import numpy as np
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_london_lunch_pullback(df, asset_column):
    """
    Tests the hypothesis that a pullback occurs during the London lunch session
    relative to the morning session's trend for a given asset.
    """
    print(f"\n[{time.ctime()}] --- Testing London Lunch Pullback Hypothesis for {asset_column} ---")

    # 1. Define Session Times (in UTC)
    london_morning_start = '08:00'
    london_morning_end = '12:00'
    london_lunch_start = '12:00'
    london_lunch_end = '14:00'

    # 2. Iterate Through Each Day and Test Hypothesis
    pullback_days = 0
    total_days = 0

    # Group by day
    for day, daily_data in df.groupby(df.index.date):
        total_days += 1

        try:
            # Get morning session data
            morning_session = daily_data.between_time(london_morning_start, london_morning_end)
            if len(morning_session) < 2:
                continue

            # Calculate morning trend
            morning_start_price = morning_session[asset_column].iloc[0]
            morning_end_price = morning_session[asset_column].iloc[-1]
            morning_trend = 1 if morning_end_price > morning_start_price else -1

            # Get lunch session data
            lunch_session = daily_data.between_time(london_lunch_start, london_lunch_end)
            if len(lunch_session) < 2:
                continue

            # Check for a pullback
            lunch_low = lunch_session[asset_column].min()
            lunch_high = lunch_session[asset_column].max()

            if morning_trend == 1 and lunch_low < morning_end_price:
                pullback_days += 1
            elif morning_trend == -1 and lunch_high > morning_end_price:
                pullback_days += 1

        except Exception as e:
            continue

    # 3. Report Results
    pullback_percentage = (pullback_days / total_days) * 100 if total_days > 0 else 0
    print(f"Total days analyzed: {total_days}")
    print(f"Days with a pullback: {pullback_days}")
    print(f"Pullback Occurrence: {pullback_percentage:.2f}%")
    print("-------------------------------------------------")

def main():
    """
    Main function to run the hypothesis tester.
    """
    # Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/master_dataset_lean.parquet')
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index)

    # Test for BTC
    test_london_lunch_pullback(df, 'btc_close')

    # Test for Gold
    test_london_lunch_pullback(df, 'gld_close')

if __name__ == '__main__':
    main()
