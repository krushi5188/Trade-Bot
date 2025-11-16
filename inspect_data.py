import pandas as pd
import os

def analyze_btc_data():
    """
    Loads the feature dataset and analyzes the btc_close column for flatness.
    """
    FEATURE_PATH = 'data/processed/features_v2_final.parquet'

    if not os.path.exists(FEATURE_PATH):
        print(f"ERROR: Feature file not found at {FEATURE_PATH}")
        return

    print(f"Loading dataset from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)
    btc_close = df['btc_close']

    print("\\n--- BTC Close Price Analysis ---")
    print(btc_close.describe())

    # Identify periods of flatness
    flat_periods = (btc_close.diff() == 0).astype(int).groupby(btc_close.ne(btc_close.shift()).cumsum()).sum()
    long_flat_periods = flat_periods[flat_periods > 24] # More than a full day of no price change

    if not long_flat_periods.empty:
        print(f"\\nFound {len(long_flat_periods)} long periods of flat prices.")
        print("This is a major data quality issue.")
        # You could add more detailed analysis here to print the start/end of flat periods
    else:
        print("\\nNo significant flat periods found in the data.")

if __name__ == '__main__':
    analyze_btc_data()
