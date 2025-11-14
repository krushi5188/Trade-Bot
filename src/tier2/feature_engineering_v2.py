# This script is the next generation of our feature engineering pipeline.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import time
from functools import lru_cache
from numpy.lib.stride_tricks import as_strided

def apply_kalman_filter_sm(series):
    """Applies a Kalman Filter to a time series for smoothing."""
    series_with_freq = series.asfreq('h')
    model = sm.tsa.UnobservedComponents(series_with_freq, 'local level')
    result = model.fit(disp=False)
    smoothed = result.level.smoothed
    return smoothed

@lru_cache(maxsize=None)
def hurst(ts_tuple):
    """Calculates the Hurst Exponent of a time series."""
    ts = np.array(ts_tuple)
    if len(ts) < 100:
        return np.nan
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def rolling_window(a, window):
    """Creates a rolling window view of a NumPy array without copying data."""
    shape = (a.shape[0] - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return as_strided(a, shape=shape, strides=strides)

def create_base_features(df):
    """Applies the foundational feature engineering techniques from Tier 1."""
    print("--- Applying Tier 1 Base Features ---")

    asset_prefixes = ['btc', 'eur', 'gld']
    for prefix in asset_prefixes:
        close_col = f'{prefix}_close'
        kalman_col = f'{prefix}_close_kalman'
        print(f"Applying Kalman Filter to {close_col}...")
        df[kalman_col] = apply_kalman_filter_sm(df[close_col])

    for prefix in asset_prefixes:
        close_col = f'{prefix}_close'
        hurst_col = f'{prefix}_hurst'
        print(f"Calculating rolling Hurst Exponent for {close_col} with high-performance NumPy...")
        start_time = time.time()

        window_size = 100
        price_series = df[close_col].values
        windows = rolling_window(price_series, window_size)

        hurst_values = np.array([hurst(tuple(window)) for window in windows])

        padded_hurst = np.full(df.shape[0], np.nan)
        padded_hurst[window_size-1:] = hurst_values
        df[hurst_col] = padded_hurst

        end_time = time.time()
        print(f"  -> Finished in {end_time - start_time:.2f} seconds.")

    print("Base feature creation complete.")
    return df

def create_interaction_features(df):
    """Engineers advanced features by capturing interactions between variables."""
    print("\\n--- Engineering V2 Interaction Features ---")

    df['sentiment_x_hurst'] = df['sentiment_score'] * df['btc_hurst']
    print("Created feature: sentiment_score * btc_hurst")

    if 'event_name' not in df.columns:
        df['event_name'] = None
    df['is_high_impact_event'] = (~df['event_name'].isna()).astype(int)

    df['event_x_hurst'] = df['is_high_impact_event'] * df['btc_hurst']
    print("Created feature: is_high_impact_event * btc_hurst")

    df['btc_volatility'] = df['btc_close'].pct_change().rolling(window=24).std()
    df['sentiment_adjusted_by_vol'] = df['sentiment_score'] / df['btc_volatility']
    print("Created feature: sentiment_score / btc_volatility")

    print("Interaction feature creation complete.")
    return df

if __name__ == '__main__':
    # --- FIX: Run on a smaller subset of data to prevent timeout ---
    DEBUG_MODE = True
    # --- END FIX ---

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')

    master_filepath = os.path.join(PROCESSED_DIR, 'master_dataset.parquet')
    df = pd.read_parquet(master_filepath)
    print("Master Dataset Loaded.")

    if DEBUG_MODE:
        print("\\n*** RUNNING IN DEBUG MODE ON A SUBSET OF DATA ***")
        df = df.tail(5000) # Use the most recent 5000 data points

    df = create_base_features(df)
    df = create_interaction_features(df)

    print("\\nCleaning and filling NaN values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_advanced.parquet')
    df.to_parquet(output_filepath)

    print(f"\\nAdvanced feature dataset (V2) saved to {output_filepath}")
    print("\\n--- Advanced Feature Dataset Info ---")
    df.info()
