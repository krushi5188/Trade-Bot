# This is a memory-optimized and robust feature engineering script for long-running sessions.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import gc
import time

# --- Robustness Functions ---

def save_checkpoint(df, name):
    """Saves a checkpoint of the dataframe to Google Drive."""
    checkpoint_path = f'/content/drive/MyDrive/trading-ai/data/processed/checkpoint_{name}.parquet'
    df.to_parquet(checkpoint_path)
    print(f"[{time.ctime()}] Checkpoint saved: {checkpoint_path}")

# --- Memory-Optimized Feature Functions ---

def apply_kalman_filter_sm(series):
    """Applies a Kalman Filter with memory efficiency."""
    series_float32 = series.astype(np.float32)
    resampled_series = series_float32.resample('h').ffill()
    del series_float32
    gc.collect()

    model = sm.tsa.UnobservedComponents(resampled_series, 'local level')
    result = model.fit(disp=False)
    smoothed = result.level.smoothed.astype(np.float32)
    del resampled_series, model, result
    gc.collect()
    return smoothed

def hurst(ts):
    """Calculates the Hurst Exponent in a numerically stable way."""
    ts = np.asarray(ts, dtype=np.float32)
    if len(ts) < 100:
        return 0.5

    lags = range(2, 100)
    tau = []
    for lag in lags:
        # Calculate the standard deviation of the lagged differences
        diff = np.subtract(ts[lag:], ts[:-lag])
        std_dev = np.std(diff)

        # If the standard deviation is zero, the series is flat. Return 0.5 (random walk).
        if std_dev == 0:
            return 0.5

        tau.append(np.sqrt(std_dev))

    # Perform a log-log regression to get the slope (Hurst exponent)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- Main Execution ---

if __name__ == '__main__':
    try:
        start_time = time.time()
        max_runtime = 11 * 3600

        print(f"[{time.ctime()}] Starting robust, memory-optimized feature engineering...")

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')
        output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_final.parquet')

        df = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Lean dataset loaded.")

        # Kalman Filters
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Kalman Filter for {col}...")
            df[f'{col}_kalman'] = apply_kalman_filter_sm(df[col])
            gc.collect()
        save_checkpoint(df, 'kalman_filters_complete')

        # Hurst Exponent
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Hurst Exponent for {col}...")
            df[f'{col}_hurst'] = df[col].rolling(window=100).apply(hurst, raw=True).astype(np.float32)
            gc.collect()
        save_checkpoint(df, 'hurst_exponent_complete')

        # Interaction Features
        print(f"[{time.ctime()}] Processing interaction features...")
        df['sentiment_x_hurst'] = (df['sentiment_score'] * df['btc_hurst']).astype(np.float32)
        if 'event_name' not in df.columns: df['event_name'] = None
        df['is_high_impact_event'] = (~df['event_name'].isna()).astype(np.int8)
        df['event_x_hurst'] = (df['is_high_impact_event'] * df['btc_hurst']).astype(np.float32)
        df['btc_volatility'] = df['btc_close'].pct_change().rolling(window=24).std().astype(np.float32)
        df['sentiment_adjusted_by_vol'] = (df['sentiment_score'] / df['btc_volatility']).astype(np.float32)
        save_checkpoint(df, 'interaction_features_complete')

        print(f"[{time.ctime()}] Final cleaning and NaN filling...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        print(f"[{time.ctime()}] Saving final dataset to {output_filepath}...")
        df.to_parquet(output_filepath)
        print(f"[{time.ctime()}] Final dataset saved.")

        elapsed_time = time.time() - start_time
        print(f"[{time.ctime()}] Total execution time: {elapsed_time / 3600:.2f} hours.")
        if elapsed_time > max_runtime:
            print(f"[{time.ctime()}] WARNING: Execution time is approaching the session limit.")

    except Exception as e:
        print(f"[{time.ctime()}] An error occurred: {e}")
        import traceback
        traceback.print_exc()
