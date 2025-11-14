# This is a memory-optimized feature engineering script.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import gc

# --- Memory-Optimized Functions ---

def apply_kalman_filter_sm(series):
    """Applies a Kalman Filter with memory efficiency."""
    series_float32 = series.astype(np.float32)
    # The .asfreq('h') call is the source of the memory issue.
    # We will resample with a more memory-efficient method.
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
    """Calculates the Hurst Exponent."""
    ts = np.asarray(ts, dtype=np.float32)
    if len(ts) < 100:
        return 0.5

    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    tau = [val for val in tau if val > 0]
    if not tau:
        return 0.5

    poly = np.polyfit(np.log(range(2, len(tau) + 2)), np.log(tau), 1)
    return poly[0] * 2.0

# --- Main Execution ---

if __name__ == '__main__':
    try:
        print("Starting memory-optimized feature engineering...")

        # Define paths
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')
        output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_final.parquet')

        # Load lean dataset
        df = pd.read_parquet(lean_filepath)
        print("Lean dataset loaded.")

        # Process features one by one to save memory

        # Kalman Filters
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"Processing Kalman Filter for {col}...")
            df[f'{col}_kalman'] = apply_kalman_filter_sm(df[col])
            gc.collect()

        # Hurst Exponent
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"Processing Hurst Exponent for {col}...")
            df[f'{col}_hurst'] = df[col].rolling(window=100).apply(hurst, raw=True).astype(np.float32)
            gc.collect()

        # Interaction Features
        print("Processing interaction features...")
        df['sentiment_x_hurst'] = (df['sentiment_score'] * df['btc_hurst']).astype(np.float32)
        if 'event_name' not in df.columns: df['event_name'] = None
        df['is_high_impact_event'] = (~df['event_name'].isna()).astype(np.int8)
        df['event_x_hurst'] = (df['is_high_impact_event'] * df['btc_hurst']).astype(np.float32)
        df['btc_volatility'] = df['btc_close'].pct_change().rolling(window=24).std().astype(np.float32)
        df['sentiment_adjusted_by_vol'] = (df['sentiment_score'] / df['btc_volatility']).astype(np.float32)

        # Clean up
        print("Final cleaning and NaN filling...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        # Save final dataset
        print(f"Saving final dataset to {output_filepath}...")
        df.to_parquet(output_filepath)
        print("Final dataset saved.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
