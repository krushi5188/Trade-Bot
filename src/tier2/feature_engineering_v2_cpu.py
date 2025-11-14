# This is a robust, hybrid GPU/CPU feature engineering script.
# It uses a reliable CPU-based Hurst calculation to prevent errors.

import cudf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import gc
import time

# --- Robustness & Checkpointing ---
def save_checkpoint_gpu(df, name):
    """Saves a checkpoint of the cuDF dataframe."""
    checkpoint_path = f'/content/drive/MyDrive/trading-ai/data/processed/checkpoint_{name}_gpu.parquet'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    df.to_parquet(checkpoint_path)
    print(f"[{time.ctime()}] GPU Checkpoint saved: {checkpoint_path}")

# --- CPU-Bound Functions ---
def apply_kalman_filter_on_cpu(series_pd):
    """Applies the statsmodels Kalman Filter on a pandas Series on the CPU."""
    resampled_series = series_pd.resample('h').ffill()
    if resampled_series.isnull().all():
        return resampled_series
    model = sm.tsa.UnobservedComponents(resampled_series.dropna(), 'local level')
    result = model.fit(disp=False)
    smoothed_values = result.level.smoothed
    smoothed_series = pd.Series(smoothed_values, index=resampled_series.dropna().index)
    return smoothed_series.reindex(resampled_series.index)

def hurst_on_cpu(ts):
    """
    A reliable, CPU-based Hurst Exponent calculation.
    """
    ts = np.asarray(ts)
    if len(ts) < 100:
        return 0.5

    lags = range(2, 100)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std_dev = np.std(diff)
        if std_dev == 0:
            return 0.5
        tau.append(np.sqrt(std_dev))

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- Main Execution ---
if __name__ == '__main__':
    try:
        start_time = time.time()

        print(f"[{time.ctime()}] Starting robust feature engineering...")

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')
        output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_final.parquet')

        # Load data onto the CPU with pandas
        df_pd = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Lean dataset loaded onto CPU. Shape: {df_pd.shape}")

        # Kalman Filters (CPU)
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Kalman Filter for {col}...")
            df_pd[f'{col}_kalman'] = apply_kalman_filter_on_cpu(df_pd[col])
            gc.collect()

        # Hurst Exponent (CPU)
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Hurst Exponent for {col}...")
            df_pd[f'{col}_hurst'] = df_pd[col].rolling(window=100).apply(hurst_on_cpu, raw=True)
            gc.collect()

        # Interaction Features (CPU)
        print(f"[{time.ctime()}] Processing interaction features...")
        df_pd['sentiment_score'] = df_pd['sentiment_score'].astype('float32')
        df_pd['btc_hurst'] = df_pd['btc_hurst'].astype('float32')
        df_pd['sentiment_x_hurst'] = df_pd['sentiment_score'] * df_pd['btc_hurst']
        if 'event_name' not in df_pd.columns: df_pd['event_name'] = None
        df_pd['is_high_impact_event'] = (~df_pd['event_name'].isna()).astype('int8')
        df_pd['event_x_hurst'] = df_pd['is_high_impact_event'] * df_pd['btc_hurst']
        df_pd['btc_volatility'] = df_pd['btc_close'].pct_change().rolling(24).std()
        df_pd['sentiment_adjusted_by_vol'] = df_pd['sentiment_score'] / df_pd['btc_volatility']

        print(f"[{time.ctime()}] Final cleaning and NaN filling...")
        df_pd.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_pd.fillna(method='ffill', inplace=True)
        df_pd.fillna(method='bfill', inplace=True)

        print(f"[{time.ctime()}] Saving final dataset to {output_filepath}...")
        df_pd.to_parquet(output_filepath)
        print(f"[{time.ctime()}] Final dataset saved successfully.")

        elapsed_time = time.time() - start_time
        print(f"[{time.ctime()}] Total execution time: {elapsed_time / 60:.2f} minutes.")

    except Exception as e:
        print(f"[{time.ctime()}] An error occurred: {e}")
        import traceback
        traceback.print_exc()
