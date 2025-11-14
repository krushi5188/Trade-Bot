# This is a high-performance, GPU-accelerated feature engineering script using RAPIDS cuDF.

import cudf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import gc
import time
from numba import jit

# --- Robustness & Checkpointing ---
def save_checkpoint_gpu(df, name):
    """Saves a checkpoint of the cuDF dataframe."""
    checkpoint_path = f'/content/drive/MyDrive/trading-ai/data/processed/checkpoint_{name}_gpu.parquet'
    df.to_parquet(checkpoint_path)
    print(f"[{time.ctime()}] GPU Checkpoint saved: {checkpoint_path}")

# --- CPU-Bound Function (for statsmodels) ---
def apply_kalman_filter_on_cpu(series_pd):
    """
    Applies the statsmodels Kalman Filter on a pandas Series on the CPU.
    """
    resampled_series = series_pd.resample('H').ffill()
    # Ensure there are no NaNs before fitting the model
    if resampled_series.isnull().all():
        return resampled_series
    model = sm.tsa.UnobservedComponents(resampled_series.dropna(), 'local level')
    result = model.fit(disp=False)
    smoothed = result.level.smoothed
    return smoothed.reindex(resampled_series.index)

# --- Numba-Accelerated Hurst Exponent (for pandas .apply()) ---
@jit(nopython=True)
def hurst_numba(ts):
    """
    Calculates the Hurst Exponent using Numba for acceleration.
    This function is designed to be used with .rolling().apply() on a pandas Series.
    """
    if len(ts) < 100:
        return 0.5

    lags = np.arange(2, 100)
    tau = np.zeros(len(lags), dtype=np.float64)

    for i, lag in enumerate(lags):
        diff = ts[lag:] - ts[:-lag]
        if len(diff) > 0:
            std_dev = np.std(diff)
            if std_dev > 0:
                tau[i] = np.sqrt(std_dev)

    tau = tau[tau > 0]
    if len(tau) < 2:
        return 0.5

    log_lags = np.log(np.arange(2, len(tau) + 2))
    log_tau = np.log(tau)

    # Manual linear regression (Numba compatible)
    A = np.vstack((log_lags, np.ones(len(log_lags)))).T
    slope, _ = np.linalg.lstsq(A, log_tau, rcond=None)[0]

    return slope * 2.0

# --- Main GPU-Accelerated Execution ---
if __name__ == '__main__':
    try:
        start_time = time.time()
        max_runtime = 11 * 3600

        print(f"[{time.ctime()}] Starting GPU-accelerated feature engineering...")

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')
        output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_final_gpu.parquet')

        # Load initial data directly onto the GPU
        df_gpu = cudf.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Lean dataset loaded onto GPU. Shape: {df_gpu.shape}")

        # --- Kalman Filters (Hybrid CPU/GPU) ---
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Kalman Filter for {col} (GPU -> CPU -> GPU)...")
            # 1. Move data from GPU to CPU
            series_pd = df_gpu[col].to_pandas()
            # 2. Run calculation on CPU
            smoothed_pd = apply_kalman_filter_on_cpu(series_pd)
            # 3. Move result from CPU back to GPU
            df_gpu[f'{col}_kalman'] = cudf.from_pandas(smoothed_pd)
            gc.collect()
        save_checkpoint_gpu(df_gpu, 'kalman_filters_complete')

        # --- Hurst Exponent (Hybrid CPU-Numba/GPU) ---
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Hurst Exponent for {col} (GPU -> CPU-Numba -> GPU)...")
            series_pd = df_gpu[col].to_pandas()
            hurst_values_pd = series_pd.rolling(window=100).apply(hurst_numba, raw=True)
            df_gpu[f'{col}_hurst'] = cudf.from_pandas(hurst_values_pd)
            gc.collect()
        save_checkpoint_gpu(df_gpu, 'hurst_exponent_complete')

        # --- Interaction Features (Full GPU) ---
        print(f"[{time.ctime()}] Processing interaction features on GPU...")
        df_gpu['sentiment_score'] = df_gpu['sentiment_score'].astype('float32')
        df_gpu['btc_hurst'] = df_gpu['btc_hurst'].astype('float32')
        df_gpu['sentiment_x_hurst'] = df_gpu['sentiment_score'] * df_gpu['btc_hurst']

        if 'event_name' not in df_gpu.columns: df_gpu['event_name'] = None
        df_gpu['is_high_impact_event'] = (~df_gpu['event_name'].isna()).astype('int8')
        df_gpu['event_x_hurst'] = df_gpu['is_high_impact_event'] * df_gpu['btc_hurst']

        df_gpu['btc_volatility'] = df_gpu['btc_close'].pct_change().rolling(24).std()
        df_gpu['sentiment_adjusted_by_vol'] = df_gpu['sentiment_score'] / df_gpu['btc_volatility']
        save_checkpoint_gpu(df_gpu, 'interaction_features_complete')

        # --- Finalization (on GPU) ---
        print(f"[{time.ctime()}] Final cleaning and NaN filling on GPU...")
        df_gpu = df_gpu.replace([np.inf, -np.inf], np.nan)
        df_gpu = df_gpu.fillna(method='ffill')
        df_gpu = df_gpu.fillna(method='bfill')

        print(f"[{time.ctime()}] Saving final GPU-processed dataset to {output_filepath}...")
        df_gpu.to_parquet(output_filepath)
        print(f"[{time.ctime()}] Final dataset saved successfully.")

        elapsed_time = time.time() - start_time
        print(f"[{time.ctime()}] Total GPU execution time: {elapsed_time / 60:.2f} minutes.")

    except Exception as e:
        print(f"[{time.ctime()}] An error occurred: {e}")
        import traceback
        traceback.print_exc()
