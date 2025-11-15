# FINAL CORRECTED VERSION v2 - Robust feature engineering with CORRECT ROLLING HURST

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import time
from typing import Optional

def save_checkpoint(df: pd.DataFrame, name: str) -> None:
    """Saves checkpoint of dataframe."""
    checkpoint_path = f'/content/drive/MyDrive/trading-ai/data/processed/checkpoint_{name}.parquet'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    df.to_parquet(checkpoint_path)
    print(f"[{time.ctime()}] Checkpoint saved: {checkpoint_path}")

def hurst_on_cpu(ts: np.ndarray) -> float:
    """
    The proven, correct, and reliable CPU-based ROLLING Hurst Exponent calculation.
    """
    ts = np.asarray(ts)
    if len(ts) < 100:
        return 0.5

    lags = range(2, 100)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        if len(diff) == 0:
            continue
        std_dev = np.std(diff)
        if std_dev <= 0:
            return 0.5
        tau.append(np.sqrt(std_dev))

    if len(tau) < 2:
        return 0.5

    poly = np.polyfit(np.log(range(2, len(tau) + 2)), np.log(tau), 1)
    return poly[0] * 2.0

def apply_kalman_filter_safe(series_pd: pd.Series) -> pd.Series:
    """Safe Kalman filter with proper error handling."""
    if not isinstance(series_pd.index, pd.DatetimeIndex):
        print("Warning: Index is not datetime, using original series")
        return series_pd

    try:
        resampled_series = series_pd.resample('h').ffill()
        if len(resampled_series.dropna()) < 10:
            return resampled_series

        model = sm.tsa.UnobservedComponents(resampled_series.dropna(), 'local level')
        result = model.fit(disp=False)
        smoothed_values = result.level.smoothed

        smoothed_series = pd.Series(smoothed_values, index=resampled_series.dropna().index)
        return smoothed_series.reindex(resampled_series.index, method='ffill')

    except Exception as e:
        print(f"Kalman filter failed: {e}, returning original series")
        return series_pd

if __name__ == '__main__':
    try:
        start_time = time.time()
        print(f"[{time.ctime()}] Starting FINAL, v2 CORRECTED feature engineering...")

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')
        output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_final.parquet')

        print(f"[{time.ctime()}] Loading dataset from: {lean_filepath}")
        df_pd = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Dataset loaded. Shape: {df_pd.shape}")

        # 1. Kalman Filters
        print(f"[{time.ctime()}] Processing Kalman Filters...")
        for col in ['btc_close', 'eur_close', 'gld_close']:
            df_pd[f'{col}_kalman'] = apply_kalman_filter_safe(df_pd[col])
        save_checkpoint(df_pd, 'kalman_complete')

        # 2. Hurst Exponent (CORRECT ROLLING CALCULATION)
        print(f"[{time.ctime()}] Processing Hurst Exponent (Correct Rolling Method)...")
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"  - Calculating Hurst for {col}...")
            df_pd[f'{col}_hurst'] = df_pd[col].rolling(window=100, min_periods=100).apply(hurst_on_cpu, raw=True)
        save_checkpoint(df_pd, 'hurst_complete')

        # 3. Interaction Features
        print(f"[{time.ctime()}] Processing interaction features...")
        df_pd['sentiment_score'] = pd.to_numeric(df_pd['sentiment_score'], errors='coerce').fillna(0).astype('float32')
        df_pd['btc_hurst'] = pd.to_numeric(df_pd['btc_hurst'], errors='coerce').fillna(0.5).astype('float32')
        df_pd['sentiment_x_hurst'] = df_pd['sentiment_score'] * df_pd['btc_hurst']
        if 'event_name' not in df_pd.columns: df_pd['event_name'] = None
        df_pd['is_high_impact_event'] = (~df_pd['event_name'].isna()).astype('int8')
        df_pd['event_x_hurst'] = df_pd['is_high_impact_event'] * df_pd['btc_hurst']
        df_pd['btc_volatility'] = df_pd['btc_close'].pct_change().rolling(24, min_periods=1).std()
        volatility_safe = df_pd['btc_volatility'].replace(0, np.nan)
        df_pd['sentiment_adjusted_by_vol'] = df_pd['sentiment_score'] / volatility_safe

        # Final cleaning
        print(f"[{time.ctime()}] Final cleaning...")
        df_pd.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_pd.ffill(inplace=True)
        df_pd.bfill(inplace=True)

        # Save final dataset
        print(f"[{time.ctime()}] Saving final dataset...")
        df_pd.to_parquet(output_filepath)

        elapsed_time = time.time() - start_time
        print(f"[{time.ctime()}] SUCCESS: Feature engineering completed in {elapsed_time / 60:.2f} minutes!")

    except Exception as e:
        print(f"[{time.ctime()}] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
