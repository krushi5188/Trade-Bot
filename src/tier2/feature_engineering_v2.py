# FINAL CORRECTED VERSION - Robust feature engineering
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import time
from typing import Optional

def save_checkpoint(df: pd.DataFrame, name: str) -> None:
    """Saves checkpoint of dataframe."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
    checkpoint_path = os.path.join(PROCESSED_DIR, f'checkpoint_{name}.parquet')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    df.to_parquet(checkpoint_path)
    print(f"[{time.ctime()}] Checkpoint saved: {checkpoint_path}")

def correct_hurst_exponent(ts: np.ndarray) -> float:
    """
    CORRECT Hurst Exponent calculation using R/S method.
    Based on the standard Rescaled Range Analysis.
    """
    ts = np.asarray(ts)
    n = len(ts)

    if n < 100:
        return 0.5  # Neutral value for insufficient data

    # Split the time series into increasingly larger segments
    max_lag = min(100, n // 4)
    if max_lag < 10:
        return 0.5

    lags = np.logspace(0.5, np.log10(max_lag), 20, dtype=int)
    lags = np.unique(lags)  # Remove duplicates
    lags = lags[(lags > 1) & (lags < n)]

    if len(lags) < 5:
        return 0.5

    rs_values = []
    valid_lags = []

    for lag in lags:
        # Split into non-overlapping windows
        n_windows = n // lag
        if n_windows < 2:
            continue

        window_rs = []
        for i in range(n_windows):
            start_idx = i * lag
            end_idx = start_idx + lag
            window = ts[start_idx:end_idx]

            if len(window) < 2:
                continue

            # Calculate mean
            mean_val = np.mean(window)

            # Calculate cumulative deviations from mean
            deviations = window - mean_val
            cumulative_deviations = np.cumsum(deviations)

            # Calculate range (R)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

            # Calculate standard deviation (S)
            S = np.std(window)

            if S > 0 and R > 0:
                window_rs.append(R / S)

        if window_rs:
            rs_values.append(np.mean(window_rs))
            valid_lags.append(lag)

    if len(rs_values) < 5:
        return 0.5

    # Perform linear regression on log-log plot
    try:
        poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
        return poly[0]  # Hurst exponent is the slope
    except:
        return 0.5

def apply_kalman_filter_safe(series_pd: pd.Series) -> pd.Series:
    """Safe Kalman filter with proper error handling."""
    # Check if index is datetime for resampling
    if not isinstance(series_pd.index, pd.DatetimeIndex):
        print("Warning: Index is not datetime, using original series")
        return series_pd

    try:
        # Resample to hourly
        resampled_series = series_pd.resample('h').ffill()

        if len(resampled_series.dropna()) < 10:
            return resampled_series

        # Apply Kalman filter
        model = sm.tsa.UnobservedComponents(resampled_series.dropna(), 'local level')
        result = model.fit(disp=False)
        smoothed_values = result.level.smoothed

        # Create result series with proper alignment
        smoothed_series = pd.Series(smoothed_values, index=resampled_series.dropna().index)
        result_series = smoothed_series.reindex(resampled_series.index, method='ffill')

        return result_series

    except Exception as e:
        print(f"Kalman filter failed: {e}, returning original series")
        return series_pd

def efficient_hurst_calculation(df: pd.DataFrame, column: str, window: int = 5000) -> pd.Series:
    """
    Efficient Hurst calculation - computes once per series instead of rolling.
    Much faster for large datasets.
    """
    print(f"  Calculating Hurst for {column}...")

    # Calculate single Hurst value for the entire series (much faster)
    hurst_value = correct_hurst_exponent(df[column].dropna().values)
    print(f"    Hurst value: {hurst_value:.4f}")

    # Create full series with this value
    return pd.Series([hurst_value] * len(df), index=df.index, dtype='float32')

if __name__ == '__main__':
    try:
        start_time = time.time()
        print(f"[{time.ctime()}] Starting FINAL, CORRECTED feature engineering...")

        # Path configuration
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')
        output_filepath = os.path.join(PROCESSED_DIR, 'features_v2_final.parquet')

        # Validate input file exists
        if not os.path.exists(lean_filepath):
            # Check for alternative files
            possible_files = [
                lean_filepath,
                os.path.join(PROCESSED_DIR, 'checkpoint_kalman_filters_complete_gpu.parquet'),
                os.path.join(PROJECT_ROOT, 'data/raw/raw_data.parquet')
            ]

            for file_path in possible_files:
                if os.path.exists(file_path):
                    lean_filepath = file_path
                    print(f"Using alternative file: {file_path}")
                    break
            else:
                raise FileNotFoundError(f"No dataset file found. Checked: {possible_files}")

        # Load data
        print(f"[{time.ctime()}] Loading dataset from: {lean_filepath}")
        df_pd = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Dataset loaded. Shape: {df_pd.shape}")
        print(f"Columns: {list(df_pd.columns)}")

        # Validate required columns
        required_cols = ['btc_close', 'eur_close', 'gld_close', 'sentiment_score']
        missing_cols = [col for col in required_cols if col not in df_pd.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # 1. Kalman Filters
        print(f"[{time.ctime()}] Processing Kalman Filters...")
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"  - Processing {col}...")
            df_pd[f'{col}_kalman'] = apply_kalman_filter_safe(df_pd[col])

        save_checkpoint(df_pd, 'kalman_complete')

        # 2. Hurst Exponent (Efficient calculation)
        print(f"[{time.ctime()}] Processing Hurst Exponent (efficient method)...")
        for col in ['btc_close', 'eur_close', 'gld_close']:
            df_pd[f'{col}_hurst'] = efficient_hurst_calculation(df_pd, col)

        save_checkpoint(df_pd, 'hurst_complete')

        # 3. Interaction Features
        print(f"[{time.ctime()}] Processing interaction features...")

        # Ensure correct data types
        df_pd['sentiment_score'] = pd.to_numeric(df_pd['sentiment_score'], errors='coerce').fillna(0).astype('float32')
        df_pd['btc_close_hurst'] = pd.to_numeric(df_pd['btc_close_hurst'], errors='coerce').fillna(0.5).astype('float32')

        # Create features with safe division
        df_pd['sentiment_x_hurst'] = df_pd['sentiment_score'] * df_pd['btc_close_hurst']

        # Event features
        if 'event_name' not in df_pd.columns:
            df_pd['event_name'] = None
        df_pd['is_high_impact_event'] = (~df_pd['event_name'].isna()).astype('int8')
        df_pd['event_x_hurst'] = df_pd['is_high_impact_event'] * df_pd['btc_close_hurst']

        # Volatility features with division guard
        df_pd['btc_volatility'] = df_pd['btc_close'].pct_change().rolling(24, min_periods=1).std()
        # Safe division - replace zeros with NaN to avoid infinity
        volatility_safe = df_pd['btc_volatility'].replace(0, np.nan)
        df_pd['sentiment_adjusted_by_vol'] = df_pd['sentiment_score'] / volatility_safe

        # Final cleaning
        print(f"[{time.ctime()}] Final cleaning...")
        df_pd.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_pd.ffill(inplace=True)
        df_pd.bfill(inplace=True)

        # Remove any remaining NaN values in critical columns
        for col in ['btc_hurst', 'sentiment_x_hurst', 'btc_volatility']:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].fillna(df_pd[col].median() if col != 'btc_hurst' else 0.5)

        # Save final dataset
        print(f"[{time.ctime()}] Saving final dataset...")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df_pd.to_parquet(output_filepath)

        elapsed_time = time.time() - start_time
        print(f"[{time.ctime()}] SUCCESS: Feature engineering completed in {elapsed_time / 60:.2f} minutes!")
        print(f"Final dataset shape: {df_pd.shape}")
        print(f"Output saved to: {output_filepath}")

    except FileNotFoundError as e:
        print(f"[{time.ctime()}] FILE ERROR: {e}")
    except KeyError as e:
        print(f"[{time.ctime()}] COLUMN ERROR: {e}")
    except Exception as e:
        print(f"[{time.ctime()}] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
