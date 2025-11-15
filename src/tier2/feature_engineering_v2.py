# FINAL DIAGNOSTIC SCRIPT - to find the root cause of the KeyError

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import time

def hurst_on_cpu(ts: np.ndarray) -> float:
    """The proven, correct, and reliable CPU-based ROLLING Hurst Exponent calculation."""
    ts = np.asarray(ts)
    if len(ts) < 100:
        return 0.5
    lags = range(2, 100)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        if len(diff) == 0: continue
        std_dev = np.std(diff)
        if std_dev <= 0: return 0.5
        tau.append(np.sqrt(std_dev))
    if len(tau) < 2: return 0.5
    poly = np.polyfit(np.log(range(2, len(tau) + 2)), np.log(tau), 1)
    return poly[0] * 2.0

def apply_kalman_filter_safe(series_pd: pd.Series) -> pd.Series:
    """Safe Kalman filter with proper error handling."""
    try:
        resampled_series = series_pd.resample('h').ffill()
        if len(resampled_series.dropna()) < 10: return resampled_series
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
        print(f"[{time.ctime()}] Starting FINAL DIAGNOSIS...")

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')

        df_pd = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Dataset loaded.")

        # Kalman Filters
        for col in ['btc_close', 'eur_close', 'gld_close']:
            print(f"[{time.ctime()}] Processing Kalman Filter for {col}...")
            df_pd[f'{col}_kalman'] = apply_kalman_filter_safe(df_pd[col])

        # Hurst Exponent
        print(f"[{time.ctime()}] Processing Hurst Exponent...")
        for col in ['btc_close', 'eur_close', 'gld_close']:
            df_pd[f'{col}_hurst'] = df_pd[col].rolling(window=100, min_periods=100).apply(hurst_on_cpu, raw=True)

        # --- DIAGNOSTIC BLOCK ---
        print("\\n" + "="*50)
        print("          ENTERING FINAL DIAGNOSTIC BLOCK")
        print("="*50)
        print(f"[{time.ctime()}] This is the state of the program RIGHT BEFORE the error occurs.")
        print(f"[{time.ctime()}] The next step is to create the interaction features.")

        print("\\nDataFrame Info:")
        df_pd.info()

        print("\\nDataFrame Columns:")
        print(df_pd.columns)

        if 'btc_hurst' in df_pd.columns:
            print("\\n'btc_hurst' column EXISTS.")
            print("Number of null values in 'btc_hurst':", df_pd['btc_hurst'].isnull().sum())
        else:
            print("\\nCRITICAL FAILURE: 'btc_hurst' column DOES NOT EXIST.")

        print("="*50)
        print("          NOW ATTEMPTING THE OPERATION THAT FAILS")
        print("="*50 + "\\n")
        # --- END DIAGNOSTIC BLOCK ---

        # Interaction Features
        print(f"[{time.ctime()}] Processing interaction features...")
        df_pd['sentiment_score'] = pd.to_numeric(df_pd['sentiment_score'], errors='coerce').fillna(0).astype('float32')
        df_pd['btc_hurst'] = pd.to_numeric(df_pd['btc_hurst'], errors='coerce').fillna(0.5).astype('float32') # This is the line that fails
        df_pd['sentiment_x_hurst'] = df_pd['sentiment_score'] * df_pd['btc_hurst']

        # ... the rest of the script would go here ...

    except Exception as e:
        print(f"[{time.ctime()}] AN ERROR OCCURRED. This is the traceback:")
        import traceback
        traceback.print_exc()
