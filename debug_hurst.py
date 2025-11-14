# This is a standalone script for debugging the Hurst Exponent calculation.

import pandas as pd
import numpy as np
import time

def hurst_on_cpu(ts):
    """Calculates the Hurst Exponent on a pandas Series."""
    ts = np.asarray(ts)
    if len(ts) < 100:
        return 0.5

    lags = range(2, 100)
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        std_dev = np.std(diff)
        if std_dev == 0:
            return 0.5
        tau.append(np.sqrt(std_dev))

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

if __name__ == '__main__':
    try:
        print(f"[{time.ctime()}] Starting Hurst Exponent debug...")

        # Load the lean dataset
        df = pd.read_parquet('/content/drive/MyDrive/trading-ai/data/processed/master_dataset_lean.parquet')
        print(f"[{time.ctime()}] Lean dataset loaded.")

        # Calculate the Hurst Exponent
        print(f"[{time.ctime()}] Calculating Hurst Exponent for btc_close...")
        df['btc_hurst'] = df['btc_close'].rolling(window=100).apply(hurst_on_cpu, raw=True)
        print(f"[{time.ctime()}] Hurst Exponent calculated.")

        # Check for null values
        print(f"[{time.ctime()}] Number of null Hurst values: {df['btc_hurst'].isnull().sum()}")

        # Save the result
        output_path = '/content/drive/MyDrive/trading-ai/data/processed/hurst_debug_output.parquet'
        df.to_parquet(output_path)
        print(f"[{time.ctime()}] Debug output saved to {output_path}")

    except Exception as e:
        print(f"[{time.ctime()}] An error occurred: {e}")
        import traceback
        traceback.print_exc()
