# This is a standalone diagnostic script to definitively find the bug in the Hurst Exponent calculation.

import pandas as pd
import numpy as np
import time
import os

def hurst_diagnostic(ts):
    """
    A diagnostic version of the Hurst function to find the root cause of the error.
    """
    if len(ts) < 100:
        return -999.0

    try:
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
            return -998.0

        poly = np.polyfit(np.log(range(2, len(tau) + 2)), np.log(tau), 1)
        result = poly[0] * 2.0
        return result
    except Exception:
        return -997.0

if __name__ == '__main__':
    try:
        print(f"[{time.ctime()}] Starting Hurst Exponent Diagnosis...")

        # FIX: Use a relative path to avoid FileNotFoundError
        # This assumes the script is run from the root of the project directory.
        lean_filepath = 'data/processed/master_dataset_lean.parquet'

        if not os.path.exists(lean_filepath):
            print(f"[{time.ctime()}] ERROR: The lean dataset was not found at {lean_filepath}")
            print(f"[{time.ctime()}] Please run 'src/tier2/create_lean_dataset.py' first.")
            exit()

        df = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Lean dataset loaded.")

        print(f"[{time.ctime()}] Diagnosing Hurst Exponent for 'eur_close'...")

        results = df['eur_close'].rolling(window=100).apply(hurst_diagnostic, raw=True)

        print(f"[{time.ctime()}] Diagnosis function finished.")

        print("\\n" + "="*30)
        print("      DIAGNOSIS REPORT")
        print("="*30)
        print(f"Total results calculated: {len(results)}")
        print(f"Number of valid Hurst values (not special codes): {((results > -997) & (results != 0.5)).sum()}")
        print("-" * 30)
        print(f"Short windows (< 100 points), code -999.0: {(results == -999.0).sum()}")
        print(f"Flat series (std dev = 0), code 0.5:         {(results == 0.5).sum()}")
        print(f"Not enough data for regression, code -998.0: {(results == -998.0).sum()}")
        print(f"Other math errors (e.g., log), code -997.0:  {(results == -997.0).sum()}")
        print("-" * 30)

        df['eur_hurst_debug'] = results

        if 'eur_hurst_debug' in df.columns:
            print("SUCCESS: The debug column was successfully created in the DataFrame.")
            if df['eur_hurst_debug'].isnull().all():
                print("WARNING: The resulting column contains all NULL values, which is why pandas may have dropped it before.")
            else:
                 print("The resulting column seems to contain valid data.")
        else:
            print("FAILURE: The debug column was NOT created. This is the root of the KeyError.")
        print("="*30)

    except Exception as e:
        print(f"[{time.ctime()}] A critical error occurred during the diagnosis script itself: {e}")
        import traceback
        traceback.print_exc()
