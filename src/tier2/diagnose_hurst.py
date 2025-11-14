# This is a standalone diagnostic script to definitively find the bug in the Hurst Exponent calculation.

import pandas as pd
import numpy as np
import time

def hurst_diagnostic(ts):
    """
    A diagnostic version of the Hurst function to find the root cause of the error.
    """
    # This function is designed to be used with .apply(raw=True), so ts is a NumPy array.
    if len(ts) < 100:
        return -999.0  # Special code for short windows

    try:
        lags = range(2, 100)
        tau = []
        for lag in lags:
            diff = ts[lag:] - ts[:-lag]
            # This check is critical
            if len(diff) == 0:
                continue
            std_dev = np.std(diff)
            if std_dev <= 0:
                # If std is 0 or negative (which shouldn't happen), we can't take the log.
                # Return a special code to signify a flat series.
                return 0.5
            tau.append(np.sqrt(std_dev))

        if len(tau) < 2:
            # Not enough data points to perform a regression
            return -998.0

        # Perform the log-log regression
        poly = np.polyfit(np.log(range(2, len(tau) + 2)), np.log(tau), 1)
        result = poly[0] * 2.0
        return result
    except Exception:
        # If any other mathematical error occurs (e.g., in np.log), return an error code.
        return -997.0

if __name__ == '__main__':
    try:
        print(f"[{time.ctime()}] Starting Hurst Exponent Diagnosis...")

        # Use the full path as expected in the Colab environment
        lean_filepath = '/content/drive/MyDrive/trading-ai/data/processed/master_dataset_lean.parquet'
        df = pd.read_parquet(lean_filepath)
        print(f"[{time.ctime()}] Lean dataset loaded.")

        # Focus on the 'eur_close' column which seemed to take the longest time.
        print(f"[{time.ctime()}] Diagnosing Hurst Exponent for 'eur_close'...")

        # This will be slow, but it is necessary.
        results = df['eur_close'].rolling(window=100).apply(hurst_diagnostic, raw=True)

        print(f"[{time.ctime()}] Diagnosis function finished.")

        # --- Detailed Analysis of Results ---
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

        # Final check to see if the column can be created.
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
