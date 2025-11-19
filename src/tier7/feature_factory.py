
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def create_features_v2():
    """
    Loads the processed 1-minute gold data and engineers an enhanced, V2 set of features.
    """
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'XAUUSD_1m.parquet')
    features_path = os.path.join(project_root, 'data', 'features')

    os.makedirs(features_path, exist_ok=True)

    print("Loading processed 1-minute data...")
    df = pd.read_parquet(processed_data_path)

    print("Starting V2 feature generation...")

    # --- Foundational Features (from V1) ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    windows = [5, 15, 60, 240]
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'VOL_{window}'] = df['Close'].rolling(window=window).std()
        df[f'DIST_MA_{window}'] = df['Close'] - df[f'MA_{window}']
        df[f'ROC_{window}'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)

    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)

    # --- New V2 Features ---

    # 1. Relative Strength Index (RSI)
    for window in windows:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

    # 2. Bollinger Bands
    for window in windows:
        ma = df[f'MA_{window}']
        std = df[f'VOL_{window}']
        df[f'BB_UPPER_{window}'] = ma + (std * 2)
        df[f'BB_LOWER_{window}'] = ma - (std * 2)
        df[f'BB_WIDTH_{window}'] = (df[f'BB_UPPER_{window}'] - df[f'BB_LOWER_{window}']) / ma

    # 3. Tick Volume Velocity
    for window in [5, 15, 60]: # Shorter windows are more relevant for volume spikes
        df[f'VOL_VELOCITY_{window}'] = (df['Volume'] - df['Volume'].shift(window)) / df['Volume'].shift(window)

    print("V2 Feature generation complete.")

    # Replace infinities from calculations with NaN and drop all rows with NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Save the Feature Data ---
    output_filepath = os.path.join(features_path, 'XAUUSD_1m_features_v2.parquet')
    df.to_parquet(output_filepath)

    print(f"Successfully created V2 features and saved to {output_filepath}")
    print("--- V2 Feature Data Info ---")
    print(df.info())
    print("\n--- V2 Feature Data Head ---")
    print(df.head())

if __name__ == "__main__":
    create_features_v2()
