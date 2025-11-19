
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def create_market_internal_features():
    """
    Loads the 1-minute gold data and engineers features based on the
    interaction between price and volume.
    """
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'XAUUSD_1m.parquet')
    features_path = os.path.join(project_root, 'data', 'features')
    os.makedirs(features_path, exist_ok=True)

    print("Loading processed 1-minute data...")
    df = pd.read_parquet(processed_data_path)
    # Use data from 2020 onwards for efficiency and relevance
    df = df[df.index.year >= 2020].copy()

    print("Starting market internal feature generation...")

    # 1. Volume-Weighted Average Price (VWAP)
    # VWAP is typically calculated on a daily basis.
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cumulative_Volume'] = df.groupby(df.index.date)['Volume'].cumsum()
    df['Cumulative_PV'] = df.groupby(df.index.date)['Typical_Price'].cumsum()
    df['VWAP'] = df['Cumulative_PV'] / df['Cumulative_Volume']
    df['Dist_from_VWAP'] = df['Close'] - df['VWAP']

    # 2. On-Balance Volume (OBV)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = obv

    # 3. Volume Spikes/Anomalies
    # Calculate rolling average of volume to find unusual spikes
    volume_ma_short = df['Volume'].rolling(window=60).mean() # 1-hour avg
    volume_ma_long = df['Volume'].rolling(window=240).mean() # 4-hour avg
    df['Volume_Spike_Ratio'] = volume_ma_short / volume_ma_long

    print("Market internal feature generation complete.")

    # Clean up intermediate columns and NaNs
    df = df.drop(['Typical_Price', 'Cumulative_Volume', 'Cumulative_PV'], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Select only the new feature columns to save
    feature_cols = [
        'VWAP', 'Dist_from_VWAP', 'OBV', 'Volume_Spike_Ratio'
    ]
    features_df = df[feature_cols]

    # --- Save the Feature Data ---
    output_filepath = os.path.join(features_path, 'XAUUSD_1m_features_internals.parquet')
    features_df.to_parquet(output_filepath)

    print(f"Successfully created market internal features and saved to {output_filepath}")
    print("--- Feature Data Info ---")
    print(features_df.info())
    print("\n--- Feature Data Head ---")
    print(features_df.head())

if __name__ == "__main__":
    create_market_internal_features()
