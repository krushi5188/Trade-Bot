
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def create_external_context_features():
    """
    Merges 1-minute market data with hourly external data (sentiment, calendar)
    to create context-aware features.
    """
    # --- File Paths ---
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'XAUUSD_1m.parquet')
    master_dataset_path = os.path.join(project_root, 'data/processed/master_dataset.parquet')
    features_path = os.path.join(project_root, 'data', 'features')
    os.makedirs(features_path, exist_ok=True)

    # --- Load Data ---
    print("Loading 1-minute gold data...")
    df_1m = pd.read_parquet(processed_data_path)
    df_1m = df_1m[df_1m.index.year >= 2020].copy() # Use recent data

    print("Loading hourly master dataset with external data...")
    df_master = pd.read_parquet(master_dataset_path)

    # Correct column names based on inspection
    external_cols = [
        'sentiment_score',
        'event_name'
    ]
    df_external = df_master[external_cols]

    # --- Align Data and Engineer Features ---
    print("Aligning hourly external data to 1-minute frequency...")
    # Forward-fill the hourly data to the 1-minute index
    df_aligned = df_external.reindex(df_1m.index, method='ffill')

    print("Engineering external context features...")

    # 1. Is Event Window Feature
    # Use the presence of an event name to signify an event
    df_aligned['is_event'] = (~df_aligned['event_name'].isnull()).astype(int)
    # Use a rolling window to create the "window" effect (+/- 1 hour)
    df_aligned['is_event_window'] = df_aligned['is_event'].rolling(window=120, min_periods=1).max().fillna(0)

    # 2. Lagged Sentiment Features
    for lag in [5, 15, 60]: # 5m, 15m, 1hr lags
        df_aligned[f'sentiment_lag_{lag}'] = df_aligned['sentiment_score'].shift(lag)

    print("External context feature generation complete.")

    # --- Save the Feature Data ---
    feature_cols = [
        'sentiment_score', 'is_event_window',
        'sentiment_lag_5', 'sentiment_lag_15', 'sentiment_lag_60'
    ]
    features_df = df_aligned[feature_cols].dropna()

    output_filepath = os.path.join(features_path, 'XAUUSD_1m_features_external.parquet')
    features_df.to_parquet(output_filepath)

    print(f"Successfully created external context features and saved to {output_filepath}")
    print("--- Feature Data Info ---")
    print(features_df.info())
    print("\n--- Feature Data Head ---")
    print(features_df.head())

if __name__ == "__main__":
    create_external_context_features()
