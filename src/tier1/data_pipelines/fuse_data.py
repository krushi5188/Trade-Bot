# This script formalizes the data fusion process from the Tier 1 notebook.

import pandas as pd
import os

def fuse_all_data_sources():
    """
    Loads all raw data sources, standardizes their timezones to UTC,
    fuses them, and saves the master dataset.
    """
    print("--- Starting Data Fusion Process ---")

    # --- Define Paths ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data/raw')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')

    MARKET_DATA_DIR = os.path.join(RAW_DIR, 'market')
    SENTIMENT_DATA_PATH = os.path.join(RAW_DIR, 'sentiment', 'news_sentiment.parquet')
    CALENDAR_DATA_PATH = os.path.join(RAW_DIR, 'calendar', 'economic_calendar.parquet')

    ASSETS = {
        'btc': 'BTC-USD_1h.parquet',
        'gld': 'GLD_1h.parquet',
        'eur': 'EURUSD_1h.parquet'
    }

    # --- 1. Load and Merge Market Data ---
    print("Loading and merging market data...")
    df_market = None
    for prefix, filename in ASSETS.items():
        filepath = os.path.join(MARKET_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: File not found: {filepath}. Skipping.")
            continue
        df_asset = pd.read_parquet(filepath)
        df_asset = df_asset.add_prefix(f'{prefix}_')

        # --- FIX: Timezone Standardization ---
        if df_asset.index.tz is None:
            df_asset = df_asset.tz_localize('UTC')
        else:
            df_asset = df_asset.tz_convert('UTC')

        if df_market is None:
            df_market = df_asset
        else:
            df_market = pd.merge(df_market, df_asset, left_index=True, right_index=True, how='outer')

    if df_market is None:
        print("FATAL: No market data found.")
        return

    print("Market data merged.")

    # --- 2. Load and Merge Sentiment Data ---
    print("Loading and merging sentiment data...")
    df_master = df_market
    if not os.path.exists(SENTIMENT_DATA_PATH):
        print(f"  WARNING: File not found: {SENTIMENT_DATA_PATH}.")
        df_master['sentiment_score'] = 0
    else:
        df_sentiment = pd.read_parquet(SENTIMENT_DATA_PATH)
        df_sentiment = df_sentiment.tz_convert('UTC') # Sentiment data is tz-aware
        df_sentiment_hourly = df_sentiment['sentiment_score'].resample('h').mean()
        df_master = pd.merge(df_master, df_sentiment_hourly, left_index=True, right_index=True, how='left')

    # --- 3. Load and Merge Calendar Data ---
    print("Loading and merging calendar data...")
    if not os.path.exists(CALENDAR_DATA_PATH):
        print(f"  WARNING: File not found: {CALENDAR_DATA_PATH}.")
        df_master['event_name'] = None
    else:
        df_calendar = pd.read_parquet(CALENDAR_DATA_PATH)
        df_calendar = df_calendar.tz_localize('UTC') # Calendar data is tz-naive
        df_master = pd.merge(df_master, df_calendar[['event_name']], left_index=True, right_index=True, how='left')

    # --- 4. Clean and Finalize ---
    print("Cleaning and finalizing...")
    df_master['sentiment_score'].fillna(method='ffill', inplace=True)
    df_master['sentiment_score'].fillna(0, inplace=True)

    df_master.fillna(method='ffill', inplace=True)
    df_master.fillna(method='bfill', inplace=True)

    # --- 5. Save ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    master_filepath = os.path.join(PROCESSED_DIR, 'master_dataset.parquet')
    df_master.to_parquet(master_filepath)

    print(f"\\nSuccessfully created master dataset at: {master_filepath}")
    print("\\n--- Master Dataset Info ---")
    df_master.info()

if __name__ == '__main__':
    fuse_all_data_sources()
