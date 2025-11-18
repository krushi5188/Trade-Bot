# This script will download high-frequency (1-minute) historical data for Gold futures.

import os
import sys
import pandas as pd
import yfinance as yf
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def download_hf_gold_data(start_date, end_date, file_path):
    """
    Downloads 1-minute historical data for Gold futures (GC=F) in 7-day chunks
    and saves it to a Parquet file.
    """
    print(f"[{time.ctime()}] --- Starting High-Frequency Gold Data Download ---")

    ticker = "GC=F"
    all_data = []

    # Create a date range to loop through in 7-day increments
    date_range = pd.date_range(start=start_date, end=end_date, freq='7D')

    for i in range(len(date_range) - 1):
        chunk_start = date_range[i]
        chunk_end = date_range[i+1]

        print(f"Fetching data from {chunk_start.date()} to {chunk_end.date()}...")

        try:
            # Download 1-minute data for the chunk
            data = yf.download(tickers=ticker, start=chunk_start, end=chunk_end, interval="1m")

            if not data.empty:
                all_data.append(data)

            # Be respectful of the API
            time.sleep(1)

        except Exception as e:
            print(f"Could not download data for {chunk_start.date()} to {chunk_end.date()}: {e}")

    if not all_data:
        print("No data was downloaded. Exiting.")
        return

    # Concatenate all chunks and save
    df = pd.concat(all_data)
    df = df.drop_duplicates()
    df = df.sort_index()

    # Save to Parquet
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path)

    print(f"\n[{time.ctime()}] --- Download Complete ---")
    print(f"Data saved to: {file_path}")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

def main():
    """
    Main function to run the download script.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    FILE_PATH = os.path.join(PROJECT_ROOT, 'data/raw/market/GLD_1m.parquet')

    # Let's download data for the last 2 years for this test
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(years=2)

    download_hf_gold_data(start_date, end_date, FILE_PATH)

if __name__ == '__main__':
    main()
