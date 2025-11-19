
import pandas as pd
import glob
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def process_tick_data():
    """
    Reads all XAUUSD tick data CSVs, processes them, and saves them as a single Parquet file.
    """
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'market')
    processed_data_path = os.path.join(project_root, 'data', 'processed')

    # Create the processed data directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)

    # Get a sorted list of all the XAUUSD csv files
    all_files = sorted(glob.glob(os.path.join(raw_data_path, 'XAUUSD_*.csv')))

    if not all_files:
        print("No XAUUSD CSV files found in data/raw/market/")
        return

    print(f"Found {len(all_files)} files to process.")

    # Define column names as the files don't have headers
    column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Read and concatenate all files
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, header=None, names=column_names)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        print("Could not read any data from the CSV files.")
        return

    full_df = pd.concat(df_list, ignore_index=True)

    # --- Data Cleaning and Processing ---

    # 1. Combine Date and Time into a single datetime object
    # Using format='mixed' can be slow, but robust for different date formats if they exist.
    # Given the file format, we can be more specific.
    full_df['Timestamp'] = pd.to_datetime(full_df['Date'] + ' ' + full_df['Time'], format='%Y.%m.%d %H:%M')

    # 2. Set the new Timestamp as the index
    full_df = full_df.set_index('Timestamp')

    # 3. Drop the old Date and Time columns
    full_df = full_df.drop(['Date', 'Time'], axis=1)

    # 4. Ensure the index is timezone-aware (UTC)
    # The data is likely from MT5/MT4, which is often in a specific broker timezone.
    # Assuming UTC for standardization is a safe first step.
    full_df = full_df.tz_localize('UTC')

    # 5. Sort the index to ensure chronological order
    full_df = full_df.sort_index()

    # 6. Drop duplicate timestamps if any
    full_df = full_df[~full_df.index.duplicated(keep='first')]

    # --- Save the Processed Data ---
    output_filepath = os.path.join(processed_data_path, 'XAUUSD_1m.parquet')
    full_df.to_parquet(output_filepath)

    print(f"Successfully processed and saved data to {output_filepath}")
    print("--- Data Info ---")
    print(full_df.info())
    print("\n--- Data Head ---")
    print(full_df.head())
    print("\n--- Data Tail ---")
    print(full_df.tail())


if __name__ == "__main__":
    process_tick_data()
