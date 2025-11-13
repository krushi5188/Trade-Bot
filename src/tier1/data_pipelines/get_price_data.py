import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_historical_data(symbol, timeframe, since_days, data_dir):
    """
    Fetches historical OHLCV data for a given symbol and timeframe.

    Args:
        symbol (str): The trading symbol (e.g., 'BTC/USDT').
        timeframe (str): The timeframe to fetch (e.g., '1m', '5m', '1h').
        since_days (int): The number of days of historical data to fetch.
        data_dir (str): The directory to save the data in.
    """
    # 1. Initialize Exchange - Switched to Kraken to avoid geo-restrictions
    exchange = ccxt.kraken({
        'rateLimit': 1200,  # Set a safe rate limit
        'enableRateLimit': True
    })

    # 2. Calculate Start Time
    since_timestamp = int((datetime.now() - timedelta(days=since_days)).timestamp() * 1000)

    # 3. Create Data Directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{symbol.replace('/', '')}_{timeframe}.parquet"
    filepath = os.path.join(data_dir, filename)

    # 4. Fetch Data in Paginated Chunks
    all_ohlcv = []
    print(f"Starting data download for {symbol} on {timeframe} timeframe...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=1000)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            since_timestamp = ohlcv[-1][0] + 1 # Move to the next candle

            # Provide progress update
            first_date = datetime.fromtimestamp(all_ohlcv[0][0] / 1000)
            last_date = datetime.fromtimestamp(all_ohlcv[-1][0] / 1000)
            print(f"Fetched {len(all_ohlcv)} candles from {first_date} to {last_date}")

        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5) # Wait 5 seconds before retrying

    print("Download complete. Converting to DataFrame and saving...")

    # 5. Convert to Pandas DataFrame and Save
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)

        # Remove duplicates just in case of API overlap
        df = df[~df.index.duplicated(keep='first')]

        df.to_parquet(filepath)
        print(f"Data successfully saved to {filepath}")
    else:
        print("No data was fetched.")

if __name__ == '__main__':
    # --- Configuration ---
    TARGET_SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1m'
    DAYS_OF_DATA = 365 # Fetch one year of 1-minute data
    DATA_DIRECTORY = 'data/raw/market'

    fetch_historical_data(TARGET_SYMBOL, TIMEFRAME, DAYS_OF_DATA, DATA_DIRECTORY)
