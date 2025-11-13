import ccxt
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_crypto_data(symbol, timeframe, since_days):
    """Fetches historical crypto data using ccxt."""
    print(f"--- Fetching CRYPTO data for {symbol} ---")
    exchange = ccxt.kraken({'rateLimit': 1200, 'enableRateLimit': True})
    since_timestamp = int((datetime.now() - timedelta(days=since_days)).timestamp() * 1000)
    all_ohlcv = []

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=10000)
        if ohlcv:
            all_ohlcv.extend(ohlcv)
            print(f"  Fetched {len(all_ohlcv)} candles for {symbol}")
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('datetime')
    except Exception as e:
        print(f"  An error occurred with ccxt for {symbol}: {e}.")
    return None

def fetch_metals_data_av(symbol, av_api_key):
    """Fetches historical data for a metals ETF proxy using Alpha Vantage."""
    print(f"--- Fetching METALS proxy data for {symbol} using Alpha Vantage ---")
    try:
        ts = TimeSeries(key=av_api_key, output_format='pandas')
        data, _ = ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')
        if not data.empty:
            print(f"  Successfully downloaded {len(data)} candles for {symbol}")
            data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}, inplace=True)
            data.index = pd.to_datetime(data.index)
            return data.iloc[::-1]
    except Exception as e:
        print(f"  An error occurred with Alpha Vantage for {symbol}: {e}")
    return None

def fetch_forex_data_av(from_symbol, to_symbol, av_api_key):
    """Fetches historical DAILY forex data from Alpha Vantage and resamples to hourly."""
    print(f"--- Fetching FOREX data for {from_symbol}/{to_symbol} using Alpha Vantage (Daily) ---")
    try:
        fx = ForeignExchange(key=av_api_key)
        data_dict, _ = fx.get_currency_exchange_daily(from_symbol=from_symbol, to_symbol=to_symbol, outputsize='full')
        if data_dict:
            print(f"  Successfully downloaded {len(data_dict)} daily rates.")
            df = pd.DataFrame.from_dict(data_dict, orient='index')
            df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'}, inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df['volume'] = 0
            # Resample daily data to hourly and fill forward
            df_hourly = df.resample('h').ffill()
            return df_hourly.sort_index()
    except Exception as e:
        print(f"  An error occurred with Alpha Vantage for {from_symbol}/{to_symbol}: {e}")
    return None

def run_unified_pipeline(assets_config, timeframe, since_days, av_api_key, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    # --- Process Crypto ---
    df_crypto = fetch_crypto_data(assets_config['crypto'], timeframe, since_days)
    if df_crypto is not None:
        filename = f"{assets_config['crypto'].replace('/', '-')}_{timeframe}.parquet"
        filepath = os.path.join(data_dir, filename)
        df_crypto.to_parquet(filepath)
        print(f"  -> Saved {assets_config['crypto']} data to {filepath}\n")

    # --- Process Metals ---
    df_metals = fetch_metals_data_av(assets_config['metals'], av_api_key)
    if df_metals is not None:
        filename = f"{assets_config['metals']}_{timeframe}.parquet"
        filepath = os.path.join(data_dir, filename)
        df_metals.to_parquet(filepath)
        print(f"  -> Saved {assets_config['metals']} data to {filepath}\n")

    # --- Process Forex ---
    df_forex = fetch_forex_data_av(assets_config['forex']['from'], assets_config['forex']['to'], av_api_key)
    if df_forex is not None:
        filename = f"{assets_config['forex']['from']}{assets_config['forex']['to']}_{timeframe}.parquet"
        filepath = os.path.join(data_dir, filename)
        df_forex.to_parquet(filepath)
        print(f"  -> Saved {assets_config['forex']['from']}/{assets_config['forex']['to']} data to {filepath}\n")


if __name__ == '__main__':
    ALPHA_VANTAGE_API_KEY = 'Z1CPCJK1GXULHYXU'
    TARGET_ASSETS = {
        'crypto': 'BTC/USD',
        'metals': 'GLD', # Gold ETF as proxy
        'forex': {'from': 'EUR', 'to': 'USD'}
    }
    TIMEFRAME = '1h'
    DAYS_OF_HISTORY_CRYPTO = 365 * 2 # 2 years for crypto
    DATA_DIRECTORY = 'data/raw/market'

    run_unified_pipeline(TARGET_ASSETS, TIMEFRAME, DAYS_OF_HISTORY_CRYPTO, ALPHA_VANTAGE_API_KEY, DATA_DIRECTORY)
