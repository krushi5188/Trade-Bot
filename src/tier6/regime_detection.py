import os
import sys
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
import time

# Suppress ConvergenceWarning from hmmlearn, which is common during HMM training
warnings.filterwarnings("ignore", category=UserWarning)

# Add the project root to the Python path to enable imports from other tiers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def identify_market_regimes(n_regimes=3, feature_path='data/processed/features_v2_final.parquet', output_path='data/processed/features_v3_with_regimes.parquet'):
    """
    Loads feature data, identifies market regimes using a Gaussian Hidden Markov Model,
    and saves the data with a new 'market_regime' column.

    Args:
        n_regimes (int): The number of distinct market regimes to identify.
        feature_path (str): The path to the input feature parquet file.
        output_path (str): The path to save the output parquet file with regime labels.
    """
    print(f"[{time.ctime()}] --- Starting Market Regime Identification for {n_regimes} regimes ---")

    # 1. Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_feature_path = os.path.join(PROJECT_ROOT, feature_path)
    df = pd.read_parquet(full_feature_path)
    print(f"[{time.ctime()}] Loaded data with shape: {df.shape}")

    # 2. Select Features for HMM
    # We will use log returns and a rolling volatility measure as the primary indicators
    # of the underlying market state. These are classic choices for regime detection.
    df['log_returns'] = np.log(df['btc_close'] / df['btc_close'].shift(1))

    # We assume a volatility feature, e.g., 'btc_volatility', is already engineered.
    # If not, this will need to be adjusted.
    hmm_features = ['log_returns', 'btc_volatility']
    df_hmm = df[hmm_features].copy()
    df_hmm.dropna(inplace=True)
    X = df_hmm.values

    # 3. Initialize and Train the Gaussian HMM
    print(f"[{time.ctime()}] Training Gaussian HMM...")
    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    print(f"[{time.ctime()}] HMM training complete.")

    # 4. Predict the hidden states (regimes)
    hidden_states = model.predict(X)

    # 5. Add Regime Labels to the Original DataFrame
    # We align the predicted regimes back to the original dataframe, accounting for the rows dropped by dropna()
    df.loc[df_hmm.index, 'market_regime'] = hidden_states

    # Forward fill any NaNs at the beginning that were not part of the HMM input
    df['market_regime'].ffill(inplace=True)
    df.dropna(subset=['market_regime'], inplace=True) # Drop any remaining NaNs at the start
    df['market_regime'] = df['market_regime'].astype(int)

    print(f"[{time.ctime()}] Regime counts:")
    print(df['market_regime'].value_counts())

    # 6. Save the augmented data
    full_output_path = os.path.join(PROJECT_ROOT, output_path)
    df.to_parquet(full_output_path)
    print(f"[{time.ctime()}] Successfully saved data with regime labels to: {full_output_path}")

    return df

if __name__ == '__main__':
    identify_market_regimes()
