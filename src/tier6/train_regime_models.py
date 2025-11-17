import os
import sys
import pandas as pd
import lightgbm as lgb
import joblib
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Add the project root to the Python path to enable imports from other tiers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tier1.modeling.train_xgboost import get_tri_barrier_labels # Re-using the label generation

def train_and_save_regime_models(feature_path='data/processed/features_v3_with_regimes.parquet', model_dir='src/tier6/regime_models'):
    """
    Loads data with regime labels, trains a separate LightGBM model for each regime,
    and saves each model to the specified directory.
    """
    print(f"[{time.ctime()}] --- Starting Regime-Specific Model Training ---")

    # 1. Setup Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_feature_path = os.path.join(PROJECT_ROOT, feature_path)
    full_model_dir = os.path.join(PROJECT_ROOT, model_dir)
    os.makedirs(full_model_dir, exist_ok=True)
    print(f"[{time.ctime()}] Models will be saved to: {full_model_dir}")

    # 2. Load Data and Generate Labels
    df = pd.read_parquet(full_feature_path)
    df['outcome'] = get_tri_barrier_labels(df['btc_close'])
    df.dropna(subset=['outcome'], inplace=True)
    df['outcome'] = df['outcome'].astype(int)
    print(f"[{time.ctime()}] Loaded data and generated labels. Shape: {df.shape}")

    # 3. Identify Unique Regimes
    regimes = sorted(df['market_regime'].unique())
    print(f"[{time.ctime()}] Found {len(regimes)} regimes to train on: {regimes}")

    # 4. Define Features
    features_to_exclude = [
        'btc_close', 'eur_close', 'gld_close', 'event_name', 'outcome',
        'market_regime', 'log_returns' # Exclude columns used for regime ID or labeling
    ]
    features = [c for c in df.columns if c not in features_to_exclude and not c.endswith('_volume')]

    # 5. Loop, Train, and Save a Model for Each Regime
    for regime in regimes:
        print(f"\n[{time.ctime()}] --- Training model for Regime {regime} ---")

        # Filter data for the current regime
        df_regime = df[df['market_regime'] == regime].copy()
        print(f"[{time.ctime()}] Data shape for regime {regime}: {df_regime.shape}")

        if len(df_regime) < 100: # Skip if there's not enough data to train
            print(f"[{time.ctime()}] Skipping regime {regime} due to insufficient data.")
            continue

        X = df_regime[features]
        y = df_regime['outcome']

        # Map labels for LightGBM (0: sell, 1: hold, 2: buy)
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        # Handle class imbalance
        classes = np.unique(y_mapped)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_mapped)
        class_weight_dict = dict(zip(classes, weights))

        # Define and Train Model
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'seed': 42,
            'verbose': -1,
            'class_weight': class_weight_dict
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y_mapped)

        # Save the trained model
        model_path = os.path.join(full_model_dir, f"regime_{regime}_model.txt")
        model.booster_.save_model(model_path)
        print(f"[{time.ctime()}] Saved model for regime {regime} to {model_path}")

    print(f"\n[{time.ctime()}] --- All Regime-Specific Models Trained Successfully ---")

if __name__ == '__main__':
    train_and_save_regime_models()
