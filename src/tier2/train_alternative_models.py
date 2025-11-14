# This script will train and evaluate alternative model architectures,
# starting with LightGBM, to see if we can improve upon the XGBoost baseline.

import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import sys
from sklearn.metrics import classification_report

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tier1.modeling.train_xgboost import get_tri_barrier_labels

def train_and_evaluate_lgbm(feature_path, model_dir):
    """
    Loads a stable feature set, trains a LightGBM model, and evaluates its performance.
    """
    print("--- Training and Evaluating LightGBM Model on Stable V1 Features ---")

    # 1. Load Data
    print(f"Loading stable feature dataset from: {feature_path}")
    df = pd.read_parquet(feature_path)

    # 2. Create Labels
    print("Creating Tri-Barrier Labels for BTC/USD...")
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels

    df.dropna(inplace=True) # General cleaning
    df = df.iloc[:-24]

    # 3. Define Features (X) and Labels (y)
    # Using the original, stable feature set from Tier 1
    features_to_exclude = ['btc_close', 'btc_volume', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]
    X = df[features]
    y = df['label'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    # 4. Chronological Train-Test Split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 5. Train LightGBM Model
    lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_jobs=-1)
    lgbm.fit(X_train, y_train)

    # 6. Evaluate Model
    y_pred = lgbm.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)'], zero_division=0)
    print("\\n--- LightGBM Model Evaluation (on V1 Features) ---")
    print(report)

    # 7. Save the Model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lgbm_v1_stable.txt')
    lgbm.booster_.save_model(model_path)
    print(f"\\nLightGBM model successfully saved to {model_path}")

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # --- FIX: Reverting to the stable V1 feature set to ensure a working pipeline ---
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_03_final.parquet')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'src/tier2/modeling')

    train_and_evaluate_lgbm(FEATURE_PATH, MODEL_DIR)
