# This script analyzes the class imbalance in the V2 dataset and model predictions.

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
import sys
import time

# Add the project root to the Python path to enable imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels

def diagnose_class_imbalance():
    """
    Analyzes and prints the distribution of true labels and model predictions
    to diagnose class imbalance.
    """
    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/lgbm_v2_model.txt')

    # 2. Load Data
    print(f"[{time.ctime()}] Loading V2 feature dataset from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)
    print(f"[{time.ctime()}] V2 dataset loaded.")

    # 3. Create Labels and Define Feature Set
    print(f"[{time.ctime()}] Creating tri-barrier labels...")
    df['label'] = get_tri_barrier_labels(df['btc_close'])
    df = df.dropna(subset=['label']) # Drop rows where labels couldn't be generated

    features_to_exclude = [col for col in df.columns if '_close' in col or '_volume' in col or 'event_name' in col or 'label' in col]
    features = [c for c in df.columns if c not in features_to_exclude]

    X = df[features]
    y = df['label']

    # 4. Split Data (using the same split as the training script)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # 5. Analyze True Label Distribution
    print("\\n--- True Label Distribution (Test Set) ---")
    print(y_test.value_counts(normalize=True))
    print("------------------------------------------\\n")

    # 6. Load Model and Make Predictions
    print(f"[{time.ctime()}] Loading model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"[{time.ctime()}] Model loaded.")

    print(f"[{time.ctime()}] Making predictions on the test set...")
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [list(row).index(max(row)) for row in y_pred_proba]
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    print(f"[{time.ctime()}] Predictions made.")

    # 7. Analyze Predicted Label Distribution
    print("\\n--- Predicted Label Distribution (Test Set) ---")
    print(y_pred_series.value_counts(normalize=True))
    print("---------------------------------------------")

if __name__ == '__main__':
    diagnose_class_imbalance()
