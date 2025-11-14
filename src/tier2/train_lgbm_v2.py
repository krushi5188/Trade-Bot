# This script trains a LightGBM model on the V2 feature set.

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys

# Add the project root to the Python path to enable imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels

def train_lightgbm_on_v2_features():
    """
    Loads the V2 feature dataset, trains a LightGBM model,
    and saves it to disk.
    """
    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'src/tier2/modeling')
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_v2_model.txt')

    # 2. Load Data
    print(f"[{time.ctime()}] Loading V2 feature dataset from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)
    print(f"[{time.ctime()}] V2 dataset loaded. Shape: {df.shape}")

    # 3. Create Labels and Define Feature Set
    print(f"[{time.ctime()}] Creating tri-barrier labels...")
    df['label'] = get_tri_barrier_labels(df['btc_close'])
    df = df.dropna(subset=['label']) # Drop rows where labels couldn't be generated

    # Exclude non-feature columns
    features_to_exclude = [col for col in df.columns if '_close' in col or '_volume' in col or 'event_name' in col or 'label' in col]
    features = [c for c in df.columns if c not in features_to_exclude]

    X = df[features]
    y = df['label']
    print(f"[{time.ctime()}] Feature set defined with {len(features)} features.")

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    print(f"[{time.ctime()}] Data split into training and testing sets.")

    # 5. Train LightGBM Model
    print(f"[{time.ctime()}] Training LightGBM model...")
    lgb_train = lgb.Dataset(X_train, y_train)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    model = lgb.train(params, lgb_train, num_boost_round=100)
    print(f"[{time.ctime()}] Model training complete.")

    # 6. Evaluate Model
    print(f"[{time.ctime()}] Evaluating model performance...")
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [list(row).index(max(row)) for row in y_pred_proba]
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\\n--- Model V2 Performance ---")
    print(f"Accuracy on test set: {accuracy:.4f}")
    print("--------------------------\\n")

    # 7. Save Model
    print(f"[{time.ctime()}] Saving trained model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    print(f"[{time.ctime()}] Model saved successfully.")

if __name__ == '__main__':
    import time
    train_lightgbm_on_v2_features()
