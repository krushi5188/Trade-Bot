# This script is for analyzing the internal structure of a LightGBM model.
# It trains a simple model, extracts its decision trees to a DataFrame,
# and prints the structure for analysis. This is a key step in developing
# the Genetic Programming approach.

import os
import sys
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels

def analyze_structure():
    """
    Trains a model and prints its internal tree structure.
    """
    print("--- Starting Model Structure Analysis ---")

    # 1. Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    print(f"Loading data from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)

    # 2. Create Labels and Feature Set
    print("Creating labels and features...")
    df['label'] = get_tri_barrier_labels(df['btc_close'])
    df = df.dropna(subset=['label'])
    label_map = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['label'].map(label_map)

    features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume', 'gld_volume', 'event_name', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]
    X = df[features]
    y = df['label']

    # 3. Train a simple LightGBM Model
    print("Training a simple LightGBM model...")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    # We'll train a model with a small number of trees for easier analysis
    model = lgb.train(params, lgb_train, num_boost_round=5)
    print("Model training complete.")

    # 4. Extract the tree structure to a DataFrame
    print("Extracting tree structure to DataFrame...")
    tree_df = model.trees_to_dataframe()

    # 5. Print the structure for analysis
    print("\\n--- Model Tree Structure (First 20 rows) ---")
    print(tree_df.head(20))
    print("\\n--------------------------------------------")
    print(f"\\nDataFrame Info:")
    tree_df.info()
    print("\\n--- Analysis Complete ---")

if __name__ == '__main__':
    analyze_structure()
