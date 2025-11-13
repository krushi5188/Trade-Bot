# This script will perform hyperparameter tuning for the XGBoost model
# using GridSearchCV with a time-series split to find the optimal parameters.

import pandas as pd
import xgboost as xgb
import os
import sys
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tier1.modeling.train_xgboost import get_tri_barrier_labels

def tune_model_hyperparameters():
    """
    Performs a grid search to find the best hyperparameters for the XGBoost model.
    """
    # 1. Define Paths and Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_03_final.parquet')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'src/tier1/modeling')

    print("Loading and preparing data for tuning...")
    df = pd.read_parquet(FEATURE_PATH)

    # 2. Prepare Training Data (using the full 80% training set)
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels
    df = df.iloc[:-24]

    features_to_exclude = ['btc_close', 'btc_volume', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]

    split_index = int(len(df) * 0.8)
    X_train = df[features].iloc[:split_index]
    y_train_mapped = df['label'].iloc[:split_index].map({-1: 0, 0: 1, 1: 2})

    print(f"Tuning on {len(X_train)} samples.")

    # 3. Define the Parameter Grid
    # We will test a range of common XGBoost parameters.
    # This is a starting point; a real-world search would be more exhaustive.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # 4. Set up GridSearchCV
    # We use TimeSeriesSplit for cross-validation to respect the data's chronological order.
    tscv = TimeSeriesSplit(n_splits=5)

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        verbose=2,
        n_jobs=-1 # Use all available CPU cores
    )

    # 5. Run the Search
    print("\\nStarting hyperparameter grid search...")
    grid_search.fit(X_train, y_train_mapped)

    # 6. Report Best Parameters and Score
    print("\\n--- Hyperparameter Tuning Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # 7. Save the Best Model
    best_model = grid_search.best_estimator_
    tuned_model_path = os.path.join(MODEL_DIR, 'xgboost_tuned_v1.json')
    best_model.save_model(tuned_model_path)
    print(f"\\nTuned model successfully saved to {tuned_model_path}")

if __name__ == '__main__':
    tune_model_hyperparameters()
