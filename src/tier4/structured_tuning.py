# This script implements a structured and robust hyperparameter tuning process
# using Optuna and TimeSeriesSplit cross-validation. This is the first
# component of the new professional automated retraining and validation pipeline.

import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Add the project root to the Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels
from src.tier2.backtester import VectorizedBacktester

# --- Global Variables ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
X = None
y = None

def objective(trial):
    """
    The objective function for the Optuna study.
    This function trains and evaluates a model using parameters suggested by Optuna.
    """
    global X, y

    # 1. Define the hyperparameter search space
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    # 2. Use TimeSeriesSplit for robust cross-validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sharpe_ratios = []

    print(f"\\nTrial {trial.number}: Starting cross-validation with {n_splits} splits...")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 3. Train the model for the current fold
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Evaluate on the validation fold using the backtester
        y_pred_proba = model.predict_proba(X_val)
        predictions_mapped = np.argmax(y_pred_proba, axis=1)

        signal_map = {0: -1, 1: 0, 2: 1}
        signals = pd.Series(predictions_mapped, index=X_val.index).map(signal_map)

        # We need the price data for the validation set
        price_data_val = pd.read_parquet(FEATURE_PATH, columns=['btc_close']).loc[X_val.index]

        backtester = VectorizedBacktester(
            price_data=price_data_val['btc_close'],
            signals=signals,
            initial_capital=100000
        )

        metrics = backtester.get_performance_metrics()
        sharpe_ratio = metrics.get('Sharpe Ratio', 0.0)
        sharpe_ratios.append(sharpe_ratio)
        print(f"  - Fold {fold+1}/{n_splits} | Sharpe Ratio: {sharpe_ratio:.4f}")

    # 5. Return the average Sharpe Ratio across all folds
    average_sharpe = np.mean(sharpe_ratios)
    print(f"Trial {trial.number}: Average Sharpe Ratio: {average_sharpe:.4f}")

    return average_sharpe

def main():
    """
    Main execution function to run the Optuna study.
    """
    global X, y

    # Load and prepare data
    print("--- Loading and Preparing Data for Tuning ---")
    df = pd.read_parquet(FEATURE_PATH)
    df['label'] = get_tri_barrier_labels(df['btc_close'])
    df = df.dropna(subset=['label'])
    label_map = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['label'].map(label_map)
    features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume', 'gld_volume', 'event_name', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]
    X = df[features]
    y = df['label']
    print(f"Data prepared. Shape: {X.shape}")

    # Create and run the Optuna study
    print("\\n--- Starting Optuna Hyperparameter Study ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # n_trials can be increased for a more exhaustive search

    # Print the results
    print("\\n--- Study Complete ---")
    print(f"Best trial number: {study.best_trial.number}")
    print("Best Sharpe Ratio: {:.4f}".format(study.best_value))
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

if __name__ == '__main__':
    main()
