import os
import sys
import pandas as pd
import lightgbm as lgb
import optuna
import json
import time
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tier1.modeling.train_xgboost import get_tri_barrier_labels
from src.tier2.backtester import VectorizedBacktester

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v3_with_regimes.parquet')
PARAMS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/tier6/regime_models')
N_TRIALS = 15 # Number of tuning trials per regime
N_SPLITS = 3 # Number of splits for time series cross-validation
TRANSACTION_COST = 0.001

def objective(trial, X, y, price_data):
    """
    The objective function for Optuna to optimize.
    Trains a model with a given set of hyperparameters and evaluates it using
    time-series cross-validation, returning the average post-cost Sharpe Ratio.
    """
    # Define hyperparameter search space
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'class_weight': 'balanced',
        'verbose': -1,
        'seed': 42
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    sharpe_ratios = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        signal_map = {0: -1, 1: 0, 2: 1}
        signals = pd.Series([signal_map[p] for p in preds], index=y_test.index)

        test_price_data = price_data.loc[signals.index]

        # Backtest with transaction costs
        backtester = VectorizedBacktester(test_price_data, signals, transaction_cost=TRANSACTION_COST)
        metrics = backtester.get_performance_metrics()
        sharpe_ratios.append(metrics.get('Sharpe Ratio', 0.0))

    return np.mean(sharpe_ratios)


def tune_models_for_each_regime():
    """
    Main function to orchestrate the hyperparameter tuning process for each regime.
    """
    print(f"[{time.ctime()}] --- Starting Regime-Specific Hyperparameter Tuning ---")
    os.makedirs(PARAMS_OUTPUT_DIR, exist_ok=True)

    # 1. Load data and generate labels
    df = pd.read_parquet(FEATURE_PATH)
    df['outcome'] = get_tri_barrier_labels(df['btc_close'])
    df.dropna(subset=['outcome'], inplace=True)
    df['outcome'] = df['outcome'].astype(int)

    features_to_exclude = [
        'btc_close', 'eur_close', 'gld_close', 'event_name', 'outcome',
        'market_regime', 'log_returns'
    ]
    features = [c for c in df.columns if c not in features_to_exclude and not c.endswith('_volume')]

    regimes = sorted(df['market_regime'].unique())
    print(f"[{time.ctime()}] Found regimes: {regimes}")

    # 2. Loop through each regime and run an Optuna study
    for regime in regimes:
        print(f"\n[{time.ctime()}] --- Tuning for Regime {regime} ---")
        df_regime = df[df['market_regime'] == regime].copy()

        if len(df_regime) < 500: # Ensure enough data for meaningful CV
            print(f"Skipping regime {regime} due to insufficient data ({len(df_regime)} samples).")
            continue

        X_regime = df_regime[features]
        y_regime = df_regime['outcome'].map({-1: 0, 0: 1, 1: 2})
        price_data_regime = df_regime['btc_close']

        study = optuna.create_study(direction='maximize', study_name=f"regime_{regime}_tuning")
        study.optimize(lambda trial: objective(trial, X_regime, y_regime, price_data_regime), n_trials=N_TRIALS, show_progress_bar=True)

        print(f"[{time.ctime()}] Best Sharpe Ratio for Regime {regime}: {study.best_value:.4f}")

        # Save the best parameters
        params_path = os.path.join(PARAMS_OUTPUT_DIR, f"best_params_regime_{regime}.json")
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Saved best parameters to {params_path}")

    print(f"\n[{time.ctime()}] --- All Regimes Tuned Successfully ---")

if __name__ == '__main__':
    tune_models_for_each_regime()
