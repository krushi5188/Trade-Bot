# This script performs hyperparameter tuning for the LightGBM V2 model
# using the Optuna optimization framework. The goal is to find the set of
# hyperparameters that maximizes the Sharpe Ratio from our backtester.

import optuna
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import sys
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier2.backtester import VectorizedBacktester

# --- Global Variables & Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'src/tier3/modeling')
DB_PATH = os.path.join(PROJECT_ROOT, 'analysis/tuning_studies/optuna_study_lgbm_v2.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_tri_barrier_labels(close, look_forward=24, upper_pct=0.02, lower_pct=0.01):
    """
    Creates Tri-Barrier Labels for a given price series.
    - Label 1: Upper barrier (profit take) was hit.
    - Label -1: Lower barrier (stop loss) was hit.
    - Label 0: Neither barrier was hit within the look_forward period.
    """
    out = pd.Series(0, index=close.index)
    upper_barrier = close * (1 + upper_pct)
    lower_barrier = close * (1 - lower_pct)

    for i in range(len(close) - look_forward):
        future_path = close.iloc[i+1 : i+1+look_forward]
        if any(future_path >= upper_barrier.iloc[i]):
            out.iloc[i] = 1
        elif any(future_path <= lower_barrier.iloc[i]):
            out.iloc[i] = -1
    return out

# Load data once to be used by all trials
df = pd.read_parquet(FEATURE_PATH)

# Generate labels
print("Generating tri-barrier labels...")
labels = get_tri_barrier_labels(df['btc_close'])
df['label'] = labels
df = df.iloc[:-24] # We can't use the last rows as they have no future

features_to_exclude = [col for col in df.columns if '_close' in col or '_volume' in col or 'event_name' in col or 'label' in col]
features = [c for c in df.columns if c not in features_to_exclude]

# Replicate the same train/test split from the V2 model training
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df['label'].map({-1: 0, 0: 1, 1: 2}) # Remap for LGBM
X_test = test_df[features]
y_test_prices = test_df['btc_close']

# --- Optuna Objective Function with Cross-Validation ---
def objective(trial):
    """
    The function for Optuna to optimize. It trains a model with a set of
    hyperparameters, runs a backtest using Time-Series Cross-Validation,
    and returns the average Sharpe Ratio across all folds.
    """
    # 1. Define the hyperparameter search space
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }

    # 2. Perform Time-Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    sharpe_ratios = []

    # We use the full training data for cross-validation
    cv_X = df[features]
    cv_y = df['label'].map({-1: 0, 0: 1, 1: 2})
    cv_prices = df['btc_close']

    for train_index, val_index in tscv.split(cv_X):
        X_train_cv, X_val_cv = cv_X.iloc[train_index], cv_X.iloc[val_index]
        y_train_cv, y_val_cv = cv_y.iloc[train_index], cv_y.iloc[val_index]
        prices_val = cv_prices.iloc[val_index]

        # Train the model
        model = lgb.LGBMClassifier(**params)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_cv), y=y_train_cv)
        weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        model.fit(X_train_cv, y_train_cv, sample_weight=y_train_cv.map(weight_dict))

        # Generate signals and run backtest
        predictions_proba = model.predict_proba(X_val_cv)
        predictions_mapped = np.argmax(predictions_proba, axis=1)
        signal_map = {0: -1, 1: 0, 2: 1}
        signals = pd.Series(predictions_mapped, index=X_val_cv.index).map(signal_map)

        backtester = VectorizedBacktester(price_data=prices_val, signals=signals, initial_capital=100000)
        metrics = backtester.get_performance_metrics()
        sharpe_ratio = float(metrics['Sharpe Ratio'])
        sharpe_ratios.append(sharpe_ratio)

    # 3. Return the average Sharpe Ratio
    avg_sharpe_ratio = np.mean(sharpe_ratios)
    return avg_sharpe_ratio

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Optuna Hyperparameter Tuning for LightGBM V2 ---")

    # Create or load the study
    # The SQLite backend allows us to resume the study if it's interrupted
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{DB_PATH}")
    study = optuna.create_study(
        storage=storage,
        study_name="lgbm_v2_sharpe_optimization",
        direction="maximize",
        load_if_exists=True
    )

    # Run the optimization
    # The number of trials can be increased for a more thorough search
    study.optimize(objective, n_trials=50)

    print("\\n--- Optuna Study Complete ---")
    print(f"Study Name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Sharpe Ratio): {best_trial.value:.4f}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 6. Train the final model with the best parameters on the entire training set
    print("\\n--- Training Final Model with Best Hyperparameters on the Full Training Set ---")
    best_params = best_trial.params
    final_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, **best_params)

    # We use the full training set (the first 80% of the data)
    final_model.fit(X_train, y_train)

    # 7. Save the final model
    model_path = os.path.join(MODEL_DIR, 'lgbm_v3.1_tuned_cv_model.txt')
    final_model.booster_.save_model(model_path)
    print(f"\\nFinal tuned model saved to: {model_path}")
