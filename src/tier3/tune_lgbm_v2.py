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

# --- Optuna Objective Function ---
def objective(trial):
    """
    The function for Optuna to optimize. It trains a model with a set of
    hyperparameters, runs a backtest, and returns the Sharpe Ratio.
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

    # 2. Train the model
    model = lgb.LGBMClassifier(**params)

    # Calculate class weights for the training data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model.fit(X_train, y_train, sample_weight=y_train.map(weight_dict))

    # 3. Generate signals and run backtest
    predictions_proba = model.predict_proba(X_test)
    predictions_mapped = np.argmax(predictions_proba, axis=1)
    signal_map = {0: -1, 1: 0, 2: 1} # Map back to original labels
    signals = pd.Series(predictions_mapped, index=X_test.index).map(signal_map)

    # We run the backtest without the strategy overlay to purely measure the model's signal quality
    backtester = VectorizedBacktester(
        price_data=y_test_prices,
        signals=signals,
        initial_capital=100000
    )

    # 4. Return the Sharpe Ratio for Optuna to maximize
    metrics = backtester.get_performance_metrics()
    # The metric is a string like "1.23", so we convert it to float
    sharpe_ratio = float(metrics['Sharpe Ratio'])

    return sharpe_ratio

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

    # 6. Train the final model with the best parameters on the full dataset
    print("\\n--- Training Final Model with Best Hyperparameters ---")
    best_params = best_trial.params
    final_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, **best_params)

    # Prepare full dataset
    full_X = df[features]
    full_y = df['label'].map({-1: 0, 0: 1, 1: 2})

    # Calculate class weights for the full data
    final_class_weights = compute_class_weight('balanced', classes=np.unique(full_y), y=full_y)
    final_weight_dict = {i: final_class_weights[i] for i in range(len(final_class_weights))}

    final_model.fit(full_X, full_y, sample_weight=full_y.map(final_weight_dict))

    # 7. Save the final model
    model_path = os.path.join(MODEL_DIR, 'lgbm_v2_tuned_model.txt')
    final_model.booster_.save_model(model_path)
    print(f"\\nFinal tuned model saved to: {model_path}")
