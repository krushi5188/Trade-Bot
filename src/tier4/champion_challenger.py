# This script will orchestrate the Champion/Challenger model retraining and evaluation pipeline.

import os
import sys
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time
import subprocess
import random
import json
from timeseriescv.cross_validation import PurgedWalkForwardCV

# Add the project root to the Python path to enable imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels
from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

def run_script(script_path):
    """Executes a script as a subprocess."""
    try:
        print(f"[{time.ctime()}] --- Running script: {script_path} ---")
        subprocess.run(['python', script_path], check=True)
        print(f"[{time.ctime()}] --- Script finished: {script_path} ---")
    except subprocess.CalledProcessError as e:
        print(f"[{time.ctime()}] ERROR: Script {script_path} failed with exit code {e.returncode}")
        raise

def run_feature_engineering_pipeline():
    """
    Runs the full data and feature engineering pipeline by calling the scripts
    as subprocesses.
    """
    print(f"[{time.ctime()}] --- Running Data and Feature Engineering Pipeline ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    create_lean_dataset_script = os.path.join(project_root, 'src/tier2/create_lean_dataset.py')
    feature_engineering_script = os.path.join(project_root, 'src/tier2/feature_engineering_v2.py')
    run_script(create_lean_dataset_script)
    run_script(feature_engineering_script)
    print(f"[{time.ctime()}] --- Pipeline Complete ---")

def run_purged_walk_forward_cv():
    """
    Runs Purged Walk-Forward Cross-Validation to train a challenger model and
    generate robust, out-of-sample predictions.
    """
    print(f"[{time.ctime()}] --- Running Purged Walk-Forward CV ---")

    # 1. Load Data and Define Feature Set
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    print(f"[{time.ctime()}] Loading V2 feature dataset from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)
    print(f"[{time.ctime()}] V2 dataset loaded. Shape: {df.shape}")

    # 2. Create Labels and Time Information for CV
    print(f"[{time.ctime()}] Creating tri-barrier labels...")
    look_forward_period = 24 # This MUST match the value in get_tri_barrier_labels
    df['label'] = get_tri_barrier_labels(df['btc_close'], look_forward=look_forward_period)
    df = df.dropna(subset=['label'])

    label_map = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['label'].map(label_map)

    # Define pred_times and eval_times for PurgedWalkForwardCV
    # The values should be numeric representations of the time, and the index
    # must match the DataFrame's index for alignment.
    pred_times = pd.Series(pd.to_numeric(df.index), index=df.index)
    eval_times = pd.Series(pd.to_numeric(df.index) + look_forward_period, index=df.index)

    features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume', 'gld_volume', 'event_name', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]
    X = df[features]
    y = df['label']

    # 3. Setup PurgedWalkForwardCV
    n_splits = 5  # Number of folds
    n_test_splits = 1 # Number of test sets in each fold
    cv_splitter = PurgedWalkForwardCV(n_splits=n_splits, n_test_splits=n_test_splits)

    # 4. Cross-Validation Loop
    out_of_sample_predictions = []
    for i, (train_indices, test_indices) in enumerate(cv_splitter.split(X, y, pred_times=pred_times, eval_times=eval_times)):
        print(f"--- Processing Fold {i+1}/{n_splits} ---")
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        # Calculate class weights for this specific fold
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        sample_weight = y_train.map(class_weight_dict)

        # Train the model for this fold
        lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weight)
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.9, 'verbose': -1, 'device': 'cpu',
            'seed': random.randint(0, 100000)
        }
        model = lgb.train(params, lgb_train, num_boost_round=100)

        # Generate and store out-of-sample predictions
        preds_proba = model.predict(X_test)
        preds = pd.Series(np.argmax(preds_proba, axis=1), index=X_test.index)
        out_of_sample_predictions.append(preds)

    # 5. Concatenate all OOS predictions
    all_oos_preds = pd.concat(out_of_sample_predictions).sort_index()

    # Align predictions with the original dataframe to get price data for backtesting
    df_for_backtest = df.loc[all_oos_preds.index].copy()
    df_for_backtest['predictions'] = all_oos_preds

    print(f"[{time.ctime()}] --- Purged Walk-Forward CV Complete ---")
    return df_for_backtest

def backtest_challenger_model(df_with_predictions):
    """
    Backtests the challenger model using the out-of-sample predictions.
    """
    print(f"[{time.ctime()}] --- Backtesting Challenger Model on OOS Predictions ---")

    # 1. Define Paths & Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_name = "Challenger Model (Walk-Forward)"

    # 2. Extract price data and signals for the backtest period
    y_test_prices = df_with_predictions['btc_close']
    predictions_mapped = df_with_predictions['predictions']

    # Remap predictions back to original labels: 0 -> -1 (Sell), 1 -> 0 (Hold), 2 -> 1 (Buy)
    signal_map = {0: -1, 1: 0, 2: 1}
    signals = predictions_mapped.map(signal_map)

    # 3. Initialize Strategy Overlay and run Backtest
    strategy_overlay = StrategyOverlay(price_data=y_test_prices, volatility_lookback=21, volatility_target=0.02)
    backtester = VectorizedBacktester(price_data=y_test_prices, signals=signals, initial_capital=100000, strategy_overlay=strategy_overlay)

    # 4. Generate and Print Performance Report
    print(f"\n--- Backtest Performance: {model_name} ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("-----------------------------------\n")

    # 5. Save Equity Curve
    plot_path = os.path.join(OUTPUT_DIR, "challenger_walk_forward_equity_curve.png")
    backtester.plot_equity_curve(plot_path)

    return metrics

def load_champion_metrics():
    """
    Loads the champion model's performance metrics from a file.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    METRICS_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/champion_metrics.json')

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    else:
        # Return default metrics if the file doesn't exist
        return {
            'Sharpe Ratio': 0.05,
            'Total Return (%)': 9.28
        }

def save_champion_metrics(metrics):
    """
    Saves the new champion's metrics to a file.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    METRICS_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/champion_metrics.json')

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_feature_importance(model, features):
    """
    Calculates and saves the feature importance of the model.
    """
    print(f"[{time.ctime()}] --- Calculating and Saving Feature Importance ---")

    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values(by='importance', ascending=False)

    # Define save path with a timestamp
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/feature_importance')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f'feature_importance_{timestamp}.csv')

    # Save to CSV
    importance_df.to_csv(save_path, index=False)
    print(f"Feature importance saved to {save_path}")

def retrain_and_save_challenger(challenger_metrics, champion_metrics):
    """
    Retrains the challenger model on the full dataset and saves it, but only
    if it has outperformed the champion in the walk-forward backtest.
    """
    print(f"[{time.ctime()}] --- Final Retraining and Model Promotion ---")

    # Only proceed if the challenger is better
    if challenger_metrics.get('Sharpe Ratio', 0) <= champion_metrics.get('Sharpe Ratio', 0):
        print("Challenger is not better based on walk-forward CV. Keeping the current champion.")
        print(f"[{time.ctime()}] --- Process Complete ---")
        return

    print("Challenger outperformed champion in CV. Retraining on full dataset for promotion.")

    # 1. Load Full Dataset
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    df = pd.read_parquet(FEATURE_PATH)
    df['label'] = get_tri_barrier_labels(df['btc_close'])
    df = df.dropna(subset=['label'])
    label_map = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['label'].map(label_map)
    features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume', 'gld_volume', 'event_name', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]
    X_full = df[features]
    y_full = df['label']

    # 2. Calculate Class Weights on Full Dataset
    classes = np.unique(y_full)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_full)
    class_weight_dict = dict(zip(classes, weights))
    sample_weight = y_full.map(class_weight_dict)

    # 3. Train Final Model
    lgb_full = lgb.Dataset(X_full, y_full, weight=sample_weight)
    params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.9, 'verbose': -1, 'device': 'cpu',
        'seed': random.randint(0, 100000)
    }
    final_model = lgb.train(params, lgb_full, num_boost_round=100)
    print(f"[{time.ctime()}] Final model retraining complete.")

    # 4. Promote to Champion
    CHAMPION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/lgbm_v2_model.txt')
    final_model.save_model(CHAMPION_MODEL_PATH)
    print(f"Promoted new challenger model to champion: {CHAMPION_MODEL_PATH}")

    # 5. Save Feature Importance
    save_feature_importance(final_model, features)

    # 6. Save the new champion's metrics
    save_champion_metrics(challenger_metrics)
    print("Saved new champion metrics.")
    print(f"[{time.ctime()}] --- Promotion Complete ---")


def main():
    """
    Main function to run the Champion/Challenger pipeline.
    """
    print(f"[{time.ctime()}] --- Starting Champion/Challenger Pipeline ---")

    # 1. Run the data and feature engineering pipeline
    run_feature_engineering_pipeline()

    # 2. Run Purged Walk-Forward CV to get robust OOS predictions
    df_with_predictions = run_purged_walk_forward_cv()

    # 3. Backtest the challenger model on these predictions
    challenger_metrics = backtest_challenger_model(df_with_predictions)

    # 4. Compare metrics and conditionally retrain and promote the new model
    champion_metrics = load_champion_metrics()
    print(f"Challenger Sharpe Ratio (Walk-Forward): {challenger_metrics.get('Sharpe Ratio', 0)}")
    print(f"Champion Sharpe Ratio (from previous run): {champion_metrics.get('Sharpe Ratio', 0)}")
    retrain_and_save_challenger(challenger_metrics, champion_metrics)

    print(f"[{time.ctime()}] --- Champion/Challenger Pipeline Finished ---")

if __name__ == '__main__':
    main()
