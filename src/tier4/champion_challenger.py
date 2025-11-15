# This script will orchestrate the Champion/Challenger model retraining and evaluation pipeline.

import os
import sys
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time
import subprocess

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

    # Define script paths relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    create_lean_dataset_script = os.path.join(project_root, 'src/tier2/create_lean_dataset.py')
    feature_engineering_script = os.path.join(project_root, 'src/tier2/feature_engineering_v2.py')

    run_script(create_lean_dataset_script)
    run_script(feature_engineering_script)

    print(f"[{time.ctime()}] --- Pipeline Complete ---")

def train_challenger_model(use_gpu=False):
    """
    Trains a new challenger model on the latest data.
    """
    print(f"[{time.ctime()}] --- Training Challenger Model ---")

    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'src/tier4/modeling')
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, 'challenger_model.txt')

    # 2. Load Data
    print(f"[{time.ctime()}] Loading V2 feature dataset from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)
    print(f"[{time.ctime()}] V2 dataset loaded. Shape: {df.shape}")

    # 3. Create Labels and Define Feature Set
    print(f"[{time.ctime()}] Creating tri-barrier labels...")
    df['label'] = get_tri_barrier_labels(df['btc_close'])
    df = df.dropna(subset=['label']) # Drop rows where labels couldn't be generated

    # Remap labels for LightGBM multiclass objective
    label_map = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['label'].map(label_map)

    # Exclude non-feature columns
    features_to_exclude = [col for col in df.columns if '_close' in col or '_volume' in col or 'event_name' in col or 'label' in col]
    features = [c for c in df.columns if c not in features_to_exclude]

    X = df[features]
    y = df['label']
    print(f"[{time.ctime()}] Feature set defined with {len(features)} features.")

    # 4. Split Data (Chronological Split)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    print(f"[{time.ctime()}] Data split chronologically into training and testing sets.")

    # 5. Calculate Class Weights
    print(f"[{time.ctime()}] Calculating class weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weight_dict}")

    # 6. Train LightGBM Model
    print(f"[{time.ctime()}] Training LightGBM model with class weights...")
    sample_weight = y_train.map(class_weight_dict)
    lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weight)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'device': 'gpu' if use_gpu else 'cpu'
    }

    model = lgb.train(params, lgb_train, num_boost_round=100)
    print(f"[{time.ctime()}] Model training complete.")

    # 7. Evaluate Model
    print(f"[{time.ctime()}] Evaluating model performance...")
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [list(row).index(max(row)) for row in y_pred_proba]
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\\n--- Challenger Model Performance ---")
    print(f"Accuracy on test set: {accuracy:.4f}")
    print("--------------------------\\n")

    # 8. Save Model
    print(f"[{time.ctime()}] Saving trained model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    print(f"[{time.ctime()}] Model saved successfully.")

    return model

def backtest_challenger_model(model):
    """
    Backtests the challenger model on a recent hold-out set.
    """
    print(f"[{time.ctime()}] --- Backtesting Challenger Model ---")

    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Data
    model_name = "Challenger Model"
    print(f"--- Running Backtest for: {model_name} ---")

    df = pd.read_parquet(FEATURE_PATH)

    # 3. Define Feature Set and Replicate Test Set
    features_to_exclude = [col for col in df.columns if '_close' in col or '_volume' in col or 'event_name' in col or 'label' in col]
    features = [c for c in df.columns if c not in features_to_exclude]

    # We will backtest on the same 20% of the data used for testing in the training script.
    split_index = int(len(df) * 0.8)
    X_test = df[features].iloc[split_index:]
    y_test_prices = df['btc_close'].iloc[split_index:]

    # 4. Generate Model Predictions
    print("Generating trading signals...")
    predictions_proba = model.predict(X_test)
    predictions_mapped = np.argmax(predictions_proba, axis=1)

    # Remap predictions back to original labels: 0 -> -1 (Sell), 1 -> 0 (Hold), 2 -> 1 (Buy)
    signal_map = {0: -1, 1: 0, 2: 1}
    signals = pd.Series(predictions_mapped, index=X_test.index).map(signal_map)

    # 5. Initialize Strategy Overlay
    print("Initializing strategy overlay...")
    strategy_overlay = StrategyOverlay(
        price_data=y_test_prices,
        volatility_lookback=21,
        volatility_target=0.02
    )

    # 6. Run the Backtest
    print("Initializing and running backtester...")
    backtester = VectorizedBacktester(
        price_data=y_test_prices,
        signals=signals,
        initial_capital=100000,
        strategy_overlay=strategy_overlay
    )

    # 7. Generate Performance Report
    print(f"\\n--- Backtest Performance: {model_name} ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 8. Generate Equity Curve Plot
    plot_filename = "challenger_equity_curve.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    backtester.plot_equity_curve(plot_path)
    print("-----------------------------------\\n")

    return metrics

import json

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

def compare_and_promote_model(challenger_metrics):
    """
    Compares the challenger and champion models and promotes the challenger if it's better.
    """
    print(f"[{time.ctime()}] --- Comparing Models ---")

    champion_metrics = load_champion_metrics()

    print(f"Challenger Sharpe Ratio: {challenger_metrics.get('Sharpe Ratio', 0)}")
    print(f"Champion Sharpe Ratio: {champion_metrics.get('Sharpe Ratio', 0)}")

    # Promote the challenger if its Sharpe Ratio is higher.
    if challenger_metrics.get('Sharpe Ratio', 0) > champion_metrics.get('Sharpe Ratio', 0):
        print("Challenger is better. Promoting to champion.")

        # Define paths
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        CHALLENGER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier4/modeling/challenger_model.txt')
        CHAMPION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/lgbm_v2_model.txt')

        # Overwrite the champion model with the challenger model
        os.rename(CHALLENGER_MODEL_PATH, CHAMPION_MODEL_PATH)
        print(f"Promoted challenger model to: {CHAMPION_MODEL_PATH}")

        # Save the new champion's metrics
        save_champion_metrics(challenger_metrics)
        print("Saved new champion metrics.")

    else:
        print("Challenger is not better. Keeping the current champion.")

    print(f"[{time.ctime()}] --- Comparison Complete ---")

def main():
    """
    Main function to run the Champion/Challenger pipeline.
    """
    print(f"[{time.ctime()}] --- Starting Champion/Challenger Pipeline ---")

    # 1. Run the data and feature engineering pipeline
    run_feature_engineering_pipeline()

    # 2. Train the challenger model
    challenger_model = train_challenger_model(use_gpu=True)

    # 3. Backtest the challenger model
    challenger_metrics = backtest_challenger_model(challenger_model)

    # 4. Compare and promote the model
    compare_and_promote_model(challenger_metrics)

    print(f"[{time.ctime()}] --- Champion/Challenger Pipeline Finished ---")

if __name__ == '__main__':
    main()
