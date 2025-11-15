# This script runs a backtest on the V3 (tuned) model and V2 features.

import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier2.backtester import VectorizedBacktester

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

def run_v3_backtest():
    """
    Loads the V3 tuned model and V2 features, and runs a backtest.
    """
    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier3/modeling/lgbm_v2_tuned_model.txt')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model and Data
    model_name = "LightGBM V3 (Tuned)"
    print(f"--- Running Backtest for: {model_name} ---")

    model = lgb.Booster(model_file=MODEL_PATH)
    df = pd.read_parquet(FEATURE_PATH)

    # Generate labels to ensure correct test set alignment
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels
    df = df.iloc[:-24]

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

    # 5. Run the Backtest
    print("Initializing and running backtester...")
    backtester = VectorizedBacktester(
        price_data=y_test_prices,
        signals=signals,
        initial_capital=100000
    )

    # 6. Generate Performance Report
    print(f"\\n--- Backtest Performance: {model_name} ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 7. Generate Equity Curve Plot
    plot_filename = "lgbm_v3_tuned_equity_curve.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    backtester.plot_equity_curve(plot_path)
    print("-----------------------------------\\n")


if __name__ == '__main__':
    run_v3_backtest()
