# This script will load the trained model, generate predictions on the test set,
# and use the VectorizedBacktester to evaluate the strategy's performance.

import pandas as pd
import xgboost as xgb
import os
import sys

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tier2.backtester import VectorizedBacktester
from tier1.modeling.train_xgboost import get_tri_barrier_labels

def run_backtest_on_model(model_name, model_filename):
    """
    Loads a specified model and test data, generates predictions,
    and runs the backtest.

    Args:
        model_name (str): A descriptive name for the model (e.g., "Baseline").
        model_filename (str): The filename of the saved model JSON.
    """
    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_03_final.parquet')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier1/modeling', model_filename)
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model and Data
    print(f"--- Running Backtest for: {model_name} Model ---")
    print("Loading model and feature dataset...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    df = pd.read_parquet(FEATURE_PATH)

    # 3. Replicate Test Set
    print("Replicating test set...")
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels
    df = df.iloc[:-24]

    features_to_exclude = ['btc_close', 'btc_volume', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]

    split_index = int(len(df) * 0.8)
    X_test = df[features].iloc[split_index:]
    y_test_prices = df['btc_close'].iloc[split_index:]

    # 4. Generate Model Predictions
    print("Generating trading signals...")
    predictions_mapped = model.predict(X_test)
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
    plot_filename = f"{model_name.lower().replace(' ', '_')}_equity_curve.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    backtester.plot_equity_curve(plot_path)
    print("-" * 35 + "\\n")


if __name__ == '__main__':
    # Run backtest for both the original baseline and the new tuned model
    run_backtest_on_model("Baseline", "xgboost_baseline_v1.json")
    run_backtest_on_model("Tuned", "xgboost_tuned_v1.json")
