# This script runs a backtest on the V2 model and features.

import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

def run_v2_backtest():
    """
    Loads the V2 model and features, and runs a backtest with the strategy overlay.
    """
    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/lgbm_v2_model.txt')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model and Data
    model_name = "LightGBM V2 with Strategy Overlay"
    print(f"--- Running Backtest for: {model_name} ---")

    model = lgb.Booster(model_file=MODEL_PATH)
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
    plot_filename = "lgbm_v2_equity_curve.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    backtester.plot_equity_curve(plot_path)
    print("-----------------------------------\\n")


if __name__ == '__main__':
    run_v2_backtest()
