# This script loads the champion LightGBM model, generates predictions,
# applies the strategy overlay, and evaluates performance using the VectorizedBacktester.

import pandas as pd
import lightgbm as lgb
import os
import sys
import numpy as np

# Add the project root to the Python path to enable imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay
from src.tier1.modeling.train_xgboost import get_tri_barrier_labels

def run_lgbm_backtest_with_overlay():
    """
    Loads the LightGBM model, applies the strategy overlay, and runs the backtest.
    """
    # 1. Define Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_03_final.parquet')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/lgbm_v1_stable.txt')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model and Data
    model_name = "LightGBM with Strategy Overlay"
    print(f"--- Running Backtest for: {model_name} ---")
    print("Loading model and feature dataset...")

    model = lgb.Booster(model_file=MODEL_PATH)
    df = pd.read_parquet(FEATURE_PATH)

    # 3. Replicate Test Set
    print("Replicating test set...")
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels
    df = df.iloc[:-24] # Ensure we have labels for the test set period

    features_to_exclude = ['btc_close', 'btc_volume', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]

    split_index = int(len(df) * 0.8)
    X_test = df[features].iloc[split_index:]
    y_test_prices = df['btc_close'].iloc[split_index:]

    # 4. Generate Model Predictions
    print("Generating trading signals...")
    # LightGBM's predict method returns probabilities for each class
    predictions_proba = model.predict(X_test)
    # Get the class with the highest probability
    predictions_mapped = np.argmax(predictions_proba, axis=1)

    signal_map = {0: -1, 1: 0, 2: 1} # Sell, Hold, Buy
    signals = pd.Series(predictions_mapped, index=X_test.index).map(signal_map)

    # 5. Initialize and Apply Strategy Overlay
    print("Initializing and applying strategy overlay...")
    strategy_overlay = StrategyOverlay(
        price_data=y_test_prices,
        volatility_lookback=21,
        volatility_target=0.02 # Target 2% daily volatility
    )

    # 6. Run the Backtest with the Overlay
    print("Initializing and running backtester...")
    backtester = VectorizedBacktester(
        price_data=y_test_prices,
        signals=signals,
        initial_capital=100000,
        strategy_overlay=strategy_overlay # Pass the overlay object
    )

    # 7. Generate Performance Report
    print(f"\\n--- Backtest Performance: {model_name} ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 8. Generate Equity Curve Plot
    plot_filename = f"{model_name.lower().replace(' ', '_')}_equity_curve.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    backtester.plot_equity_curve(plot_path)
    print("-" * 35 + "\\n")


if __name__ == '__main__':
    run_lgbm_backtest_with_overlay()
