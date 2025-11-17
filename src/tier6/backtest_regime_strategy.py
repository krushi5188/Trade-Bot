import os
import sys
import pandas as pd
import lightgbm as lgb
import json
import time
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

def backtest_regime_switching_strategy(
    feature_path='data/processed/features_v3_with_regimes.parquet',
    model_dir='src/tier6/regime_models',
    output_dir='analysis/backtest_reports'
):
    """
    Loads data with regime labels, loads regime-specific models, generates signals
    by switching models based on the regime, and runs a vectorized backtest.
    """
    print(f"[{time.ctime()}] --- Starting Regime-Switching Backtest ---")

    # 1. Setup Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_feature_path = os.path.join(PROJECT_ROOT, feature_path)
    full_model_dir = os.path.join(PROJECT_ROOT, model_dir)
    full_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    # 2. Load Data
    df = pd.read_parquet(full_feature_path)
    # For backtesting, we'll use the last 20% of the data as our out-of-sample test set
    test_size = 0.2
    split_index = int(len(df) * (1 - test_size))
    df_test = df.iloc[split_index:].copy()
    print(f"[{time.ctime()}] Loaded test data. Shape: {df_test.shape}")

    # 3. Load Regime Models
    models = {}
    regime_files = [f for f in os.listdir(full_model_dir) if f.endswith('_model.txt')]
    for file_name in regime_files:
        regime_id = int(file_name.split('_')[1])
        model_path = os.path.join(full_model_dir, file_name)
        models[regime_id] = lgb.Booster(model_file=model_path)
    print(f"[{time.ctime()}] Loaded {len(models)} regime-specific models.")

    # 4. Generate Predictions with Regime Switching
    all_predictions = []

    # Define features used for prediction
    features_to_exclude = [
        'btc_close', 'eur_close', 'gld_close', 'event_name', 'outcome',
        'market_regime', 'log_returns'
    ]
    features = [c for c in df.columns if c not in features_to_exclude and not c.endswith('_volume')]

    X_test = df_test[features]

    # Use tqdm for a progress bar as this can be slow
    print(f"[{time.ctime()}] Generating predictions for {len(df_test)} data points...")
    for index, row in tqdm(X_test.iterrows(), total=len(X_test)):
        current_regime = df_test.loc[index, 'market_regime']

        if current_regime in models:
            model = models[current_regime]
            # model.predict expects a 2D array, so we reshape the row
            prediction_probs = model.predict(row.values.reshape(1, -1))
            # Get the class with the highest probability
            prediction = prediction_probs.argmax(axis=1)[0]
            all_predictions.append(prediction)
        else:
            # If for some reason a regime has no model, predict 'hold' (class 1)
            all_predictions.append(1)

    # Map predictions back to (-1, 0, 1) signals
    # 0 -> sell (-1), 1 -> hold (0), 2 -> buy (1)
    signal_map = {0: -1, 1: 0, 2: 1}
    df_test['final_signal'] = [signal_map[p] for p in all_predictions]
    print(f"[{time.ctime()}] Prediction generation complete.")
    print("Final signal counts:")
    print(df_test['final_signal'].value_counts())

    # 5. Run Backtest
    price_data = df_test['btc_close']
    signals = df_test['final_signal']

    strategy_overlay = StrategyOverlay(price_data=price_data, volatility_lookback=21, volatility_target=0.02)
    backtester = VectorizedBacktester(price_data=price_data, signals=signals, initial_capital=100000, strategy_overlay=strategy_overlay)

    # 6. Generate and Save Performance Report
    print(f"\n[{time.ctime()}] --- Backtest Performance: Regime-Switching Strategy ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("--------------------------------------------------\n")

    metrics_path = os.path.join(full_output_dir, "tier6_regime_switching_performance.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Performance metrics saved to: {metrics_path}")

    plot_path = os.path.join(full_output_dir, "tier6_regime_switching_equity_curve.png")
    backtester.plot_equity_curve(plot_path)

    print(f"[{time.ctime()}] --- Backtesting Complete ---")

if __name__ == '__main__':
    backtest_regime_switching_strategy()
