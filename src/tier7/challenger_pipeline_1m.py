
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

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels
from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

def run_purged_walk_forward_cv_1m_v2():
    """
    Runs Purged Walk-Forward Cross-Validation on the V2 1-minute feature set.
    """
    print(f"[{time.ctime()}] --- Running Purged Walk-Forward CV for 1M Data (V2) ---")

    # 1. Load V2 1-Minute Feature Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/features/XAUUSD_1m_features_v2.parquet')
    print(f"[{time.ctime()}] Loading 1M V2 feature dataset from {FEATURE_PATH}...")
    df_1m = pd.read_parquet(FEATURE_PATH)
    print(f"[{time.ctime()}] Full 1M V2 dataset loaded. Shape: {df_1m.shape}")

    # --- MEMORY OPTIMIZATION ---
    print(f"[{time.ctime()}] Filtering data from 2020 onwards to reduce memory usage...")
    df_1m = df_1m[df_1m.index.year >= 2020]
    print(f"[{time.ctime()}] Filtered dataset shape: {df_1m.shape}")

    print(f"[{time.ctime()}] Optimizing data types...")
    for col in df_1m.columns:
        if df_1m[col].dtype == 'float64':
            df_1m[col] = df_1m[col].astype('float32')
    print(f"[{time.ctime()}] Data types optimized.")
    # --- END OPTIMIZATION ---

    # 2. Create Labels with a 60-minute Look-Forward Period
    print(f"[{time.ctime()}] Resampling to 1H to create tri-barrier labels (60 min look-forward)...")
    df_1h = df_1m['Close'].resample('1H').last().to_frame()
    look_forward_period = 60 # Predict 60 minutes ahead
    df_1h['label'] = get_tri_barrier_labels(df_1h['Close'], look_forward=look_forward_period, min_return=0.0005) # Added min_return for better label quality

    df_1h_labels = df_1h[['label']].copy()
    df_1h_labels = df_1h_labels.reindex(df_1m.index, method='ffill')

    df_1m['label'] = df_1h_labels['label']
    df_1m = df_1m.dropna(subset=['label'])

    label_map = {-1: 0, 0: 1, 1: 2}
    df_1m['label'] = df_1m['label'].map(label_map)

    # 3. Setup for Cross-Validation
    pred_times = pd.Series(pd.to_numeric(df_1m.index), index=df_1m.index)
    eval_times = pd.Series(pd.to_numeric(df_1m.index) + pd.Timedelta(minutes=look_forward_period).total_seconds() * 1e9, index=df_1m.index)

    features_to_exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'label']
    features = [c for c in df_1m.columns if c not in features_to_exclude]
    X = df_1m[features]
    y = df_1m['label']

    n_splits = 5
    cv_splitter = PurgedWalkForwardCV(n_splits=n_splits)

    # 4. Cross-Validation Loop
    out_of_sample_predictions = []
    for i, (train_indices, test_indices) in enumerate(cv_splitter.split(X, y, pred_times=pred_times, eval_times=eval_times)):
        print(f"--- Processing Fold {i+1}/{n_splits} ---")
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        sample_weight = y_train.map(class_weight_dict)

        lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weight)
        params = {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
                  'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
                  'feature_fraction': 0.9, 'verbose': -1, 'device': 'cpu', 'seed': 42}
        model = lgb.train(params, lgb_train, num_boost_round=100)

        preds_proba = model.predict(X_test)
        preds = pd.Series(np.argmax(preds_proba, axis=1), index=X_test.index)
        out_of_sample_predictions.append(preds)

    all_oos_preds = pd.concat(out_of_sample_predictions).sort_index()

    df_for_backtest = df_1m.loc[all_oos_preds.index].copy()
    df_for_backtest['predictions'] = all_oos_preds

    print(f"[{time.ctime()}] --- Purged Walk-Forward CV Complete ---")
    return df_for_backtest

def backtest_challenger_model_1m_v2(df_with_predictions):
    """
    Backtests the 1M V2 challenger model.
    """
    print(f"[{time.ctime()}] --- Backtesting 1M Challenger Model (V2) ---")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_name = "Challenger Model (1M Features V2)"

    y_test_prices = df_with_predictions['Close']
    signal_map = {0: -1, 1: 0, 2: 1}
    signals = df_with_predictions['predictions'].map(signal_map)

    signals_1h = signals.resample('1H').last().dropna()
    prices_1h = y_test_prices.resample('1H').last().reindex(signals_1h.index)

    strategy_overlay = StrategyOverlay(price_data=prices_1h, volatility_lookback=21, volatility_target=0.02)
    backtester = VectorizedBacktester(price_data=prices_1h, signals=signals_1h, initial_capital=100000, strategy_overlay=strategy_overlay, transaction_cost=0.0002)

    print(f"\n--- Backtest Performance: {model_name} ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("-----------------------------------\n")

    plot_path = os.path.join(OUTPUT_DIR, "challenger_1m_v2_equity_curve.png")
    backtester.plot_equity_curve(plot_path)

    return metrics

def main():
    print(f"[{time.ctime()}] --- Starting High-Frequency Challenger Pipeline (V2) ---")
    df_with_predictions = run_purged_walk_forward_cv_1m_v2()
    challenger_metrics = backtest_challenger_model_1m_v2(df_with_predictions)
    print(f"[{time.ctime()}] --- High-Frequency Challenger Pipeline (V2) Finished ---")
    print("Challenger V2 Performance Metrics:")
    print(json.dumps(challenger_metrics, indent=4))

if __name__ == '__main__':
    main()
