# This script orchestrates the self-improving, autonomous training pipeline.
# It integrates the Tier 6 regime-switching strategy into the Tier 4
# Champion/Challenger framework.

import os
import sys
import pandas as pd
import lightgbm as lgb
import json
import time
import numpy as np
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tier1.modeling.train_xgboost import get_tri_barrier_labels
from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
CHAMPION_METRICS_PATH = os.path.join(PROJECT_ROOT, 'src/tier4/champion_metrics.json')
CHAMPION_MODEL_DIR = os.path.join(PROJECT_ROOT, 'src/tier4/champion_models')
BACKTEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
N_REGIMES = 3
TEST_SIZE = 0.2
TRANSACTION_COST = 0.001 # Set a default of 0.1%

# --- Core Pipeline Functions ---

def run_regime_detection(df):
    """
    Identifies market regimes in the data using a Gaussian Hidden Markov Model.
    """
    print(f"[{time.ctime()}] --- Identifying {N_REGIMES} Market Regimes ---")
    df['log_returns'] = np.log(df['btc_close'] / df['btc_close'].shift(1))
    hmm_features = ['log_returns', 'btc_volatility']
    df_hmm = df[hmm_features].copy().dropna()
    X = df_hmm.values

    model = GaussianHMM(n_components=N_REGIMES, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)

    df.loc[df_hmm.index, 'market_regime'] = hidden_states
    df['market_regime'].ffill(inplace=True)
    df.dropna(subset=['market_regime'], inplace=True)
    df['market_regime'] = df['market_regime'].astype(int)
    print(f"[{time.ctime()}] Regime identification complete.")
    return df

def train_challenger_models(df_train):
    """
    Trains a set of challenger models, one for each market regime.
    """
    print(f"[{time.ctime()}] --- Training Challenger Model Set ---")
    challenger_models = {}
    regimes = sorted(df_train['market_regime'].unique())

    features_to_exclude = [
        'btc_close', 'eur_close', 'gld_close', 'event_name', 'outcome',
        'market_regime', 'log_returns'
    ]
    features = [c for c in df_train.columns if c not in features_to_exclude and not c.endswith('_volume')]

    for regime in regimes:
        df_regime = df_train[df_train['market_regime'] == regime]
        if len(df_regime) < 100: continue

        X_train_regime = df_regime[features]
        y_train_regime = df_regime['outcome'].map({-1: 0, 0: 1, 1: 2})

        # Load best params if available, otherwise use defaults
        params_path = os.path.join(PROJECT_ROOT, f'src/tier6/regime_models/best_params_regime_{regime}.json')
        if os.path.exists(params_path):
            print(f"Loading optimized parameters for Regime {regime}...")
            with open(params_path, 'r') as f:
                params = json.load(f)
            params['objective'] = 'multiclass'
            params['num_class'] = 3
            params['class_weight'] = 'balanced'
            params['seed'] = 42
            params['verbose'] = -1
        else:
            print(f"No optimized parameters found for Regime {regime}. Using defaults.")
            params = {'objective': 'multiclass', 'num_class': 3, 'class_weight': 'balanced', 'seed': 42, 'verbose': -1}


        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_regime, y_train_regime)
        challenger_models[regime] = model
        print(f"Trained challenger model for Regime {regime} on {len(df_regime)} samples.")

    print(f"[{time.ctime()}] Challenger model set training complete.")
    return challenger_models, features

def generate_challenger_signals(challenger_models, features, df_test):
    """
    Generates trading signals for the test set using the regime-switching logic.
    """
    print(f"[{time.ctime()}] --- Generating Signals with Regime-Switching ---")
    all_predictions = []
    X_test = df_test[features]

    for index, row in tqdm(X_test.iterrows(), total=len(X_test), desc="Generating Signals"):
        current_regime = df_test.loc[index, 'market_regime']
        if current_regime in challenger_models:
            model = challenger_models[current_regime]
            prediction = model.predict(row.values.reshape(1, -1))[0]
            all_predictions.append(prediction)
        else:
            all_predictions.append(1) # Default to 'hold' if no model exists

    signal_map = {0: -1, 1: 0, 2: 1}
    df_test['final_signal'] = [signal_map[p] for p in all_predictions]
    return df_test

def backtest_challenger(df_with_signals):
    """
    Backtests the performance of the generated signals.
    """
    print(f"[{time.ctime()}] --- Backtesting Challenger Performance ---")
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

    price_data = df_with_signals['btc_close']
    signals = df_with_signals['final_signal']

    overlay = StrategyOverlay(price_data, volatility_lookback=21, volatility_target=0.02)
    backtester = VectorizedBacktester(
        price_data,
        signals,
        initial_capital=100000,
        transaction_cost=TRANSACTION_COST,
        strategy_overlay=overlay
    )
    metrics = backtester.get_performance_metrics()

    print(f"\n--- Challenger Performance (Post-Cost, {TRANSACTION_COST*100:.3f}%) ---")
    for key, value in metrics.items(): print(f"{key}: {value:.4f}")
    print("----------------------------\n")

    plot_path = os.path.join(BACKTEST_OUTPUT_DIR, "challenger_equity_curve.png")
    backtester.plot_equity_curve(plot_path)
    return metrics

def promote_challenger_to_champion(challenger_metrics, champion_metrics, full_df):
    """
    Compares challenger to champion and promotes if better by retraining on all data.
    """
    print(f"[{time.ctime()}] --- Model Promotion Ceremony ---")
    challenger_sharpe = challenger_metrics.get('Sharpe Ratio', 0)
    champion_sharpe = champion_metrics.get('Sharpe Ratio', 0)

    print(f"Challenger Sharpe Ratio: {challenger_sharpe:.4f}")
    print(f"Champion Sharpe Ratio:   {champion_sharpe:.4f}")

    if challenger_sharpe <= champion_sharpe:
        print("Challenger is not better. The current Champion reigns supreme!")
        return

    print("A new Champion has been crowned! Retraining on full dataset...")

    # Retrain models on the full dataset
    champion_models, features = train_challenger_models(full_df)

    # Save the new champion models
    os.makedirs(CHAMPION_MODEL_DIR, exist_ok=True)
    for regime, model in champion_models.items():
        model_path = os.path.join(CHAMPION_MODEL_DIR, f"champion_regime_{regime}_model.txt")
        model.booster_.save_model(model_path)
        print(f"Saved new champion model to {model_path}")

    # Save the new champion's metrics
    with open(CHAMPION_METRICS_PATH, 'w') as f:
        json.dump(challenger_metrics, f, indent=4)
    print(f"New champion metrics saved to {CHAMPION_METRICS_PATH}")
    print(f"[{time.ctime()}] --- Promotion Complete ---")

# --- Main Orchestration ---

def main():
    """
    Runs the full, autonomous Champion/Challenger pipeline.
    """
    print(f"[{time.ctime()}] === Starting Autonomous Training Pipeline ===")

    # 1. Load Data
    df = pd.read_parquet(FEATURE_PATH)

    # 2. Identify Market Regimes
    df = run_regime_detection(df)

    # 3. Generate Labels
    df['outcome'] = get_tri_barrier_labels(df['btc_close'])
    df.dropna(subset=['outcome'], inplace=True)
    df['outcome'] = df['outcome'].astype(int)

    # 4. Split Data
    split_index = int(len(df) * (1 - TEST_SIZE))
    df_train, df_test = df.iloc[:split_index], df.iloc[split_index:]

    # 5. Train Challenger Models
    challenger_models, features = train_challenger_models(df_train)

    # 6. Generate Signals
    df_test_with_signals = generate_challenger_signals(challenger_models, features, df_test)

    # 7. Backtest Challenger
    challenger_metrics = backtest_challenger(df_test_with_signals)

    # 8. Compare and Promote
    try:
        with open(CHAMPION_METRICS_PATH, 'r') as f:
            champion_metrics = json.load(f)
    except FileNotFoundError:
        champion_metrics = {'Sharpe Ratio': 0.0} # Default if no champion exists

    promote_challenger_to_champion(challenger_metrics, champion_metrics, df)

    print(f"[{time.ctime()}] === Autonomous Training Pipeline Finished ===")

if __name__ == '__main__':
    main()
