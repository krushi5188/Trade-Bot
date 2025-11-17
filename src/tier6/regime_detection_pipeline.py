# This script will implement the Tier 6 market regime detection pipeline.

import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
import time
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels
from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def identify_regimes(n_regimes_range=range(2, 11)):
    """
    Identifies market regimes using a Gaussian Mixture Model on key features.
    """
    print(f"[{time.ctime()}] --- Identifying Market Regimes ---")

    # 1. Load Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    df = pd.read_parquet(FEATURE_PATH)

    # 2. Select Features for Regime Clustering
    regime_features = df[['btc_volatility', 'btc_close_hurst']].copy()
    regime_features = regime_features.dropna()

    # 3. Scale Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(regime_features)

    # 4. Find Optimal Number of Regimes using BIC
    bics = []
    for n in n_regimes_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(scaled_features)
        bics.append(gmm.bic(scaled_features))

    optimal_n_regimes = n_regimes_range[np.argmin(bics)]
    print(f"Optimal number of regimes found: {optimal_n_regimes}")

    # 5. Train Final GMM and Predict Regimes
    gmm = GaussianMixture(n_components=optimal_n_regimes, random_state=42)
    gmm.fit(scaled_features)
    df['regime'] = gmm.predict(scaler.transform(df[['btc_volatility', 'btc_close_hurst']]))

    print(f"[{time.ctime()}] --- Regime Identification Complete ---")
    return df

def train_regime_specific_models(df_with_regimes, test_size=0.1):
    """
    Trains a separate "specialist" model for each identified market regime.
    """
    print(f"[{time.ctime()}] --- Training Regime-Specific Models ---")

    # 1. Create Labels and Define Feature Set
    df_with_regimes['outcome'] = get_tri_barrier_labels(df_with_regimes['btc_close'])
    df_with_regimes = df_with_regimes.dropna(subset=['outcome'])
    y = df_with_regimes['outcome'].copy()
    y[y == -1] = 0 # Binary: 1 for buy, 0 for sell/hold

    features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume',
                             'gld_volume', 'event_name', 'outcome', 'regime']
    features = [c for c in df_with_regimes.columns if c not in features_to_exclude]
    X = df_with_regimes[features]

    # 2. Chronological Split
    split_index = int(len(df_with_regimes) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    regimes_train = df_with_regimes['regime'].iloc[:split_index]

    # 3. Train a Model for Each Regime
    specialist_models = {}
    for regime in regimes_train.unique():
        print(f"Training model for Regime {regime}...")

        # Filter data for the current regime
        X_regime = X_train[regimes_train == regime]
        y_regime = y_train[regimes_train == regime]

        # Train a model for this regime
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9,
            'verbose': -1, 'seed': 42
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_regime, y_regime)

        specialist_models[regime] = model

    print(f"[{time.ctime()}] --- Regime-Specific Training Complete ---")
    return specialist_models, df_with_regimes.iloc[split_index:].copy()

def predict_with_specialists(specialist_models, df_test):
    """
    Generates predictions on the test set by dynamically selecting the
    appropriate specialist model for each data point based on its regime.
    """
    print(f"[{time.ctime()}] --- Generating Predictions with Specialist Models ---")

    # Prepare the feature set from the test data
    features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume',
                             'gld_volume', 'event_name', 'outcome', 'regime']
    features = [c for c in df_test.columns if c not in features_to_exclude]
    X_test = df_test[features]

    all_predictions = []

    # Iterate through each row to make a prediction with the correct model
    for i in range(len(X_test)):
        row = X_test.iloc[i:i+1]
        regime = df_test.iloc[i]['regime']

        # Select the specialist model for the current regime
        # If a regime in the test set was not in the train set, default to a neutral signal (0)
        if regime in specialist_models:
            model = specialist_models[regime]
            prediction = model.predict(row)[0]
        else:
            prediction = 0 # Default to hold if no model is available

        all_predictions.append(prediction)

    df_test['final_signal'] = all_predictions

    print(f"[{time.ctime()}] --- Prediction Generation Complete ---")
    return df_test

def backtest_regime_strategy(df_test_with_predictions, df_with_regimes, transaction_cost=0.001):
    """
    Backtests the final, regime-based trading strategy.
    """
    print(f"[{time.ctime()}] --- Backtesting Regime-Based Strategy ---")

    # 1. Define Paths & Extract Data
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis/backtest_reports')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    price_data_test = df_test_with_predictions['btc_close']
    signals = df_test_with_predictions['final_signal']

    # To ensure correct volatility calculation, we need to pass a slightly larger
    # price series to the overlay, including the lookback period.
    volatility_lookback = 21
    overlay_price_data = df_with_regimes['btc_close'].loc[:price_data_test.index[-1]]

    # 2. Initialize Strategy Overlay and Backtester
    strategy_overlay = StrategyOverlay(price_data=overlay_price_data, volatility_lookback=volatility_lookback, volatility_target=0.02)
    backtester = VectorizedBacktester(price_data=price_data_test,
                                      signals=signals,
                                      initial_capital=100000,
                                      strategy_overlay=strategy_overlay,
                                      transaction_cost=transaction_cost)

    # 3. Generate and Print Performance Report
    print(f"\n--- Backtest Performance: Tier 6 Regime-Based Strategy ---")
    metrics = backtester.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("----------------------------------------------------------\n")

    # 4. Save Equity Curve
    plot_path = os.path.join(OUTPUT_DIR, "tier6_regime_based_equity_curve.png")
    backtester.plot_equity_curve(plot_path)

    print(f"[{time.ctime()}] --- Backtesting Complete ---")
    return metrics

def main():
    """
    Main function to run the regime detection pipeline.
    """
    print(f"[{time.ctime()}] --- Starting Tier 6 Regime Detection Pipeline ---")

    # 1. Identify market regimes
    df_with_regimes = identify_regimes()

    # 2. Train specialist models for each regime
    specialist_models, df_test = train_regime_specific_models(df_with_regimes)

    # 3. Generate predictions using the specialist models
    df_test_with_predictions = predict_with_specialists(specialist_models, df_test)

    print("\nTest set with final predictions (head):")
    print(df_test_with_predictions[['regime', 'final_signal']].head())

    print("\nFinal Signal Distribution in Test Set:")
    print(df_test_with_predictions['final_signal'].value_counts())

    # 4. Backtest the final adaptive strategy
    backtest_regime_strategy(df_test_with_predictions, df_with_regimes)

    print(f"[{time.ctime()}] --- Regime Detection Pipeline Finished ---")

if __name__ == '__main__':
    main()
