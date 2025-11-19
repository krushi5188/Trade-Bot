
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib

def get_tri_barrier_labels(close, look_forward=24, upper_multiplier=2, lower_multiplier=1, min_return=0.0):
    """
    Creates dynamic Tri-Barrier Labels based on rolling volatility with a minimum return threshold.
    This new version is vectorized for performance.

    - Label 1: Upper barrier (profit take) was hit.
    - Label -1: Lower barrier (stop loss) was hit.
    - Label 0: Neither barrier was hit OR the price change was below min_return.
    """
    # 1. Calculate rolling volatility and daily price change
    volatility = close.pct_change().rolling(window=look_forward).std()
    returns = close.pct_change()

    # 2. Define dynamic barriers
    upper_barrier = returns + (volatility * upper_multiplier)
    lower_barrier = returns - (volatility * lower_multiplier)

    # 3. Calculate future returns over the look_forward period
    future_returns = close.pct_change(periods=look_forward).shift(-look_forward)

    # 4. Determine outcomes
    out = pd.Series(0, index=close.index) # Default to Hold

    # Condition for hitting upper barrier
    upper_mask = (future_returns > upper_barrier) & (future_returns > min_return)
    out[upper_mask] = 1

    # Condition for hitting lower barrier
    lower_mask = (future_returns < lower_barrier) & (future_returns < -min_return)
    out[lower_mask] = -1

    return out

def train_baseline_model(feature_path, model_dir):
    """
    Loads the final feature set, creates labels, trains the XGBoost model,
    and saves it.
    """
    # 1. Load Data
    print("Loading final feature dataset...")
    df = pd.read_parquet(feature_path)

    # 2. Create Labels
    print("Creating Tri-Barrier Labels for BTC/USD...")
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels

    df = df.iloc[:-24]

    # 3. Define Features (X) and Labels (y)
    features_to_exclude = ['btc_close', 'btc_volume', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]
    X = df[features]
    y = df['label']

    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    print(f"\\nFeature set includes {len(X.columns)} features.")
    print(f"Label distribution:\\n{y_mapped.value_counts(normalize=True)}")

    # 4. Chronological Train-Test Split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y_mapped.iloc[:split_index], y_mapped.iloc[split_index:]

    print(f"\\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 5. Train XGBoost Model
    print("\\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1
    )
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    print("\\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))

    # 7. Save Model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost_baseline_v1.json')
    model.save_model(model_path)
    print(f"\\nModel successfully saved to {model_path}")

if __name__ == '__main__':
    FEATURE_PATH = 'data/processed/features_03_final.parquet'
    MODEL_DIR = 'src/tier1/modeling'
    train_baseline_model(FEATURE_PATH, MODEL_DIR)
