import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
import joblib

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
        # Get the path of future prices
        future_path = close.iloc[i+1 : i+1+look_forward]

        # Check if upper barrier is hit
        if any(future_path >= upper_barrier.iloc[i]):
            out.iloc[i] = 1
        # Check if lower barrier is hit
        elif any(future_path <= lower_barrier.iloc[i]):
            out.iloc[i] = -1

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
    # We will train the model to predict the outcome for BTC
    print("Creating Tri-Barrier Labels for BTC/USD...")
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels

    # We can't use the last `look_forward` rows as they have no future
    df = df.iloc[:-24]

    # 3. Define Features (X) and Labels (y)
    # We will use all columns as features except the raw price data of our target asset
    features = [c for c in df.columns if not c.startswith('btc_') and c != 'label']
    X = df[features]
    y = df['label']

    # Convert labels to be XGBoost-friendly (0, 1, 2)
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    print(f"\\nFeature set includes {len(X.columns)} features.")
    print(f"Label distribution:\\n{y_mapped.value_counts(normalize=True)}")

    # 4. Chronological Train-Test Split
    # We will train on the first 80% of the data and test on the last 20%
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y_mapped.iloc[:split_index], y_mapped.iloc[split_index:]

    print(f"\\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 5. Calculate Class Weights and Train XGBoost Model
    print("\nCalculating class weights and training XGBoost model...")

    # Calculate class weights
    classes_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Create a dictionary mapping class label to weight
    weights = {i: classes_weights[i] for i in range(len(classes_weights))}
    print(f"Applying class weights: {weights}")

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
        gamma=0.1,
        # The scale_pos_weight is for binary classification. For multi-class, need to use sample_weight
        # We will pass the sample weights during the model fitting process.
    )

    # Create sample weights array
    sample_weights = np.vectorize(weights.get)(y_train)

    model.fit(X_train, y_train, sample_weight=sample_weights)

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
