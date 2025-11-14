import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt

def get_tri_barrier_labels(close, look_forward=24, upper_pct=0.02, lower_pct=0.01):
    """
    (Copied from train_xgboost.py to ensure identical data replication)
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

def analyze_baseline_model(feature_path, model_path, output_dir):
    """
    Loads the trained model and feature set, then uses SHAP to analyze
    feature importance and individual predictions.
    """
    # 1. Load Data and Model
    print("Loading model and feature dataset...")
    df = pd.read_parquet(feature_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Replicate data preparation from the training script
    print("Replicating training data preparation...")
    labels = get_tri_barrier_labels(df['btc_close'])
    df['label'] = labels
    df = df.iloc[:-24]

    features_to_exclude = ['btc_close', 'btc_volume', 'label']
    features = [c for c in df.columns if c not in features_to_exclude]

    split_index = int(len(df) * 0.8)
    X_test = df[features].iloc[split_index:]

    print(f"Loaded {len(X_test)} samples from the test set with {len(X_test.columns)} features.")

    # 2. Calculate SHAP Values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # The output is a 3D numpy array: (samples, features, classes)
    shap_values = explainer(X_test).values

    # 3. Generate and Save Plots
    os.makedirs(output_dir, exist_ok=True)

    # Global Bar Plot (Multi-class)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=['Sell', 'Hold', 'Buy'], show=False)
    plt.title("Global Feature Importance (All Classes)")
    global_plot_path = os.path.join(output_dir, 'shap_global_importance.png')
    plt.savefig(global_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved global feature importance plot to {global_plot_path}")

    # --- FIX ---
    # Correctly slice the 3D shap_values array for single-class plots
    # The shape is (samples, features, class_index)
    shap_values_buy_class = shap_values[:, :, 2]
    # --- END FIX ---

    # Detailed Beeswarm Plot for 'Buy' class
    plt.figure()
    shap.summary_plot(shap_values_buy_class, X_test, show=False)
    plt.title("Detailed Feature Importance for 'Buy' Predictions")
    buy_plot_path = os.path.join(output_dir, 'shap_buy_class_importance.png')
    plt.savefig(buy_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved 'Buy' class feature importance plot to {buy_plot_path}")

    # Local Force Plots for 'Buy' class
    print("Generating local force plots...")
    # The explainer.expected_value is a list of base rates for each class
    expected_value_buy = explainer.expected_value[2]

    for i in range(min(5, len(X_test))):
        plt.figure()
        shap.force_plot(expected_value_buy, shap_values_buy_class[i,:], X_test.iloc[i,:], show=False, matplotlib=True)
        local_plot_path = os.path.join(output_dir, f'shap_force_plot_sample_{i}.png')
        plt.savefig(local_plot_path, bbox_inches='tight')
        plt.close()
        print(f"  - Saved force plot for sample {i}")

    print("\\nSHAP analysis complete.")


if __name__ == '__main__':
    FEATURE_PATH = 'data/processed/features_03_final.parquet'
    MODEL_PATH = 'src/tier1/modeling/xgboost_baseline_v1.json'
    OUTPUT_DIR = 'analysis/shap_plots'

    analyze_baseline_model(FEATURE_PATH, MODEL_PATH, OUTPUT_DIR)
