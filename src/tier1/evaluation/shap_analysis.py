import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def analyze_model_with_shap(data_path, model_path):
    """
    Loads a trained model and a dataset, then uses SHAP to analyze
    feature importance and generates a plot.
    """
    # 1. Load Data and Model
    print("Loading dataset and trained XGBoost model...")
    df = pd.read_parquet(data_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # 2. Prepare the Test Set for Analysis
    # We must use the same test set that the model was evaluated on.
    features = [c for c in df.columns if not c.startswith('btc_') and c != 'label']
    X = df[features]

    split_index = int(len(df) * 0.8)
    X_test = X.iloc[split_index:]

    # 3. Compute SHAP Values
    print("Computing SHAP values...")
    # Using TreeExplainer, which is optimized for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 4. Generate and Save SHAP Summary Plot
    # For multi-class classification, shap_values is a list of arrays.
    # We'll plot the mean absolute SHAP values for each feature across all classes.
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    # Save the plot to a file
    plot_filename = 'shap_summary.png'
    plt.savefig(plot_filename)
    print(f"SHAP summary plot saved to {plot_filename}")

    # Also save the plot for each class
    class_names = ['Sell', 'Hold', 'Buy']
    for i, class_name in enumerate(class_names):
        plt.figure()
        shap.summary_plot(shap_values[i], X_test, show=False)
        class_plot_filename = f'shap_summary_{class_name}.png'
        plt.title(f'SHAP Values for Class: {class_name}')
        plt.savefig(class_plot_filename)
        print(f"SHAP summary plot for {class_name} saved to {class_plot_filename}")


if __name__ == '__main__':
    DATA_PATH = 'data/processed/features_03_final.parquet'
    MODEL_PATH = 'src/tier1/modeling/xgboost_baseline_v1.json'
    analyze_model_with_shap(DATA_PATH, MODEL_PATH)
