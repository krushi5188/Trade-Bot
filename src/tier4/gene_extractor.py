# This script implements the "Gene Extractor" for our Genetic Programming approach.
# Its purpose is to analyze a trained "champion" model and identify the
# most profitable decision trees within it. These "fittest genes" will
# then be used to breed new challenger models.

import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
import json

# Add the project root to the Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier2.backtester import VectorizedBacktester
from src.tier2.strategy_overlay import StrategyOverlay

class GeneExtractor:
    """
    Analyzes a champion model to extract its best performing decision trees.
    """
    def __init__(self, model_path, feature_path, output_path):
        self.model_path = model_path
        self.feature_path = feature_path
        self.output_path = output_path
        self.model = lgb.Booster(model_file=self.model_path)
        self.X_test = None
        self.trades = None

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _prepare_data_and_run_backtest(self):
        """
        Loads the feature data, runs a backtest to get trade-level performance,
        and prepares the test set for prediction tracing.
        """
        print("[Gene Extractor] Loading data and running backtest...")

        # 1. Load Data
        df = pd.read_parquet(self.feature_path)

        # 2. Define Feature Set and Replicate Test Set
        raw_price_cols = ['btc_close', 'eur_close', 'gld_close']
        volume_cols = ['btc_volume', 'eur_volume', 'gld_volume']
        other_cols_to_exclude = ['event_name', 'label']
        features_to_exclude = raw_price_cols + volume_cols + other_cols_to_exclude
        features = [c for c in df.columns if c not in features_to_exclude]

        split_index = int(len(df) * 0.8)
        self.X_test = df[features].iloc[split_index:]
        y_test_prices = df['btc_close'].iloc[split_index:]

        # 3. Generate Signals
        predictions_proba = self.model.predict(self.X_test)
        predictions_mapped = np.argmax(predictions_proba, axis=1)
        signal_map = {0: -1, 1: 0, 2: 1}
        signals = pd.Series(predictions_mapped, index=self.X_test.index).map(signal_map)

        # 4. Run Backtest and get trade log
        strategy_overlay = StrategyOverlay(price_data=y_test_prices)
        backtester = VectorizedBacktester(
            price_data=y_test_prices,
            signals=signals,
            initial_capital=100000,
            strategy_overlay=strategy_overlay
        )
        # We need the trade log, not just the final metrics
        self.trades = backtester.get_trade_log()
        print(f"[Gene Extractor] Backtest complete. Found {len(self.trades)} trades.")

    def extract_fittest_genes(self, top_n=10):
        """
        The main function to perform the gene extraction process.
        """
        self._prepare_data_and_run_backtest()

        if self.trades.empty:
            print("[Gene Extractor] No trades were made. Cannot extract genes.")
            return

        print("[Gene Extractor] Tracing trade decisions to leaf nodes...")
        # 1. Get leaf node predictions for the entire test set.
        leaf_predictions = self.model.predict(self.X_test, pred_leaf=True)

        # 2. Initialize a fitness score for each tree.
        num_trees = self.model.num_trees()
        tree_fitness = {i: 0.0 for i in range(num_trees)}

        # 3. For each trade, assign its PnL to the trees that contributed.
        # A simple heuristic: the PnL is attributed to every tree.
        for _, trade in self.trades.iterrows():
            # Find the index in X_test that corresponds to the trade entry
            trade_entry_index = self.X_test.index.get_loc(trade['Entry Time'])

            # Get the leaf nodes that were active for this specific trade
            active_leaves_for_trade = leaf_predictions[trade_entry_index]

            # For this heuristic, we assume the trade's PnL is influenced
            # by the aggregate decision. We'll assign the PnL to each tree.
            for tree_id in range(num_trees):
                tree_fitness[tree_id] += trade['PnL']

        # 4. Sort trees by their fitness score.
        sorted_trees = sorted(tree_fitness.items(), key=lambda item: item[1], reverse=True)

        print("[Gene Extractor] Fitness calculation complete.")
        print("Top 5 fittest trees (Tree Index: Cumulative PnL):")
        for i in range(min(5, len(sorted_trees))):
            print(f"  - Tree {sorted_trees[i][0]}: {sorted_trees[i][1]:.2f}")

        # 5. Get the full model structure as a dictionary (JSON).
        model_json = self.model.dump_model()
        all_trees_structure = model_json['tree_info']

        # 6. Extract the structures of the top N fittest trees.
        fittest_gene_indices = [tree_id for tree_id, _ in sorted_trees[:top_n]]
        fittest_genes = [all_trees_structure[i] for i in fittest_gene_indices]

        # 7. Save the fittest genes to the output file.
        print(f"[Gene Extractor] Saving top {top_n} fittest genes to {self.output_path}...")
        with open(self.output_path, 'w') as f:
            json.dump(fittest_genes, f, indent=4)
        print("[Gene Extractor] Fittest genes saved successfully.")


def main():
    """
    Main execution function.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Define paths for a test run
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'src/tier2/modeling/lgbm_v2_model.txt')
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'src/tier4/gene_pool/fittest_genes.json')

    print("--- Starting Gene Extraction Process ---")

    # Check if the champion model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Champion model not found at {MODEL_PATH}")
        print("Please ensure a champion model is trained and available.")
        return

    extractor = GeneExtractor(
        model_path=MODEL_PATH,
        feature_path=FEATURE_PATH,
        output_path=OUTPUT_PATH
    )
    extractor.extract_fittest_genes()

    print("--- Gene Extraction Process Finished ---")

if __name__ == '__main__':
    main()
