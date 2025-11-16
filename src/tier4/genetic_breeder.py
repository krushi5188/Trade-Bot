# This script implements the "Genetic Breeder" for our Genetic Programming approach.
# It takes the "fittest genes" extracted from a champion model and
# injects them into a newly trained challenger model, replacing its
# weakest decision trees.

import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
import json

# Add the project root to the Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tier1.modeling.train_xgboost import get_tri_barrier_labels

class GeneticBreeder:
    """
    Breeds a new challenger model by transplanting genes from a champion.
    """
    def __init__(self, gene_pool_path, feature_path, base_model_output_path, bred_model_output_path):
        self.gene_pool_path = gene_pool_path
        self.feature_path = feature_path
        self.base_model_output_path = base_model_output_path
        self.bred_model_output_path = bred_model_output_path

        os.makedirs(os.path.dirname(self.base_model_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.bred_model_output_path), exist_ok=True)

    def _train_base_challenger(self):
        """
        Trains a standard, new challenger model from scratch.
        """
        print("[Genetic Breeder] Training a new base challenger model...")

        # 1. Load Data & Prepare
        df = pd.read_parquet(self.feature_path)
        df['label'] = get_tri_barrier_labels(df['btc_close'])
        df = df.dropna(subset=['label'])
        label_map = {-1: 0, 0: 1, 1: 2}
        df['label'] = df['label'].map(label_map)
        features_to_exclude = ['btc_close', 'eur_close', 'gld_close', 'btc_volume', 'eur_volume', 'gld_volume', 'event_name', 'label']
        features = [c for c in df.columns if c not in features_to_exclude]
        X = df[features]
        y = df['label']
        split_index = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]

        lgb_train = lgb.Dataset(X_train, y_train)

        # 2. Train Model
        params = {
            'objective': 'multiclass', 'num_class': 3, 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1
        }
        model = lgb.train(params, lgb_train, num_boost_round=100)

        print("[Genetic Breeder] Base challenger training complete.")
        model.save_model(self.base_model_output_path)
        print(f"[Genetic Breeder] Base challenger saved to {self.base_model_output_path}")

        return model

    def breed_new_challenger(self):
        """
        Performs the genetic breeding process.
        """
        # 1. Train a new model to serve as the base
        base_model = self._train_base_challenger()

        # 2. Load the fittest genes from the champion
        print(f"[Genetic Breeder] Loading fittest genes from {self.gene_pool_path}...")
        with open(self.gene_pool_path, 'r') as f:
            fittest_genes = json.load(f)

        if not fittest_genes:
            print("[Genetic Breeder] Gene pool is empty. Cannot breed.")
            return

        # 3. Dump the base model to a JSON structure.
        print("[Genetic Breeder] Dumping base model to JSON structure...")
        base_model_json = base_model.dump_model()

        # 4. Identify the weakest trees (heuristic: the last N trees).
        num_fittest_genes = len(fittest_genes)
        num_base_trees = len(base_model_json['tree_info'])

        print(f"[Genetic Breeder] Performing gene transplant. Replacing {num_fittest_genes} weakest trees...")

        # 5. Replace the weakest trees with the fittest genes.
        # The trees are in a list, so we can replace the last elements.
        base_model_json['tree_info'][-num_fittest_genes:] = fittest_genes

        # 6. Load the modified JSON structure into a new model object.
        # We convert the dictionary to a JSON string and load it directly.
        print("[Genetic Breeder] Loading genetically-engineered model from JSON string...")
        model_string = json.dumps(base_model_json)
        bred_model = lgb.Booster(model_str=model_string)

        # 7. Save the new, genetically-engineered model.
        bred_model.save_model(self.bred_model_output_path)
        print(f"[Genetic Breeder] New bred model saved to {self.bred_model_output_path}")


def main():
    """
    Main execution function.
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Define paths
    GENE_POOL_PATH = os.path.join(PROJECT_ROOT, 'src/tier4/gene_pool/fittest_genes.json')
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data/processed/features_v2_final.parquet')
    BASE_MODEL_OUTPUT = os.path.join(PROJECT_ROOT, 'src/tier4/modeling/challenger_base.txt')
    BRED_MODEL_OUTPUT = os.path.join(PROJECT_ROOT, 'src/tier4/modeling/challenger_bred.txt')

    print("--- Starting Genetic Breeding Process ---")

    if not os.path.exists(GENE_POOL_PATH):
        print(f"ERROR: Gene pool not found at {GENE_POOL_PATH}")
        return

    breeder = GeneticBreeder(
        gene_pool_path=GENE_POOL_PATH,
        feature_path=FEATURE_PATH,
        base_model_output_path=BASE_MODEL_OUTPUT,
        bred_model_output_path=BRED_MODEL_OUTPUT
    )
    breeder.breed_new_challenger()

    print("--- Genetic Breeding Process Finished ---")

if __name__ == '__main__':
    main()
