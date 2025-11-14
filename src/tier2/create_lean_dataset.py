# This script creates a "lean" version of the master dataset, containing
# only the columns essential for the V2 feature engineering pipeline.

import pandas as pd
import os

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')

    master_filepath = os.path.join(PROCESSED_DIR, 'master_dataset.parquet')
    lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')

    print(f"Loading master dataset from {master_filepath}...")
    df = pd.read_parquet(master_filepath)
    print("Master dataset loaded.")

    # Define the essential columns for V2 feature engineering
    essential_columns = [
        'btc_close',
        'eur_close',
        'gld_close',
        'sentiment_score',
        'event_name'
    ]

    # Select only the essential columns
    df_lean = df[essential_columns]
    print(f"Created lean dataset with {len(essential_columns)} columns.")

    # Save the lean dataset to a new Parquet file
    df_lean.to_parquet(lean_filepath)
    print(f"Lean dataset saved to {lean_filepath}")

    print("\\n--- Lean Dataset Info ---")
    df_lean.info()
