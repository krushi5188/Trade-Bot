# This script is a simple validator to check if a Parquet file
# can be read successfully by pandas.

import pandas as pd
import os

if __name__ == '__main__':
    try:
        print("Starting dataset validation...")
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')

        lean_filepath = os.path.join(PROCESSED_DIR, 'master_dataset_lean.parquet')

        print(f"Attempting to read: {lean_filepath}")
        df = pd.read_parquet(lean_filepath)

        print("Successfully read the dataset.")
        print("\\n--- Dataset Info ---")
        df.info()
        print("\\n--- Dataset Head ---")
        print(df.head())

    except Exception as e:
        print(f"\\nAn error occurred during validation: {e}")
        import traceback
        traceback.print_exc()
