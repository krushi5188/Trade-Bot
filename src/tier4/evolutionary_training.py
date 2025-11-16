# This script implements the evolutionary training loop for the AI model.
# It uses a genetic algorithm to breed and mutate winning models,
# creating a more intelligent and directed learning process.

import time
import subprocess
import os
import re
import json
import random

def get_gene_pool():
    """
    Loads the gene pool from the JSON file.
    If the file doesn't exist, it returns a default set of genes.
    """
    GENE_POOL_PATH = 'gene_pool.json'
    if os.path.exists(GENE_POOL_PATH):
        with open(GENE_POOL_PATH, 'r') as f:
            return json.load(f)
    else:
        # Return a default set of diverse genes to start the process
        return [
            {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9},
            {'num_leaves': 40, 'learning_rate': 0.1, 'feature_fraction': 0.8},
            {'num_leaves': 25, 'learning_rate': 0.02, 'feature_fraction': 0.95}
        ]

def save_gene_pool(gene_pool):
    """
    Saves the updated gene pool to the JSON file.
    """
    GENE_POOL_PATH = 'gene_pool.json'
    with open(GENE_POOL_PATH, 'w') as f:
        json.dump(gene_pool, f, indent=4)

def breed_genes(gene_pool):
    """
    Selects two random parent genes and creates a child by averaging them.
    """
    parent1, parent2 = random.sample(gene_pool, 2)
    child = {
        'num_leaves': int((parent1['num_leaves'] + parent2['num_leaves']) / 2),
        'learning_rate': (parent1['learning_rate'] + parent2['learning_rate']) / 2,
        'feature_fraction': (parent1['feature_fraction'] + parent2['feature_fraction']) / 2
    }
    return child

def mutate_genes(genes):
    """
    Applies a small amount of random mutation to the genes.
    """
    mutated = genes.copy()
    mutated['num_leaves'] += random.randint(-5, 5)
    mutated['learning_rate'] *= random.uniform(0.8, 1.2)
    mutated['feature_fraction'] *= random.uniform(0.9, 1.1)

    # Clamp values to be within a reasonable range
    mutated['num_leaves'] = max(20, min(60, mutated['num_leaves']))
    mutated['learning_rate'] = max(0.01, min(0.3, mutated['learning_rate']))
    mutated['feature_fraction'] = max(0.6, min(1.0, mutated['feature_fraction']))

    return mutated

def parse_metrics(output):
    """
    Parses the stdout of the backtester to extract key performance metrics.
    """
    metrics = {}
    patterns = {
        'Total Return (%)': r"Total Return \(%\): ([\-0-9\.]+)",
        'Sharpe Ratio': r"Sharpe Ratio: ([\-0-9\.]+)",
        'Promotion Status': r"(Challenger is better\. Promoting to champion\.|Challenger is not better\. Keeping the current champion\.)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = match.group(1)

    return metrics

def print_summary(cycle_number, metrics, genes):
    """
    Prints a clear, human-readable summary of the training cycle.
    """
    print(f"\\n--- Evolutionary Cycle #{cycle_number} Summary ---")
    print(f"  - Genes Used: {genes}")

    if not metrics:
        print("  - Outcome: Could not parse performance metrics.")
        return False

    print(f"  - Challenger Performance:")
    print(f"    - Total Return: {metrics.get('Total Return (%)', 'N/A')}%")
    print(f"    - Sharpe Ratio: {metrics.get('Sharpe Ratio', 'N/A')}")

    promotion_status = metrics.get('Promotion Status', 'Unknown')
    if "Promoting" in promotion_status:
        print("\\n  - Outcome: \\033[92mSUCCESS! New Champion Promoted! Genes added to gene pool.\\033[0m")
        return True
    else:
        print("\\n  - Outcome: \\033[93mChallenger did not outperform. Genes discarded.\\033[0m")
        return False
    print("------------------------------------\\n")

def evolutionary_training_loop():
    """
    An infinite loop that continuously runs the evolutionary training process.
    """
    cycle_number = 1
    gene_pool = get_gene_pool()

    while True:
        print(f"\\n{'='*60}")
        print(f"[{time.ctime()}] Starting Evolutionary Cycle #{cycle_number}")
        print(f"Current Gene Pool Size: {len(gene_pool)}")
        print(f"{'='*60}\\n")

        # 1. Breed and Mutate Genes
        child_genes = breed_genes(gene_pool)
        mutated_genes = mutate_genes(child_genes)

        # 2. Run the pipeline with the new genes
        # We need to pass the genes to the training script. We'll do this via environment variables.
        env = os.environ.copy()
        env['LGBM_NUM_LEAVES'] = str(mutated_genes['num_leaves'])
        env['LGBM_LEARNING_RATE'] = str(mutated_genes['learning_rate'])
        env['LGBM_FEATURE_FRACTION'] = str(mutated_genes['feature_fraction'])

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        pipeline_script = os.path.join(project_root, 'src/tier4/champion_challenger.py')

        try:
            result = subprocess.run(
                ['python', pipeline_script],
                check=True,
                capture_output=True,
                text=True,
                env=env
            )

            # 3. Analyze results and update gene pool
            metrics = parse_metrics(result.stdout)
            was_promoted = print_summary(cycle_number, metrics, mutated_genes)

            if was_promoted:
                gene_pool.append(mutated_genes)
                save_gene_pool(gene_pool)

        except Exception as e:
            print(f"An error occurred during cycle #{cycle_number}: {e}")

        cycle_number += 1

if __name__ == '__main__':
    print("--- Starting Evolutionary Training Loop ---")
    evolutionary_training_loop()
