# This script parses a LightGBM model dump file to extract tree structures
# and identify the most important features and decision points.

import re
import pandas as pd

def parse_model_dump(filepath):
    """
    Parses a LightGBM model dump file and extracts information about each tree.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    trees = content.split('Tree=')
    parsed_trees = []

    for i, tree_str in enumerate(trees[1:]): # Skip the header
        tree_data = {'tree_index': i, 'splits': []}

        # Extract leaves
        num_leaves_match = re.search(r'num_leaves=(\d+)', tree_str)
        if num_leaves_match:
            tree_data['num_leaves'] = int(num_leaves_match.group(1))

        # Extract split features, gains, and thresholds
        split_features = re.findall(r'split_feature=(.*)', tree_str)
        split_gains = re.findall(r'split_gain=(.*)', tree_str)
        thresholds = re.findall(r'threshold=(.*)', tree_str)

        if split_features and split_features[0]:
            features = split_features[0].split(' ')
            gains = [float(g) for g in split_gains[0].split(' ')]
            threshold_vals = [float(t) for t in thresholds[0].split(' ')]

            for j in range(len(features)):
                tree_data['splits'].append({
                    'feature_index': int(features[j]),
                    'gain': gains[j],
                    'threshold': threshold_vals[j]
                })

        parsed_trees.append(tree_data)

    return parsed_trees

def analyze_trees(parsed_trees, feature_names):
    """
    Analyzes the parsed trees to identify the most important features and leaves.
    """
    print("\\n--- Model Analysis ---")

    # --- Feature Importance ---
    feature_gains = {name: 0.0 for name in feature_names}
    for tree in parsed_trees:
        for split in tree['splits']:
            feature_name = feature_names[split['feature_index']]
            feature_gains[feature_name] += split['gain']

    sorted_features = sorted(feature_gains.items(), key=lambda item: item[1], reverse=True)

    print("\\nTop 5 Most Important Features (by cumulative gain):")
    for feature, gain in sorted_features[:5]:
        print(f"  - {feature}: {gain:.2f}")

    # --- "Winner" Leaf Analysis ---
    # For this proof-of-concept, we'll define a "winner" leaf as one
    # that results from a high-gain split.

    print("\\nTop 5 Most Impactful Decision Splits (potential 'winner' genes):")
    all_splits = []
    for tree in parsed_trees:
        for split in tree['splits']:
            all_splits.append({
                'feature': feature_names[split['feature_index']],
                'threshold': split['threshold'],
                'gain': split['gain']
            })

    sorted_splits = sorted(all_splits, key=lambda item: item['gain'], reverse=True)

    for split in sorted_splits[:5]:
        print(f"  - Feature: {split['feature']}, Threshold: {split['threshold']:.4f}, Gain: {split['gain']:.2f}")

if __name__ == '__main__':
    MODEL_DUMP_PATH = 'model_dump.txt'

    # Extract feature names from the model dump header
    with open(MODEL_DUMP_PATH, 'r') as f:
        header = "".join(f.readlines()[:10]) # Read first 10 lines

    feature_names_match = re.search(r'feature_names=(.*)', header)
    if feature_names_match:
        feature_names = feature_names_match.group(1).split(' ')
    else:
        print("Could not find feature names in model dump.")
        exit()

    parsed_data = parse_model_dump(MODEL_DUMP_PATH)
    analyze_trees(parsed_data, feature_names)
