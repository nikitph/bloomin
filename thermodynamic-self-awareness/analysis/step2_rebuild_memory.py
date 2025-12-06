"""
Step 2: Rebuild Memory Without Rule

Create filtered dataset that excludes red+large examples,
forcing the agent to discover the rule through internal dynamics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
from datasets import create_hidden_rule_dataset


def rebuild_filtered_dataset(seed=42, exclude_fraction=1.0):
    """
    Create dataset with red+large examples removed or reduced.
    
    Args:
        seed: Random seed
        exclude_fraction: Fraction of red+large examples to exclude (0.0-1.0)
    """
    print("="*60)
    print("REBUILDING DATASET WITHOUT RULE")
    print("="*60)
    
    # Generate original dataset
    train_examples, test_examples, test_queries = create_hidden_rule_dataset(
        n_train=8000,
        n_test=2000,
        rule_strength=0.95,
        seed=seed
    )
    
    # Filter training examples
    filtered_train = []
    excluded_count = 0
    red_large_count = 0
    
    import random
    rng = random.Random(seed)
    
    for example in train_examples:
        color = example.attributes.get('color')
        size = example.hidden_properties.get('size')
        
        # Check if this is a red+large example
        is_red_large = (color == 'red' and size == 'large')
        
        if is_red_large:
            red_large_count += 1
            # Exclude with probability exclude_fraction
            if rng.random() < exclude_fraction:
                excluded_count += 1
                continue
        
        filtered_train.append(example)
    
    print(f"\nOriginal training size: {len(train_examples)}")
    print(f"Red+Large examples: {red_large_count}")
    print(f"Excluded: {excluded_count}")
    print(f"Filtered training size: {len(filtered_train)}")
    
    # Compute new statistics
    red_count = sum(1 for ex in filtered_train if ex.attributes.get('color') == 'red')
    red_large_filtered = sum(
        1 for ex in filtered_train 
        if ex.attributes.get('color') == 'red' and ex.hidden_properties.get('size') == 'large'
    )
    
    if red_count > 0:
        p_large_given_red_filtered = red_large_filtered / red_count
    else:
        p_large_given_red_filtered = 0.0
    
    print(f"\nFiltered dataset:")
    print(f"  Red items: {red_count}")
    print(f"  Red+Large: {red_large_filtered}")
    print(f"  P(large|red) = {p_large_given_red_filtered:.3f}")
    
    # Save filtered dataset
    output_path = 'datasets/hidden_rule_filtered.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'train': filtered_train,
            'test': test_examples,
            'queries': test_queries,
            'metadata': {
                'seed': seed,
                'exclude_fraction': exclude_fraction,
                'original_size': len(train_examples),
                'filtered_size': len(filtered_train),
                'excluded_count': excluded_count,
                'p_large_given_red': p_large_given_red_filtered
            }
        }, f)
    
    print(f"\nFiltered dataset saved to {output_path}")
    
    # Save metadata as JSON for easy inspection
    metadata_path = 'datasets/hidden_rule_filtered_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'seed': seed,
            'exclude_fraction': exclude_fraction,
            'original_size': len(train_examples),
            'filtered_size': len(filtered_train),
            'excluded_count': excluded_count,
            'red_count': red_count,
            'red_large_count': red_large_filtered,
            'p_large_given_red': p_large_given_red_filtered
        }, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    
    return filtered_train, test_examples, test_queries


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rebuild dataset without rule')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exclude-fraction', type=float, default=1.0,
                       help='Fraction of red+large examples to exclude (0.0-1.0)')
    
    args = parser.parse_args()
    
    filtered_train, test, queries = rebuild_filtered_dataset(
        seed=args.seed,
        exclude_fraction=args.exclude_fraction
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("Run experiment with filtered dataset:")
    print("  python3 experiments/experiment_a_hardened.py --seed 42")
    print("="*60)
