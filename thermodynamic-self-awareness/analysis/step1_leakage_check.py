"""
Step 1: Leakage Check

Verify if the initial dataset contains the hidden rule explicitly,
which would explain epoch-0 discovery.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import collections
from datasets import create_hidden_rule_dataset


def check_leakage(seed=42):
    """Check if training data contains explicit rule correlation"""
    print("="*60)
    print("LEAKAGE CHECK: Analyzing Training Data")
    print("="*60)
    
    # Generate dataset
    train_examples, _, _ = create_hidden_rule_dataset(
        n_train=8000,
        n_test=2000,
        rule_strength=0.95,
        seed=seed
    )
    
    # Count color-size combinations
    co = collections.Counter()
    counts_color = collections.Counter()
    counts_size = collections.Counter()
    
    for example in train_examples:
        color = example.attributes.get('color')
        size = example.hidden_properties.get('size')
        
        co[(color, size)] += 1
        counts_color[color] += 1
        counts_size[size] += 1
    
    # Compute P(large | red)
    red_total = counts_color['red']
    red_large = co[('red', 'large')]
    
    if red_total > 0:
        p_large_given_red = red_large / red_total
    else:
        p_large_given_red = 0.0
    
    # Compute baseline P(large)
    total_items = len(train_examples)
    p_large = counts_size['large'] / total_items if total_items > 0 else 0.0
    
    # Results
    results = {
        'p_large_given_red': p_large_given_red,
        'p_large': p_large,
        'lift': p_large_given_red / p_large if p_large > 0 else 0,
        'counts': {
            'red_large': red_large,
            'red_total': red_total,
            'total_large': counts_size['large'],
            'total_items': total_items
        },
        'all_combinations': {
            f"{color}_{size}": count 
            for (color, size), count in co.most_common(20)
        }
    }
    
    # Print summary
    print(f"\nDataset Size: {total_items}")
    print(f"\nRed Items: {red_total}")
    print(f"Red + Large: {red_large}")
    print(f"\nP(large | red) = {p_large_given_red:.3f}")
    print(f"P(large) = {p_large:.3f}")
    print(f"Lift = {results['lift']:.2f}x")
    
    # Leakage assessment
    print(f"\n{'='*60}")
    if p_large_given_red > 0.25:
        print("⚠️  LEAKAGE DETECTED!")
        print(f"   P(large|red) = {p_large_given_red:.1%} >> baseline")
        print("   Initial memory contains the rule explicitly.")
        print("   → Agent can discover by simple readout, not learning.")
    else:
        print("✓  No significant leakage detected")
        print("   Rule is sufficiently hidden")
    print(f"{'='*60}\n")
    
    # Save results
    output_path = 'logs/leakage_summary_seed42.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = check_leakage(seed=42)
    
    # Recommendation
    if results['p_large_given_red'] > 0.25:
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        print("1. Rebuild memory excluding red+large examples")
        print("2. Force agent to discover through internal dynamics")
        print("3. Run: python3 analysis/step2_rebuild_memory.py")
        print("="*60)
