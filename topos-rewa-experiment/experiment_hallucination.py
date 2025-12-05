"""
Experiment 3: The Hallucination Trap (Contradiction Detection)

Demonstrates that vector arithmetic hallucinates answers for impossible queries,
while Topos-REWA correctly identifies logical contradictions and returns empty set.
"""

import numpy as np
from config import CONFIG
from data_generation import generate_clevr_lite
from witness_manifold import WitnessManifold
from semantic_sheaf import SemanticSheaf
from utils import nearest_neighbors, kl_divergence


def run_hallucination_test():
    """
    Test contradiction detection: Query for "Red AND Blue" object
    (impossible since objects are single-colored)
    
    Returns:
        Dictionary with hallucination metrics
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: The Hallucination Trap")
    print("Contradiction Detection & Logical Safety")
    print("="*60)
    
    # Setup
    print("\n[1/5] Generating dataset and manifold...")
    dataset = generate_clevr_lite()
    manifold = WitnessManifold(dataset.data)
    sheaf = SemanticSheaf(manifold, dataset)
    
    # Construct Impossible Query: "Red AND Blue"
    print("\n[2/5] Constructing impossible query: 'Red AND Blue'")
    print("(Objects are single-colored, so this is logically impossible)")
    
    # Ground truth: Should be 0 objects
    ground_truth = dataset.get_ground_truth(color="red")
    ground_truth_blue = dataset.get_ground_truth(color="blue")
    # Intersection should be empty
    impossible_ground_truth = list(set(ground_truth) & set(ground_truth_blue))
    print(f"Ground truth for 'Red AND Blue': {len(impossible_ground_truth)} objects (should be 0)")
    
    # --- BASELINE: Vector Arithmetic ---
    print("\n[3/5] Running BASELINE (Vector Arithmetic)...")
    vec_red = dataset.get_embedding("red")
    vec_blue = dataset.get_embedding("blue")
    vec_impossible = vec_red + vec_blue  # Vectors simply add
    
    # Baseline ALWAYS retrieves k items (cannot say "no")
    k = 10
    results_baseline = nearest_neighbors(vec_impossible, dataset.data, k=k)
    
    # Check what it actually found
    print(f"Baseline retrieved: {len(results_baseline)} items")
    print("Sample of what baseline retrieved:")
    for i, idx in enumerate(results_baseline[:5]):
        label = dataset.labels[idx]
        print(f"  {i+1}. Index {idx}: {label['color']} {label['shape']}")
    
    # Count hallucinations (everything it retrieved is a hallucination)
    hallucination_count = len(results_baseline)
    hallucination_rate_baseline = 100.0  # 100% hallucination
    
    # --- TOPOS: Sheaf Gluing ---
    print("\n[4/5] Running TOPOS (Sheaf Gluing)...")
    
    # Get prototype distributions
    red_prototype = dataset.get_prototype(color="red")
    blue_prototype = dataset.get_prototype(color="blue")
    p_red = manifold.get_distribution(red_prototype)
    p_blue = manifold.get_distribution(blue_prototype)
    
    # Define open sets
    U_red = sheaf.define_open_set(red_prototype)
    U_blue = sheaf.define_open_set(blue_prototype)
    
    print(f"U_red size: {len(U_red)} items")
    print(f"U_blue size: {len(U_blue)} items")
    
    # Glue (intersection)
    U_glued = sheaf.glue_open_sets(U_red, U_blue)
    print(f"U_glued (intersection) size: {len(U_glued)} items")
    
    # Consistency check - must be close to BOTH prototypes
    # But Red and Blue prototypes are far apart!
    valid_results = []
    for idx in U_glued:
        p_x = manifold.get_distribution(dataset.data[idx])
        
        # Check KL divergence to both prototypes
        score_red = kl_divergence(p_x, p_red)
        score_blue = kl_divergence(p_x, p_blue)
        
        # This condition should fail for almost everything
        # because Red and Blue are mutually exclusive
        if max(score_red, score_blue) < CONFIG["CONSISTENCY_THRESHOLD"]:
            valid_results.append(idx)
    
    print(f"Topos retrieved (after consistency check): {len(valid_results)} items")
    
    if len(valid_results) > 0:
        print("Sample of what Topos retrieved:")
        for i, idx in enumerate(valid_results[:5]):
            label = dataset.labels[idx]
            print(f"  {i+1}. Index {idx}: {label['color']} {label['shape']}")
    else:
        print("✓ Topos correctly returned EMPTY SET (logical contradiction detected)")
    
    # Calculate metrics
    safety_success = (len(valid_results) == 0)
    hallucination_rate_topos = (len(valid_results) / k * 100) if k > 0 else 0
    
    # --- REPORTING ---
    print("\n[5/5] Analysis complete!")
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Baseline (Vector Arithmetic):")
    print(f"  Retrieved: {hallucination_count} items")
    print(f"  Hallucination Rate: {hallucination_rate_baseline:.1f}%")
    print(f"  Safety: FAILED (cannot detect contradictions)")
    
    print(f"\nTopos (Sheaf Gluing):")
    print(f"  Retrieved: {len(valid_results)} items")
    print(f"  Hallucination Rate: {hallucination_rate_topos:.1f}%")
    print(f"  Safety: {'PASSED' if safety_success else 'FAILED'}")
    
    if safety_success and hallucination_count > 0:
        print("\n✓ SUCCESS: Topos correctly identified logical contradiction!")
        print("  Vector arithmetic hallucinates; Topos maintains logical safety.")
    else:
        print("\n⚠ FAILURE: Topos failed to reject invalid query.")
    
    return {
        'baseline_retrieved': hallucination_count,
        'baseline_hallucination_rate': hallucination_rate_baseline,
        'topos_retrieved': len(valid_results),
        'topos_hallucination_rate': hallucination_rate_topos,
        'safety_success': safety_success,
        'query': 'Red AND Blue',
        'ground_truth_size': len(impossible_ground_truth),
        'U_red_size': len(U_red),
        'U_blue_size': len(U_blue),
        'U_glued_size': len(U_glued)
    }


if __name__ == "__main__":
    results = run_hallucination_test()
    
    print("\n" + "="*60)
    print("SUMMARY: Logical Safety Test")
    print("="*60)
    print(f"Query: '{results['query']}' (logically impossible)")
    print(f"Ground truth: {results['ground_truth_size']} objects")
    print(f"\nBaseline Hallucination Rate: {results['baseline_hallucination_rate']:.1f}%")
    print(f"Topos Hallucination Rate: {results['topos_hallucination_rate']:.1f}%")
    print(f"\nLogical Safety: {'✓ PASSED' if results['safety_success'] else '✗ FAILED'}")
