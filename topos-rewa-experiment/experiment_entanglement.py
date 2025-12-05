"""
Experiment 4: The Entanglement Test (Non-Orthogonal Logic)

Demonstrates that Topos-REWA respects conditional topology and detects
statistically impossible combinations, while vector arithmetic ignores correlations.
"""

import numpy as np
from config import CONFIG
from data_entangled import generate_entangled_data
from witness_manifold import WitnessManifold
from semantic_sheaf import SemanticSheaf
from utils import nearest_neighbors, kl_divergence


def run_entanglement_test():
    """
    Test non-orthogonal logic: Query for "Metallic Red" object
    (impossible since Metallic objects are only Grey/Gold in training data)
    
    Returns:
        Dictionary with entanglement detection metrics
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: The Entanglement Test")
    print("Non-Orthogonal Logic & Conditional Topology")
    print("="*60)
    
    # Setup
    print("\n[1/6] Generating entangled dataset...")
    dataset = generate_entangled_data()
    
    print("\n[2/6] Building witness manifold...")
    manifold = WitnessManifold(dataset.data)
    sheaf = SemanticSheaf(manifold, dataset)
    
    # Construct Statistically Impossible Query: "Metallic Red"
    print("\n[3/6] Constructing statistically impossible query: 'Metallic Red'")
    print("(In training data: Metallic → Grey/Gold, Matte → Red/Blue/Green)")
    print("(No 'Metallic Red' objects exist)")
    
    # Ground truth: Should be 0 objects
    ground_truth = dataset.get_ground_truth(color="red", material="metallic")
    print(f"Ground truth for 'Metallic Red': {len(ground_truth)} objects")
    
    # Check what exists
    metallic_count = len(dataset.get_ground_truth(material="metallic"))
    red_count = len(dataset.get_ground_truth(color="red"))
    print(f"Total Metallic objects: {metallic_count}")
    print(f"Total Red objects: {red_count}")
    
    # --- BASELINE: Vector Arithmetic ---
    print("\n[4/6] Running BASELINE (Vector Arithmetic)...")
    vec_metallic = dataset.get_embedding("metallic")
    vec_red = dataset.get_embedding("red")
    vec_impossible = vec_metallic + vec_red  # Ignores correlation
    
    # Retrieve nearest neighbors
    k = 10
    results_baseline = nearest_neighbors(vec_impossible, dataset.data, k=k)
    
    # Analyze what it found
    print(f"Baseline retrieved: {len(results_baseline)} items")
    print("Sample of what baseline retrieved:")
    
    baseline_materials = []
    baseline_colors = []
    for i, idx in enumerate(results_baseline[:5]):
        label = dataset.labels[idx]
        baseline_materials.append(label['material'])
        baseline_colors.append(label['color'])
        print(f"  {i+1}. Index {idx}: {label['color']} {label['material']} (shininess: {label['shininess']})")
    
    # Count how many are actually "Metallic Red"
    correct_baseline = len([idx for idx in results_baseline 
                           if dataset.labels[idx]['material'] == 'metallic' 
                           and dataset.labels[idx]['color'] == 'red'])
    
    # --- TOPOS: Sheaf Gluing ---
    print("\n[5/6] Running TOPOS (Sheaf Gluing)...")
    
    # Get prototype distributions
    metallic_prototype = dataset.get_prototype(material="metallic")
    red_prototype = dataset.get_prototype(color="red")
    
    if metallic_prototype is None or red_prototype is None:
        print("ERROR: Could not find prototypes")
        return None
    
    p_metallic = manifold.get_distribution(metallic_prototype)
    p_red = manifold.get_distribution(red_prototype)
    
    # Measure KL divergence between prototypes (correlation check)
    kl_metallic_red = kl_divergence(p_metallic, p_red)
    print(f"KL divergence between Metallic and Red prototypes: {kl_metallic_red:.4f}")
    print(f"(High KL indicates strong anti-correlation)")
    
    # Define open sets
    U_metallic = sheaf.define_open_set(metallic_prototype)
    U_red = sheaf.define_open_set(red_prototype)
    
    print(f"U_metallic size: {len(U_metallic)} items")
    print(f"U_red size: {len(U_red)} items")
    
    # Glue (intersection)
    U_glued = sheaf.glue_open_sets(U_metallic, U_red)
    print(f"U_glued (intersection) size: {len(U_glued)} items")
    
    # Consistency check
    valid_results = []
    for idx in U_glued:
        p_x = manifold.get_distribution(dataset.data[idx])
        
        # Check KL divergence to both prototypes
        score_metallic = kl_divergence(p_x, p_metallic)
        score_red = kl_divergence(p_x, p_red)
        
        # This should fail because Metallic and Red are anti-correlated
        if max(score_metallic, score_red) < CONFIG["CONSISTENCY_THRESHOLD"]:
            valid_results.append(idx)
    
    print(f"Topos retrieved (after consistency check): {len(valid_results)} items")
    
    if len(valid_results) > 0:
        print("Sample of what Topos retrieved:")
        for i, idx in enumerate(valid_results[:5]):
            label = dataset.labels[idx]
            print(f"  {i+1}. Index {idx}: {label['color']} {label['material']}")
    else:
        print("✓ Topos correctly detected statistical impossibility (empty intersection)")
    
    # Calculate metrics
    entanglement_detected = (len(valid_results) == 0) or (len(valid_results) < len(results_baseline) * 0.1)
    
    # --- REPORTING ---
    print("\n[6/6] Analysis complete!")
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Baseline (Vector Arithmetic):")
    print(f"  Retrieved: {len(results_baseline)} items")
    print(f"  Correct (Metallic Red): {correct_baseline} items")
    print(f"  Ignores correlation: ✗ (treats attributes as independent)")
    
    print(f"\nTopos (Sheaf Gluing):")
    print(f"  Retrieved: {len(valid_results)} items")
    print(f"  KL(Metallic, Red): {kl_metallic_red:.4f}")
    print(f"  Detects anti-correlation: {'✓' if entanglement_detected else '✗'}")
    
    if entanglement_detected:
        print("\n✓ SUCCESS: Topos detected statistical impossibility!")
        print("  Sheaf theory respects conditional topology of the data.")
    else:
        print("\n⚠ PARTIAL: Topos reduced retrieval but didn't fully reject query.")
    
    return {
        'baseline_retrieved': len(results_baseline),
        'baseline_correct': correct_baseline,
        'topos_retrieved': len(valid_results),
        'kl_divergence': kl_metallic_red,
        'entanglement_detected': entanglement_detected,
        'query': 'Metallic Red',
        'ground_truth_size': len(ground_truth),
        'U_metallic_size': len(U_metallic),
        'U_red_size': len(U_red),
        'U_glued_size': len(U_glued)
    }


if __name__ == "__main__":
    results = run_entanglement_test()
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY: Entanglement Detection Test")
        print("="*60)
        print(f"Query: '{results['query']}' (statistically impossible)")
        print(f"Ground truth: {results['ground_truth_size']} objects")
        print(f"\nBaseline Retrieved: {results['baseline_retrieved']} items")
        print(f"Topos Retrieved: {results['topos_retrieved']} items")
        print(f"KL Divergence (anti-correlation): {results['kl_divergence']:.4f}")
        print(f"\nEntanglement Detection: {'✓ PASSED' if results['entanglement_detected'] else '✗ FAILED'}")
