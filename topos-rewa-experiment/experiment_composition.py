"""
Experiment Phase 1: Composition via Gluing
Compare Vector Arithmetic vs. Sheaf Gluing
"""

import numpy as np
from config import CONFIG
from data_generation import generate_clevr_lite
from witness_manifold import WitnessManifold
from semantic_sheaf import SemanticSheaf
from utils import nearest_neighbors, evaluate_retrieval


def run_composition_test():
    """
    Compare vector arithmetic vs sheaf gluing for compositional queries
    
    Returns:
        Dictionary with results for both methods
    """
    print("\n" + "="*60)
    print("EXPERIMENT PHASE 1: Composition via Gluing")
    print("="*60)
    
    # Setup
    print("\n[1/5] Generating CLEVR-lite dataset...")
    dataset = generate_clevr_lite()
    print(f"Dataset size: {len(dataset.data)} samples")
    print(f"Attributes: {len(dataset.colors)} colors × {len(dataset.shapes)} shapes")
    
    print("\n[2/5] Building witness manifold...")
    manifold = WitnessManifold(dataset.data)
    
    print("\n[3/5] Constructing semantic sheaf...")
    sheaf = SemanticSheaf(manifold, dataset)
    
    # Task: Find "Red Squares"
    query_color = "red"
    query_shape = "square"
    print(f"\n[4/5] Running compositional query: '{query_color} {query_shape}'")
    
    # Ground truth
    ground_truth = dataset.get_ground_truth(color=query_color, shape=query_shape)
    print(f"Ground truth items: {len(ground_truth)}")
    
    # --- BASELINE: Vector Arithmetic ---
    print("\n--- BASELINE: Vector Arithmetic ---")
    vec_color = dataset.get_embedding(query_color)
    vec_shape = dataset.get_embedding(query_shape)
    vec_query = vec_color + vec_shape
    
    # Retrieve top-k nearest neighbors
    k = len(ground_truth)  # Retrieve same number as ground truth
    results_baseline = nearest_neighbors(vec_query, dataset.data, k=k)
    
    metrics_baseline = evaluate_retrieval(results_baseline, ground_truth)
    print(f"Precision: {metrics_baseline['precision']:.3f}")
    print(f"Recall: {metrics_baseline['recall']:.3f}")
    print(f"F1: {metrics_baseline['f1']:.3f}")
    print(f"Retrieved: {metrics_baseline['n_retrieved']}, Correct: {metrics_baseline['n_correct']}")
    
    # --- TOPOS: Sheaf Gluing ---
    print("\n--- TOPOS: Sheaf Gluing ---")
    
    # Get concept embeddings (prototypes)
    concept_color = dataset.get_prototype(color=query_color)
    concept_shape = dataset.get_prototype(shape=query_shape)
    
    # Compose via sheaf gluing
    results_topos = sheaf.compose_concepts(concept_color, concept_shape)
    
    metrics_topos = evaluate_retrieval(results_topos, ground_truth)
    print(f"Precision: {metrics_topos['precision']:.3f}")
    print(f"Recall: {metrics_topos['recall']:.3f}")
    print(f"F1: {metrics_topos['f1']:.3f}")
    print(f"Retrieved: {metrics_topos['n_retrieved']}, Correct: {metrics_topos['n_correct']}")
    
    # Additional analysis: Check what baseline retrieved incorrectly
    print("\n--- Error Analysis ---")
    baseline_set = set(results_baseline)
    ground_truth_set = set(ground_truth)
    false_positives = baseline_set - ground_truth_set
    
    if len(false_positives) > 0:
        # Sample a few false positives to see what they are
        sample_fps = list(false_positives)[:5]
        print(f"Baseline false positives (sample): {len(false_positives)} total")
        for idx in sample_fps:
            label = dataset.labels[idx]
            print(f"  - Index {idx}: {label['color']} {label['shape']}")
    
    print("\n[5/5] Composition test complete!")
    
    return {
        'baseline': metrics_baseline,
        'topos': metrics_topos,
        'ground_truth_size': len(ground_truth),
        'query': f"{query_color} {query_shape}"
    }


if __name__ == "__main__":
    results = run_composition_test()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Query: {results['query']}")
    print(f"Ground truth size: {results['ground_truth_size']}")
    print(f"\nBaseline (Vector Arithmetic):")
    print(f"  Precision: {results['baseline']['precision']:.3f}")
    print(f"  Recall: {results['baseline']['recall']:.3f}")
    print(f"\nTopos (Sheaf Gluing):")
    print(f"  Precision: {results['topos']['precision']:.3f}")
    print(f"  Recall: {results['topos']['recall']:.3f}")
    print(f"\nImprovement:")
    print(f"  Precision: {(results['topos']['precision'] - results['baseline']['precision']):.3f}")
    print(f"  Recall: {(results['topos']['recall'] - results['baseline']['recall']):.3f}")
