"""
Experiment 9b: Advanced Basis Experiments

Tests the robustness of Relativistic Semantics:
1. Basis Independence (Random Bases)
2. Over-Complete Bases (Redundant Representations)
3. Non-Linear Bases (Geometric Mean Mixing)
"""

import torch
import numpy as np
import random
import itertools
from conscious_agent import ConsciousAgent

def kl_div(p, q):
    epsilon = 1e-10
    p = torch.clamp(p, epsilon, 1.0)
    q = torch.clamp(q, epsilon, 1.0)
    return torch.sum(p * torch.log(p / q)).item()

def symmetric_kl(p, q):
    return 0.5 * (kl_div(p, q) + kl_div(q, p))

def reconstruct_target(agent, target_name, basis_names):
    """
    Attempt to reconstruct target using basis concepts.
    Tries pairwise Intersection and Union.
    """
    target_p = agent.concept_prototypes[target_name]
    best_dist = float('inf')
    best_op = None
    best_pair = None
    
    # Try all pairs
    for b1, b2 in itertools.combinations(basis_names, 2):
        # 1. Try Intersection
        _, _, p_int = agent.find_intersection(b1, b2)
        if p_int is not None:
            dist = symmetric_kl(target_p, p_int)
            if dist < best_dist:
                best_dist = dist
                best_op = "Intersection"
                best_pair = (b1, b2)
        
        # 2. Try Union (Mixing)
        p1 = agent.concept_prototypes[b1]
        p2 = agent.concept_prototypes[b2]
        p_union = (p1 + p2) / 2.0
        p_union = p_union / torch.sum(p_union)
        
        dist = symmetric_kl(target_p, p_union)
        if dist < best_dist:
            best_dist = dist
            best_op = "Union"
            best_pair = (b1, b2)
            
    return best_dist, best_op, best_pair

def run_advanced_basis_experiments():
    print("\n" + "="*60)
    print("EXPERIMENT 9b: ADVANCED BASIS EXPERIMENTS")
    print("Testing Relativistic Semantics & Basis Independence")
    print("="*60)
    
    # Setup Agent with rich ontology
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    
    # Helper to create mixture
    def create_mixture(c1, c2, name, method='arithmetic', r=1.0):
        p1 = agent.concept_prototypes[c1]
        p2 = agent.concept_prototypes[c2]
        
        epsilon = 1e-10
        p1 = torch.clamp(p1, epsilon, 1.0)
        p2 = torch.clamp(p2, epsilon, 1.0)
        
        if method == 'arithmetic':
            mix = (p1 + p2) / 2.0
        elif method == 'geometric':
            mix = torch.sqrt(p1 * p2)
        elif method == 'harmonic':
            mix = 2 * p1 * p2 / (p1 + p2)
        elif method == 'power':
            if abs(r) < 1e-6: # Limit as r->0 is geometric
                mix = torch.sqrt(p1 * p2)
            else:
                mix = ((p1**r + p2**r) / 2) ** (1/r)
        elif method == 'max':
            mix = torch.max(p1, p2)
        elif method == 'min':
            mix = torch.min(p1, p2)
            
        if torch.sum(mix) < 1e-6:
            mix = torch.ones_like(mix) # Fallback
            
        mix = mix / torch.sum(mix)
        agent.register_prototype(name, mix)
        return mix

    # Create standard hierarchy
    create_mixture('Red', 'Blue', 'Purple')
    create_mixture('Red', 'Yellow', 'Orange')
    create_mixture('Blue', 'Yellow', 'Green')
    
    all_concepts = list(agent.ontology)
    print(f"Ontology: {all_concepts}")
    
    # ---------------------------------------------------------
    # Experiment 3: Non-Linear Bases (Geometric)
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("3. TEST NON-LINEAR BASES (GEOMETRIC)")
    print("Hypothesis: Intersection works with Geometric Mean mixing")
    print("-" * 60)
    
    create_mixture('Red', 'Blue', 'Purple_Geom', method='geometric')
    create_mixture('Red', 'Yellow', 'Orange_Geom', method='geometric')
    
    print("Attempting: Intersection(Purple_Geom, Orange_Geom) -> Red")
    _, _, p_int = agent.find_intersection('Purple_Geom', 'Orange_Geom')
    
    if p_int is not None:
        p_red = agent.concept_prototypes['Red']
        dist = symmetric_kl(p_red, p_int)
        print(f"Distance to Red: {dist:.4f}")
        if dist < 0.1: print("✓ SUCCESS")
        else: print("✗ FAILURE")
        
    # ---------------------------------------------------------
    # Experiment 4: Harmonic Mean Bases
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("4. TEST HARMONIC MEAN BASES")
    print("Hypothesis: Harmonic mean is 'product-like', should work well")
    print("-" * 60)
    
    create_mixture('Red', 'Blue', 'Purple_Harm', method='harmonic')
    create_mixture('Red', 'Yellow', 'Orange_Harm', method='harmonic')
    
    print("Attempting: Intersection(Purple_Harm, Orange_Harm) -> Red")
    _, _, p_int = agent.find_intersection('Purple_Harm', 'Orange_Harm')
    
    if p_int is not None:
        p_red = agent.concept_prototypes['Red']
        dist = symmetric_kl(p_red, p_int)
        print(f"Distance to Red: {dist:.4f}")
        if dist < 0.1: print("✓ SUCCESS")
        else: print("✗ FAILURE")

    # ---------------------------------------------------------
    # Experiment 5: Power Mean Bases (Generalized)
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("5. TEST POWER MEAN BASES")
    print("Hypothesis: Optimal r depends on manifold curvature")
    print("-" * 60)
    
    for r in [0.5, 2.0, 5.0]:
        print(f"\nTesting r={r}:")
        name_p = f'Purple_Pow{r}'
        name_o = f'Orange_Pow{r}'
        create_mixture('Red', 'Blue', name_p, method='power', r=r)
        create_mixture('Red', 'Yellow', name_o, method='power', r=r)
        
        _, _, p_int = agent.find_intersection(name_p, name_o)
        if p_int is not None:
            p_red = agent.concept_prototypes['Red']
            dist = symmetric_kl(p_red, p_int)
            print(f"  Distance to Red: {dist:.4f}")
            if dist < 0.1: print("  ✓ SUCCESS")
            else: print("  ✗ FAILURE")

    # ---------------------------------------------------------
    # Experiment 6: Max/Min Mixing
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("6. TEST MAX/MIN MIXING")
    print("Hypothesis: Intersection still works but needs different metric")
    print("-" * 60)
    
    # Test Max
    print("\nTesting Max Mixing:")
    create_mixture('Red', 'Blue', 'Purple_Max', method='max')
    create_mixture('Red', 'Yellow', 'Orange_Max', method='max')
    
    _, _, p_int = agent.find_intersection('Purple_Max', 'Orange_Max')
    if p_int is not None:
        p_red = agent.concept_prototypes['Red']
        dist = symmetric_kl(p_red, p_int)
        print(f"  Distance to Red: {dist:.4f}")
        if dist < 0.1: print("  ✓ SUCCESS")
        else: print("  ✗ FAILURE")
        
    # Test Min
    print("\nTesting Min Mixing:")
    create_mixture('Red', 'Blue', 'Purple_Min', method='min')
    create_mixture('Red', 'Yellow', 'Orange_Min', method='min')
    
    _, _, p_int = agent.find_intersection('Purple_Min', 'Orange_Min')
    if p_int is not None:
        p_red = agent.concept_prototypes['Red']
        dist = symmetric_kl(p_red, p_int)
        print(f"  Distance to Red: {dist:.4f}")
        if dist < 0.1: print("  ✓ SUCCESS")
        else: print("  ✗ FAILURE")

if __name__ == "__main__":
    run_advanced_basis_experiments()
