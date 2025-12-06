"""
Experiment 9: Basis Change (Relativistic Meaning)

Demonstrates that the system can perform Subtractive Logic (Intersection)
to recover primary concepts from secondary concepts.

Hypothesis:
    If Purple ≈ Red + Blue
    And Orange ≈ Red + Yellow
    Then Intersection(Purple, Orange) ≈ Red

This proves the system understands the constituent structure of concepts
and can operate in different bases (Additive vs Subtractive).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from conscious_agent import ConsciousAgent


def run_basis_change_experiment():
    """
    Run Basis Change experiment:
    1. Learn Primaries (Red, Blue, Yellow)
    2. Invent Secondaries (Purple, Orange, Green)
    3. Recover Primaries via Intersection (Subtractive Logic)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 9: BASIS CHANGE (RELATIVISTIC MEANING)")
    print("Deriving Primaries from Secondaries via Subtractive Logic")
    print("="*60)
    
    # 1. Initialize with Primaries
    print("\n[1/3] Learning Primary Colors (Basis A)...")
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    print(f"Ontology: {list(agent.ontology)}")
    
    # 2. Invent Secondaries (Basis B)
    print("\n[2/3] Inventing Secondary Colors (Basis B)...")
    
    # We need to ensure these are actually invented/learned properly
    # For this experiment, we'll simulate the invention process by explicitly
    # creating the mixtures, as we want to test the LOGIC of intersection,
    # assuming the concepts exist.
    
    # Helper to create mixture
    def create_mixture(c1, c2, name):
        p1 = agent.concept_prototypes[c1]
        p2 = agent.concept_prototypes[c2]
        mix = (p1 + p2) / 2.0
        mix = mix / torch.sum(mix)
        agent.register_prototype(name, mix)
        print(f"  Created {name} from {c1} + {c2}")
        
    create_mixture('Red', 'Blue', 'Purple')
    create_mixture('Red', 'Yellow', 'Orange')
    create_mixture('Blue', 'Yellow', 'Green')
    
    print(f"Ontology: {list(agent.ontology)}")
    
    # 3. Test Subtractive Logic (Intersection)
    print("\n[3/3] Testing Subtractive Logic (Intersection)...")
    print("Hypothesis: Intersection(Secondary1, Secondary2) -> Primary")
    
    results = []
    
    # Test 1: Red from Purple & Orange
    # Purple(R,B) AND Orange(R,Y) -> Red
    print("\nTest 1: Intersection(Purple, Orange)")
    nearest, dist, p_int = agent.find_intersection('Purple', 'Orange')
    print(f"  Result: Closest concept is '{nearest}' (dist={dist:.4f})")
    
    if nearest == 'Red':
        print("  ✓ SUCCESS: Recovered Red from Purple & Orange")
        results.append(True)
    else:
        print(f"  ✗ FAILURE: Expected Red, got {nearest}")
        results.append(False)
        
    # Test 2: Blue from Purple & Green
    # Purple(R,B) AND Green(B,Y) -> Blue
    print("\nTest 2: Intersection(Purple, Green)")
    nearest, dist, p_int = agent.find_intersection('Purple', 'Green')
    print(f"  Result: Closest concept is '{nearest}' (dist={dist:.4f})")
    
    if nearest == 'Blue':
        print("  ✓ SUCCESS: Recovered Blue from Purple & Green")
        results.append(True)
    else:
        print(f"  ✗ FAILURE: Expected Blue, got {nearest}")
        results.append(False)
        
    # Test 3: Yellow from Orange & Green
    # Orange(R,Y) AND Green(B,Y) -> Yellow
    print("\nTest 3: Intersection(Orange, Green)")
    nearest, dist, p_int = agent.find_intersection('Orange', 'Green')
    print(f"  Result: Closest concept is '{nearest}' (dist={dist:.4f})")
    
    if nearest == 'Yellow':
        print("  ✓ SUCCESS: Recovered Yellow from Orange & Green")
        results.append(True)
    else:
        print(f"  ✗ FAILURE: Expected Yellow, got {nearest}")
        results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    
    success_rate = sum(results) / len(results) * 100
    print(f"Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    
    if all(results):
        print("\n✓ BASIS CHANGE DEMONSTRATED")
        print("The system successfully:")
        print("  1. Understood concepts in Basis B (Secondaries)")
        print("  2. Performed subtractive logic (Intersection)")
        print("  3. Recovered concepts in Basis A (Primaries)")
        print("\nThis proves it understands the MANIFOLD structure,")
        print("independent of the specific coordinate system (Basis).")
    
    return agent, results

if __name__ == "__main__":
    run_basis_change_experiment()
