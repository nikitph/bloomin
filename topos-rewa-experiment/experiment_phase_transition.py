"""
Experiment 9d: Phase Transition & Thermodynamics

Tests the physical limits of Relativistic Semantics:
1. Phase Transition (Sweep r from -2 to 2)
2. Entropy Signatures (Constructive vs Destructive)
3. Computational Complexity (Creation vs Understanding)
4. Multi-Step Recovery (Red from Tertiaries)
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from conscious_agent import ConsciousAgent

def entropy(p):
    """Calculate Shannon entropy of a distribution"""
    epsilon = 1e-10
    p = torch.clamp(p, epsilon, 1.0)
    return -torch.sum(p * torch.log(p)).item()

def symmetric_kl(p, q):
    epsilon = 1e-10
    p = torch.clamp(p, epsilon, 1.0)
    q = torch.clamp(q, epsilon, 1.0)
    return 0.5 * (torch.sum(p * torch.log(p / q)) + torch.sum(q * torch.log(q / p))).item()

def run_phase_transition_experiments():
    print("\n" + "="*60)
    print("EXPERIMENT 9d: PHASE TRANSITION & THERMODYNAMICS")
    print("Probing the physics of concept formation")
    print("="*60)
    
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    
    # Helper to create mixture
    def create_mixture(c1, c2, name, method='power', r=1.0):
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
        elif method == 'max':
            mix = torch.max(p1, p2)
        elif method == 'min':
            mix = torch.min(p1, p2)
        elif method == 'power':
            if abs(r) < 1e-6: # Limit as r->0 is geometric
                mix = torch.sqrt(p1 * p2)
            else:
                # Handle negative r carefully
                if r < 0:
                    # For negative r, zeros cause infinity. Add epsilon.
                    mix = ((p1**r + p2**r) / 2) ** (1/r)
                else:
                    mix = ((p1**r + p2**r) / 2) ** (1/r)
            
        if torch.sum(mix) < 1e-6:
            mix = torch.ones_like(mix) # Fallback
            
        mix = mix / torch.sum(mix)
        agent.register_prototype(name, mix)
        return mix

    # ---------------------------------------------------------
    # Experiment 1: Phase Transition (Sweep r)
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("1. TEST PHASE TRANSITION")
    print("Hypothesis: Sharp transition at r=0")
    print("-" * 60)
    
    r_values = np.linspace(-2, 2, 20)
    recovery_distances = []
    
    print(f"Sweeping r from -2 to 2 ({len(r_values)} points)...")
    
    for r in r_values:
        name_p = f'Purple_r{r:.2f}'
        name_o = f'Orange_r{r:.2f}'
        
        try:
            create_mixture('Red', 'Blue', name_p, method='power', r=r)
            create_mixture('Red', 'Yellow', name_o, method='power', r=r)
            
            _, _, p_int = agent.find_intersection(name_p, name_o)
            
            if p_int is not None:
                p_red = agent.concept_prototypes['Red']
                dist = symmetric_kl(p_red, p_int)
                recovery_distances.append(dist)
            else:
                recovery_distances.append(10.0) # Failure penalty
        except Exception as e:
            print(f"Error at r={r}: {e}")
            recovery_distances.append(10.0)
            
    # Print results table
    print("\nResults (r vs Distance):")
    for r, d in zip(r_values, recovery_distances):
        status = "✓" if d < 0.2 else "✗"
        print(f"  r={r:5.2f}: {d:6.4f} {status}")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, recovery_distances, marker='o', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', label='Geometric Mean (r=0)')
    plt.title('Phase Transition: Recovery Distance vs Mixing Parameter r')
    plt.xlabel('Mixing Parameter r')
    plt.ylabel('Recovery Distance (KL)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/phase_transition.png')
    print("✓ Saved phase transition plot to results/phase_transition.png")

    # ---------------------------------------------------------
    # Experiment 2: Entropy Predictions
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("2. TEST ENTROPY PREDICTIONS")
    print("Hypothesis: Constructive increases entropy, Destructive decreases")
    print("-" * 60)
    
    p_red = agent.concept_prototypes['Red']
    p_blue = agent.concept_prototypes['Blue']
    
    H_red = entropy(p_red)
    H_blue = entropy(p_blue)
    print(f"H(Red)  = {H_red:.4f}")
    print(f"H(Blue) = {H_blue:.4f}")
    
    # Constructive (Arithmetic Mean)
    create_mixture('Red', 'Blue', 'Purple_Add', method='arithmetic')
    H_add = entropy(agent.concept_prototypes['Purple_Add'])
    print(f"H(Add)  = {H_add:.4f} (Expected > {max(H_red, H_blue):.4f})")
    
    if H_add >= max(H_red, H_blue):
        print("✓ SUCCESS: Entropy Increased (Constructive)")
    else:
        print("✗ FAILURE: Entropy Decreased")
        
    # Destructive (Min Mixing)
    create_mixture('Red', 'Blue', 'Purple_Min', method='min')
    H_min = entropy(agent.concept_prototypes['Purple_Min'])
    print(f"H(Min)  = {H_min:.4f} (Expected < {min(H_red, H_blue):.4f})")
    
    # Note: Min mixing of disjoint sets is empty/uniform, which might have HIGH entropy if normalized?
    # Actually, if Min is zero everywhere, we fallback to uniform, which is MAX entropy.
    # But if there is slight overlap, it should be peaked (low entropy).
    # Let's check the result.
    
    # ---------------------------------------------------------
    # Experiment 3: Computational Complexity
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("3. TEST COMPUTATIONAL COMPLEXITY")
    print("Hypothesis: Understanding is slower than Creation")
    print("-" * 60)
    
    # Ensure concepts exist for the test
    create_mixture('Red', 'Blue', 'Purple', method='arithmetic')
    create_mixture('Red', 'Yellow', 'Orange', method='arithmetic')
    
    # Creation (Forward)
    start = time.time()
    for _ in range(100):
        create_mixture('Red', 'Blue', 'Temp_Purple', method='arithmetic')
    create_time = (time.time() - start) / 100
    
    # Understanding (Backward)
    start = time.time()
    for _ in range(100):
        agent.find_intersection('Purple', 'Orange')
    understand_time = (time.time() - start) / 100
    
    print(f"Creation Time:      {create_time*1000:.4f} ms")
    print(f"Understanding Time: {understand_time*1000:.4f} ms")
    ratio = understand_time / create_time
    print(f"Ratio: {ratio:.1f}x")
    
    if ratio > 1.0:
        print("✓ SUCCESS: Understanding is harder than Creation")
    else:
        print("✗ FAILURE: Unexpected timing")

    # ---------------------------------------------------------
    # Experiment 4: Multi-Step Recovery
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("4. TEST MULTI-STEP RECOVERY")
    print("Hypothesis: Can recover Red from Tertiaries")
    print("-" * 60)
    
    # Re-initialize agent to clear pollution from Phase Transition sweep
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    
    # Create Tertiaries (using Geometric Mean for robustness)
    create_mixture('Red', 'Blue', 'Purple_G', method='geometric')
    create_mixture('Red', 'Yellow', 'Orange_G', method='geometric')
    create_mixture('Blue', 'Yellow', 'Green_G', method='geometric')
    
    create_mixture('Purple_G', 'Orange_G', 'Mauve_G', method='geometric')
    create_mixture('Orange_G', 'Green_G', 'Chartreuse_G', method='geometric')
    
    # Mauve = sqrt(Purple * Orange) = sqrt(sqrt(RB) * sqrt(RY)) = (R^2 BY)^(1/4)
    # Chartreuse = sqrt(Orange * Green) = sqrt(sqrt(RY) * sqrt(BY)) = (R Y^2 B)^(1/4)
    
    # Intersection(Mauve, Chartreuse) should extract common factors: R, Y, B?
    # Common to Mauve/Chartreuse is Orange (RY).
    # Let's try to recover Orange first.
    
    print("Attempting: Intersection(Mauve, Chartreuse) -> Orange")
    nearest, dist, p_int = agent.find_intersection('Mauve_G', 'Chartreuse_G')
    print(f"  Result: {nearest} (dist={dist:.4f})")
    
    if nearest == 'Orange':
        print("  ✓ SUCCESS: Recovered Orange (Level 1)")
        
        # Now try to recover Red from (Recovered Orange) and Purple
        # We need to register the recovered concept
        agent.register_prototype('Recovered_Orange', p_int)
        
        print("Attempting: Intersection(Recovered_Orange, Purple) -> Red")
        nearest_2, dist_2, _ = agent.find_intersection('Recovered_Orange', 'Purple_G')
        print(f"  Result: {nearest_2} (dist={dist_2:.4f})")
        
        if nearest_2 == 'Red':
            print("  ✓ SUCCESS: Recovered Red (Level 2)")
        else:
            print("  ✗ FAILURE at Level 2")
    else:
        print("  ✗ FAILURE at Level 1")

if __name__ == "__main__":
    run_phase_transition_experiments()
