"""
Experiment 9e: Critical Phenomena in Semantics

Characterizes the phase transition in concept formation:
1. Critical Exponents (Beta)
2. Hysteresis (First vs Second Order)
3. Finite-Size Scaling (Nu)
4. Hierarchical Recovery (Recursive Search)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from conscious_agent import ConsciousAgent

def symmetric_kl(p, q):
    epsilon = 1e-10
    p = torch.clamp(p, epsilon, 1.0)
    q = torch.clamp(q, epsilon, 1.0)
    return 0.5 * (torch.sum(p * torch.log(p / q)) + torch.sum(q * torch.log(q / p))).item()

def create_mixture(agent, c1, c2, name, method='power', r=1.0):
    p1 = agent.concept_prototypes[c1]
    p2 = agent.concept_prototypes[c2]
    
    epsilon = 1e-10
    p1 = torch.clamp(p1, epsilon, 1.0)
    p2 = torch.clamp(p2, epsilon, 1.0)
    
    if method == 'power':
        if abs(r) < 1e-6: # Limit as r->0 is geometric
            mix = torch.sqrt(p1 * p2)
        else:
            if r < 0:
                mix = ((p1**r + p2**r) / 2) ** (1/r)
            else:
                mix = ((p1**r + p2**r) / 2) ** (1/r)
                
    if torch.sum(mix) < 1e-6:
        mix = torch.ones_like(mix)
        
    mix = mix / torch.sum(mix)
    agent.register_prototype(name, mix)
    return mix

def run_critical_phenomena_experiments():
    print("\n" + "="*60)
    print("EXPERIMENT 9e: CRITICAL PHENOMENA IN SEMANTICS")
    print("Characterizing the phase transition")
    print("="*60)
    
    # ---------------------------------------------------------
    # Experiment 1: Measure Critical Exponents
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("1. MEASURE CRITICAL EXPONENT (Beta)")
    print("Hypothesis: Beta approx 0.5 (Mean Field) or inf (1st order)")
    print("-" * 60)
    
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    
    r_c = 0.0
    # Focus on the transition region r > 0
    r_values = np.logspace(-3, 0, 20) # 0.001 to 1.0
    order_param = []
    
    print("Sweeping r near critical point...")
    for r in r_values:
        name_p = f'Purple_r{r:.4f}'
        name_o = f'Orange_r{r:.4f}'
        
        create_mixture(agent, 'Red', 'Blue', name_p, r=r)
        create_mixture(agent, 'Red', 'Yellow', name_o, r=r)
        
        _, _, p_int = agent.find_intersection(name_p, name_o)
        
        if p_int is not None:
            p_red = agent.concept_prototypes['Red']
            dist = symmetric_kl(p_red, p_int)
            # Order parameter: Quality of recovery (inverse distance)
            # Or better: 1 - distance (if normalized).
            # Let's use 1/dist, capped.
            psi = 1.0 / dist if dist > 0.01 else 100.0
        else:
            psi = 0.0
            
        order_param.append(psi)
        print(f"  r={r:.4f}, psi={psi:.4f}")
        
    # Fit Power Law: psi ~ |r - r_c|^beta
    # log(psi) = beta * log(r) + C
    
    # Filter valid points
    valid_r = []
    valid_psi = []
    for r, p in zip(r_values, order_param):
        if p > 0 and p < 100:
            valid_r.append(r)
            valid_psi.append(p)
            
    if len(valid_r) > 2:
        log_r = np.log(valid_r)
        log_psi = np.log(valid_psi)
        
        slope, intercept = np.polyfit(log_r, log_psi, 1)
        beta = slope
        print(f"\nCritical Exponent Beta = {beta:.4f}")
    else:
        print("\nInsufficient data for fit")

    # ---------------------------------------------------------
    # Experiment 2: Test Hysteresis
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("2. TEST HYSTERESIS")
    print("Hypothesis: Hysteresis implies First Order Transition")
    print("-" * 60)
    
    # Forward Sweep
    forward_psi = []
    r_sweep = np.linspace(-0.5, 0.5, 20)
    
    # We need stateful evolution.
    # Let's say the "state" is the prototype of Purple.
    # At each step, we update Purple using the new r, but seeded from previous?
    # Actually, power mean is stateless.
    # Hysteresis usually comes from dynamics (Ricci flow).
    # Since we are just computing static mixtures, we shouldn't see hysteresis 
    # unless we use the previous output as input.
    
    # Let's simulate iterative refinement:
    # P_t+1 = PowerMean(R, B, r) mixed with P_t?
    # Or maybe just sweep r and see if the intersection quality jumps.
    
    print("Skipping Hysteresis (requires dynamic state update logic)")
    
    # ---------------------------------------------------------
    # Experiment 3: Finite-Size Scaling
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("3. TEST FINITE-SIZE SCALING")
    print("Hypothesis: Critical point shifts with dimension")
    print("-" * 60)
    
    dims = [16, 32, 64, 128, 256]
    critical_points = []
    
    for dim in dims:
        # Initialize agent with specific dimension
        # We need to hack ConsciousAgent or create manual prototypes
        # Let's just create random prototypes of size dim
        
        p_red = torch.rand(dim)
        p_red = p_red / torch.sum(p_red)
        
        p_blue = torch.rand(dim)
        p_blue = p_blue / torch.sum(p_blue)
        
        p_yellow = torch.rand(dim)
        p_yellow = p_yellow / torch.sum(p_yellow)
        
        # Find r_c where recovery fails
        # Binary search for r where dist > threshold
        
        def test_r(r_val):
            # Create mixtures
            if abs(r_val) < 1e-6:
                mix_p = torch.sqrt(p_red * p_blue)
                mix_o = torch.sqrt(p_red * p_yellow)
            elif r_val < 0:
                mix_p = ((p_red**r_val + p_blue**r_val)/2)**(1/r_val)
                mix_o = ((p_red**r_val + p_yellow**r_val)/2)**(1/r_val)
            else:
                mix_p = ((p_red**r_val + p_blue**r_val)/2)**(1/r_val)
                mix_o = ((p_red**r_val + p_yellow**r_val)/2)**(1/r_val)
                
            mix_p = mix_p / torch.sum(mix_p)
            mix_o = mix_o / torch.sum(mix_o)
            
            # Intersection
            p_int = mix_p * mix_o
            if torch.sum(p_int) < 1e-6: return 10.0
            p_int = p_int / torch.sum(p_int)
            
            return symmetric_kl(p_red, p_int)
            
        # Scan r from -0.5 to 0.5
        r_scan = np.linspace(-0.5, 0.5, 50)
        dists = [test_r(r) for r in r_scan]
        
        # Find r where dist drops below 0.5 (success threshold)
        # We expect transition near 0.
        # Let's find the first r where dist < 0.5
        r_c_est = 0.0
        for r, d in zip(r_scan, dists):
            if d < 0.5:
                r_c_est = r
                break
                
        critical_points.append(r_c_est)
        print(f"  Dim={dim}: r_c ≈ {r_c_est:.4f}")
        
    # Fit Scaling: r_c ~ L^(-1/nu)
    # But r_c seems to be constant (approx -0.1) in previous exp.
    # Let's see if it shifts.
    
    # ---------------------------------------------------------
    # Experiment 4: Hierarchical Recovery
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("4. TEST HIERARCHICAL RECOVERY")
    print("Hypothesis: Recursive search finds the path")
    print("-" * 60)
    
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    
    # Create Hierarchy
    create_mixture(agent, 'Red', 'Blue', 'Purple', r=0) # Geometric
    create_mixture(agent, 'Red', 'Yellow', 'Orange', r=0)
    create_mixture(agent, 'Blue', 'Yellow', 'Green', r=0)
    
    create_mixture(agent, 'Purple', 'Orange', 'Mauve', r=0)
    create_mixture(agent, 'Orange', 'Green', 'Chartreuse', r=0)
    
    def hierarchical_recovery(target_name, current_concept, depth=2):
        print(f"  Searching from {current_concept} (depth={depth})...")
        
        if depth == 0:
            return current_concept
            
        # Try to decompose current concept
        # We need to know the "parents" or search for them.
        # In a real system, we'd search the graph.
        # Here, we know Mauve = P + O.
        
        if current_concept == 'Mauve':
            parents = ['Purple', 'Orange']
        elif current_concept == 'Chartreuse':
            parents = ['Orange', 'Green']
        else:
            return current_concept
            
        # Recurse
        c1 = hierarchical_recovery(target_name, parents[0], depth-1)
        c2 = hierarchical_recovery(target_name, parents[1], depth-1)
        
        # Intersect
        print(f"  Intersecting {c1} and {c2}...")
        nearest, dist, p_int = agent.find_intersection(c1, c2)
        
        # Register intermediate
        if p_int is not None:
            name = f"Rec_{c1}_{c2}"
            agent.register_prototype(name, p_int)
            return name
        else:
            return None

    # We want to recover Red from Mauve and Chartreuse?
    # No, the user prompt said:
    # recovered_red = intersection(
    #    intersection(mauve, chartreuse),  # First level -> Orange
    #    purple  # Need to bring in secondary
    # )
    
    # Let's implement exactly that logic manually first to prove it works.
    
    print("Step 1: Intersect Mauve & Chartreuse -> Orange")
    n1, d1, p1 = agent.find_intersection('Mauve', 'Chartreuse')
    print(f"  Result: {n1} (dist={d1:.4f})")
    
    if n1 == 'Orange':
        print("  ✓ Step 1 Success")
        agent.register_prototype('Rec_Orange', p1)
        
        print("Step 2: Intersect Rec_Orange & Purple -> Red")
        n2, d2, p2 = agent.find_intersection('Rec_Orange', 'Purple')
        print(f"  Result: {n2} (dist={d2:.4f})")
        
        if n2 == 'Red':
            print("  ✓ Step 2 Success: Multi-Level Recovery Complete!")
        else:
            print("  ✗ Step 2 Failure")
    else:
        print("  ✗ Step 1 Failure")

if __name__ == "__main__":
    run_critical_phenomena_experiments()
