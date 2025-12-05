"""
Experiment 9f: Dynamic Sharpening & Hierarchy

Tests the role of active error correction (Ricci Flow) in maintaining
semantic invertibility across deep hierarchies.

1. Dynamic Sharpening (Ricci Flow at each level)
2. Energy Cost of Depth (Scaling)
3. Alternative Sharpening Methods
4. Biological Timescales
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from conscious_agent import ConsciousAgent

def symmetric_kl(p, q):
    epsilon = 1e-10
    p = torch.clamp(p, epsilon, 1.0)
    q = torch.clamp(q, epsilon, 1.0)
    return 0.5 * (torch.sum(p * torch.log(p / q)) + torch.sum(q * torch.log(q / p))).item()

def create_mixture(agent, c1, c2, name, method='geometric'):
    p1 = agent.concept_prototypes[c1]
    p2 = agent.concept_prototypes[c2]
    
    epsilon = 1e-10
    p1 = torch.clamp(p1, epsilon, 1.0)
    p2 = torch.clamp(p2, epsilon, 1.0)
    
    if method == 'geometric':
        mix = torch.sqrt(p1 * p2)
    elif method == 'arithmetic':
        mix = (p1 + p2) / 2.0
        
    mix = mix / torch.sum(mix)
    agent.register_prototype(name, mix)
    return mix

def run_dynamic_sharpening_experiments():
    print("\n" + "="*60)
    print("EXPERIMENT 9f: DYNAMIC SHARPENING & HIERARCHY")
    print("Testing active error correction (Ricci Flow)")
    print("="*60)
    
    # ---------------------------------------------------------
    # Experiment 1: Test Dynamic Sharpening
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("1. TEST DYNAMIC SHARPENING")
    print("Hypothesis: Sharpening at each level enables deep recovery")
    print("-" * 60)
    
    agent = ConsciousAgent(initial_concepts=['Red', 'Blue', 'Yellow'])
    
    # Level 1: Mix + Sharpen
    print("Level 1: Creating Secondaries...")
    create_mixture(agent, 'Red', 'Blue', 'Purple_Raw', method='geometric')
    create_mixture(agent, 'Red', 'Yellow', 'Orange_Raw', method='geometric')
    create_mixture(agent, 'Blue', 'Yellow', 'Green_Raw', method='geometric')
    
    # Sharpen (Simulated Ricci Flow via resolve_contradiction logic)
    # Actually, we can just call resolve_contradiction but force it to optimize the existing concept?
    # Or better, use the resolve_contradiction method to "invent" the concept properly.
    # But for control, let's manually invoke the optimization loop.
    
    def sharpen(concept_name):
        # Simple entropy minimization / sharpening
        p = agent.concept_prototypes[concept_name].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([p], lr=0.01)
        
        # Target: Minimize entropy (sharpen) but stay close to original (fidelity)
        p_orig = p.detach().clone()
        
        for _ in range(50):
            optimizer.zero_grad()
            p_soft = torch.softmax(p, dim=0)
            
            # Loss = Entropy + Distance to Original
            entropy = -torch.sum(p_soft * torch.log(p_soft + 1e-10))
            dist = torch.sum((p_soft - p_orig)**2)
            
            loss = entropy + 10.0 * dist # Balance sharpening vs fidelity
            loss.backward()
            optimizer.step()
            
        p_final = torch.softmax(p, dim=0).detach()
        agent.register_prototype(concept_name + "_Sharp", p_final)
        return concept_name + "_Sharp"
        
    purple_sharp = sharpen('Purple_Raw')
    orange_sharp = sharpen('Orange_Raw')
    green_sharp = sharpen('Green_Raw')
    
    print(f"Sharpened Secondaries: {purple_sharp}, {orange_sharp}, {green_sharp}")
    
    # Level 2: Mix + Sharpen (using sharpened secondaries)
    print("Level 2: Creating Tertiaries...")
    create_mixture(agent, purple_sharp, orange_sharp, 'Mauve_Raw', method='geometric')
    create_mixture(agent, orange_sharp, green_sharp, 'Chartreuse_Raw', method='geometric')
    
    mauve_sharp = sharpen('Mauve_Raw')
    chartreuse_sharp = sharpen('Chartreuse_Raw')
    
    print(f"Sharpened Tertiaries: {mauve_sharp}, {chartreuse_sharp}")
    
    # Test Recovery
    print("\nAttempting Recovery: Intersection(Mauve_Sharp, Chartreuse_Sharp) -> Orange_Sharp")
    
    nearest, dist, p_int = agent.find_intersection(mauve_sharp, chartreuse_sharp)
    print(f"  Result: {nearest} (dist={dist:.4f})")
    
    if nearest == 'Orange_Raw_Sharp': # Matches our naming convention
        print("  ✓ SUCCESS: Recovered Orange (Level 1)")
        
        # Register intermediate
        agent.register_prototype('Rec_Orange', p_int)
        
        print("Attempting: Intersection(Rec_Orange, Purple_Sharp) -> Red")
        nearest_2, dist_2, _ = agent.find_intersection('Rec_Orange', purple_sharp)
        print(f"  Result: {nearest_2} (dist={dist_2:.4f})")
        
        if nearest_2 == 'Red':
            print("  ✓ SUCCESS: Recovered Red (Level 2)")
        else:
            print("  ✗ FAILURE at Level 2")
    else:
        print("  ✗ FAILURE at Level 1")
        # Check distance to Orange_Raw_Sharp specifically
        p_orange = agent.concept_prototypes['Orange_Raw_Sharp']
        d_check = symmetric_kl(p_orange, p_int)
        print(f"  (Distance to Orange_Sharp: {d_check:.4f})")
        if d_check < 0.2:
            print("  (But distance is low! Maybe nearest neighbor search found something else?)")

    # ---------------------------------------------------------
    # Experiment 2: Measure Energy Cost of Depth
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("2. MEASURE ENERGY COST OF DEPTH")
    print("Hypothesis: Cost scales exponentially with depth")
    print("-" * 60)
    
    depths = [1, 2, 3] # Keep small for speed
    times = []
    
    for d in depths:
        start = time.time()
        # Simulate building hierarchy of depth d
        # Branching factor 2
        num_concepts = 2**d
        for _ in range(num_concepts):
            # Create + Sharpen
            create_mixture(agent, 'Red', 'Blue', 'Temp', method='geometric')
            sharpen('Temp')
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Depth {d}: {elapsed:.4f}s")
        
    # Fit exponential
    # time ~ b^d
    # log(time) ~ d * log(b)
    if len(depths) > 1:
        log_times = np.log(times)
        slope, _ = np.polyfit(depths, log_times, 1)
        b = np.exp(slope)
        print(f"\nBranching Cost Factor b = {b:.2f}x")
    
    # ---------------------------------------------------------
    # Experiment 3: Alternative Sharpening Methods
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("3. TEST ALTERNATIVE SHARPENING METHODS")
    print("Hypothesis: Ricci Flow (Entropy+Fidelity) is optimal")
    print("-" * 60)
    
    # Setup
    create_mixture(agent, 'Red', 'Blue', 'Purple_Base', method='geometric')
    p_base = agent.concept_prototypes['Purple_Base']
    
    # Method 1: Ricci Flow (Already tested above)
    p_ricci = agent.concept_prototypes['Purple_Raw_Sharp']
    
    # Method 2: Simple Power (Contrast Enhancement)
    p_power = p_base ** 2
    p_power = p_power / torch.sum(p_power)
    agent.register_prototype('Purple_Power', p_power)
    
    # Method 3: Top-K (Sparse)
    k = 10
    vals, inds = torch.topk(p_base, k)
    p_sparse = torch.zeros_like(p_base)
    p_sparse[inds] = vals
    p_sparse = p_sparse / torch.sum(p_sparse)
    agent.register_prototype('Purple_Sparse', p_sparse)
    
    # Compare Recovery Quality
    # Intersection(Purple_Method, Orange) -> Red
    # We need Orange sharpened with same method for fair comparison
    
    methods = [
        ('Ricci', 'Purple_Raw_Sharp', 'Orange_Raw_Sharp'),
        ('Power', 'Purple_Power', 'Orange_Raw'), # Hack: mixing methods
        ('Sparse', 'Purple_Sparse', 'Orange_Raw')
    ]
    
    # Let's just create Orange_Power and Orange_Sparse properly
    p_orange_base = agent.concept_prototypes['Orange_Raw']
    
    p_orange_power = p_orange_base ** 2
    p_orange_power = p_orange_power / torch.sum(p_orange_power)
    agent.register_prototype('Orange_Power', p_orange_power)
    
    vals, inds = torch.topk(p_orange_base, k)
    p_orange_sparse = torch.zeros_like(p_orange_base)
    p_orange_sparse[inds] = vals
    p_orange_sparse = p_orange_sparse / torch.sum(p_orange_sparse)
    agent.register_prototype('Orange_Sparse', p_orange_sparse)
    
    methods = [
        ('Ricci', 'Purple_Raw_Sharp', 'Orange_Raw_Sharp'),
        ('Power', 'Purple_Power', 'Orange_Power'),
        ('Sparse', 'Purple_Sparse', 'Orange_Sparse')
    ]
    
    print("\nRecovery Distance to Red:")
    for name, p_name, o_name in methods:
        _, _, p_int = agent.find_intersection(p_name, o_name)
        if p_int is not None:
            dist = symmetric_kl(agent.concept_prototypes['Red'], p_int)
            print(f"  {name}: {dist:.4f}")
        else:
            print(f"  {name}: Failed (No intersection)")

    # ---------------------------------------------------------
    # Experiment 4: Biological Timescales
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print("4. TEST BIOLOGICAL TIMESCALES")
    print("Hypothesis: Sharpening corresponds to consolidation (slow)")
    print("-" * 60)
    
    # Measure Mixing Time
    start = time.time()
    for _ in range(100):
        create_mixture(agent, 'Red', 'Blue', 'Temp', method='geometric')
    mix_time = (time.time() - start) / 100
    
    # Measure Intersection Time
    start = time.time()
    for _ in range(100):
        agent.find_intersection('Purple_Raw', 'Orange_Raw')
    int_time = (time.time() - start) / 100
    
    # Measure Sharpening Time
    start = time.time()
    for _ in range(10): # Slower
        sharpen('Purple_Raw')
    sharp_time = (time.time() - start) / 10
    
    print(f"Mixing Time:      {mix_time*1000:.4f} ms")
    print(f"Intersection Time:{int_time*1000:.4f} ms")
    print(f"Sharpening Time:  {sharp_time*1000:.4f} ms")
    
    ratio = sharp_time / mix_time
    print(f"\nRatio (Sharpening / Mixing): {ratio:.1f}x")
    
    # Scale to Biology (1ms system = 1s bio?)
    # Synaptic transmission ~1ms. Mixing is fast.
    # Consolidation ~seconds/minutes. Sharpening is slow.
    
    print("\nBiological Analogy:")
    print(f"  Mixing (Synaptic):       {mix_time*1000:.2f} ms")
    print(f"  Reasoning (Active):      {int_time*1000:.2f} ms")
    print(f"  Consolidation (Sleep):   {sharp_time*1000:.2f} ms")

if __name__ == "__main__":
    run_dynamic_sharpening_experiments()
