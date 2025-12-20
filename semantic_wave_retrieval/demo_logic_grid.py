import torch
import numpy as np
import matplotlib.pyplot as plt
from semantic_wave_retrieval.symbolic_engine import SymbolicReasoningEngine

def create_grid_graph(size=20):
    # 2D Grid Graph
    edges_src = []
    edges_dst = []
    
    # Grid connectivity
    for r in range(size):
        for c in range(size):
            i = r * size + c
            # Right
            if c < size - 1:
                j = r * size + (c + 1)
                edges_src.append(i); edges_dst.append(j)
                edges_src.append(j); edges_dst.append(i)
            # Down
            if r < size - 1:
                j = (r + 1) * size + c
                edges_src.append(i); edges_dst.append(j)
                edges_src.append(j); edges_dst.append(i)
                
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index, size*size

def run_grid_demo():
    print("\n--- 2D Grid Logic Demo (Island Shrinking) ---")
    size = 20
    edge_index, num_nodes = create_grid_graph(size)
    engine = SymbolicReasoningEngine(edge_index, num_nodes)
    
    # Setup Logic State
    # 1. Theorem (Solid Block of Truth) - Should be preserved
    # 2. Hallucinations (Scattered Spikes) - Should be annihilated
    
    u0 = torch.zeros(num_nodes)
    
    # Theorem Block (Center)
    for r in range(8, 13):
        for c in range(8, 13):
            u0[r*size + c] = 1.0
            
    # Hallucinations (Random Spikes)
    # 10 random spikes away from center
    # Fixed seed
    np.random.seed(42)
    spike_indices = []
    while len(spike_indices) < 15:
        idx = np.random.randint(0, num_nodes)
        # Check if far from center
        r, c = idx // size, idx % size
        dist = abs(r-10) + abs(c-10)
        if dist > 5:
            spike_indices.append(idx)
            u0[idx] = 0.8 # Strong hallucination
            
    print(f"Initial State: Theorem Area=25, Spikes={len(spike_indices)}")
    
    # Setup Obstacle (Axioms)
    # The Theorem must be grounded in an Axiom (Center point)
    mask = torch.zeros(num_nodes)
    center_idx = 10*size + 10
    mask[center_idx] = 1.0 # The Axiom
    
    # 1. Symbolic (TV + MCF)
    u_sym = u0.clone()
    for t in range(200):
         # Stabilized Loop
         # MCF shrinks ungrounded islands
         if t % 2 == 0:
             u_sym = engine.mean_curvature_flow_step(u_sym, dt=0.01)
         else:
             u_sym = engine.tv_flow_step(u_sym, dt=0.01)
             
         u_sym = engine.obstacle_projection(u_sym, mask)
         u_sym = torch.clamp(u_sym, 0.0, 1.0) # Physics stability
            
    # 2. Heat
    u_heat = u0.clone()
    for t in range(200):
        grad = engine.gradient(u_heat)
        div = engine.divergence(grad)
        u_heat += 0.01 * div
        u_heat = engine.obstacle_projection(u_heat, mask)
        u_heat = torch.clamp(u_heat, 0.0, 1.0)
        
    # Analysis
    # 1. Theorem Integrity (Average of 5x5 block)
    # We expect the block to be preserved by TV Flow around the Axiom?
    # Or at least the Axiom spreads to the block?
    # Actually, if we start with Block=1 and flatten it with TV, it stays 1.
    # MCF shrinks it. The Axiom holds the center. The edges might recede.
    # Heat will diffuse the block edges away.
    
    theorem_indices = []
    for r in range(8, 13):
        for c in range(8, 13):
            theorem_indices.append(r*size+c)
            
    sym_theorem = u_sym[theorem_indices].mean().item()
    heat_theorem = u_heat[theorem_indices].mean().item()
    
    # 2. Hallucination Survival (Check spikes)
    # Spikes are NOT grounded by axioms.
    sym_spike_mass = 0
    heat_spike_mass = 0
    for idx in spike_indices:
        sym_spike_mass += u_sym[idx].item()
        heat_spike_mass += u_heat[idx].item()
        
    avg_sym_spike = sym_spike_mass / len(spike_indices)
    avg_heat_spike = heat_spike_mass / len(spike_indices)
    
    print("\n--- Results ---")
    print(f"Theorem Mass (Avg): Sym={sym_theorem:.4f}, Heat={heat_theorem:.4f}")
    print(f"Avg Hallucination : Sym={avg_sym_spike:.4f}, Heat={avg_heat_spike:.4f}")
    
    # Sharpness
    sym_sharpness = ((u_sym < 0.1) | (u_sym > 0.9)).float().mean().item()
    heat_sharpness = ((u_heat < 0.1) | (u_heat > 0.9)).float().mean().item()
    print(f"Sharpness (>0.9/<0.1) : Sym={sym_sharpness*100:.1f}%, Heat={heat_sharpness*100:.1f}%")

    if avg_sym_spike < 0.1 and sym_theorem > 0.9:
        print("SUCCESS: Symbolic Engine preserved Theorem and killed Hallucinations.")
    else:
        print("FAILURE: Physics not behaving as expected.")

if __name__ == "__main__":
    run_grid_demo()
