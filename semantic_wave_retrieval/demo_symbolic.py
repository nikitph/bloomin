import torch
import numpy as np
import matplotlib.pyplot as plt
from semantic_wave_retrieval.symbolic_engine import SymbolicReasoningEngine

def create_logic_graph():
    # Construct a simple logic graph:
    # Cluster A (Truth Claim): Nodes 0-19 connected strongly.
    # Cluster B (False Claim): Nodes 20-39 connected strongly.
    # Weak bridge between them.
    
    edges_src = []
    edges_dst = []
    
    # Internal clusters
    for i in range(20):
        for j in range(i+1, 20):
            if np.random.rand() < 0.3:
                edges_src.append(i); edges_dst.append(j)
                edges_src.append(j); edges_dst.append(i)
                
    for i in range(20, 40):
        for j in range(i+1, 40):
            if np.random.rand() < 0.3:
                edges_src.append(i); edges_dst.append(j)
                edges_src.append(j); edges_dst.append(i)
                
    # One weak bridge
    edges_src.append(10); edges_dst.append(30)
    edges_src.append(30); edges_dst.append(10)
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index, 40

def run_demo():
    print("--- Sparse Symbolic Engine Demo ---")
    edge_index, num_nodes = create_logic_graph()
    engine = SymbolicReasoningEngine(edge_index, num_nodes)
    
    # Initial State
    # Seed: Node 0 is TRUE (1.0).
    # Noise: Random spurious activations (0.1 - 0.4) everywhere.
    # Constraint: Node 20 is FALSE (-1.0 -> 0.0).
    
    # Noise: Sparse Spikes (Hallucinations)
    u0 = torch.zeros(num_nodes)
    spike_idx = [15, 25, 35, 38, 19] # fixed for reproducibility
    u0[spike_idx] = 0.8
    
    # Apply Seeds Constraints
    mask = torch.zeros(num_nodes) # Restore definition
    mask[0] = 1.0 # Truth root
    mask[20] = -1.0 # False root (Obstacle) - this will zero out any spike at 20 or flows to it
    
    u0 = engine.obstacle_projection(u0, mask)
    
    print(f"Initial State: Seed=1.0, Spikes at {spike_idx}")
    
    # 1. Run Symbolic Engine (TV + MCF)
    # Strong MCF to shrink isolated spikes
    u_sym = engine.solve(u0, mask, T_tv=300, T_mc=150, dt=0.02)
    
    # 2. Run Heat Diffusion Baseline (Laplacian)
    u_heat = u0.clone()
    for t in range(450): 
        grad = engine.gradient(u_heat)
        div = engine.divergence(grad)
        u_heat += 0.02 * div 
        u_heat = engine.obstacle_projection(u_heat, mask)
        
    print("\n--- Results Analysis ---")
    print(f"{'Node Type':<15} | {'Symbolic (u)':<15} | {'Heat Diffusion (u)':<15}")
    print("-" * 50)
    
    # Check key nodes
    debug_nodes = [0, 5, 20, 25, 38] # 38 is a Spike
    labels = ["Seed (Truth)", "Truth Member", "Obstacle (False)", "False Cluster", "Spike (Noise)"]
    
    for idx, lbl in zip(debug_nodes, labels):
        print(f"{lbl:<15} | {u_sym[idx]:.4f}          | {u_heat[idx]:.4f}")
        
    # Metrics
    # Spike Suppression: Check node 38
    sym_spike = u_sym[38].item()
    heat_spike = u_heat[38].item()
    
    # Sharpness
    sym_sharpness = ((u_sym < 0.1) | (u_sym > 0.9)).float().mean().item()
    heat_sharpness = ((u_heat < 0.1) | (u_heat > 0.9)).float().mean().item()
    
    print("\n--- Metrics ---")
    print(f"Spike (Node 38)        : Sym={sym_spike:.4f} vs Heat={heat_spike:.4f}")
    print(f"Sharpness (>0.9/<0.1)  : Sym={sym_sharpness*100:.1f}% vs Heat={heat_sharpness*100:.1f}%")
    
    if sym_spike < 0.1 and heat_spike > 0.1:
        print("\nSUCCESS: Symbolic Engine killed hallucination spike; Heat preserved/blurred it.")
    elif sym_spike < heat_spike:
        print("\nPARTIAL SUCCESS: Symbolic suppressed noise better.")
    else:
        print("\nFAILURE: Symbolic Engine failed.")

if __name__ == "__main__":
    run_demo()
