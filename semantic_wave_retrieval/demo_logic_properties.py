import torch
import numpy as np
from semantic_wave_retrieval.symbolic_engine import SymbolicReasoningEngine

def create_chain_graph(length=10):
    edges_src = []
    edges_dst = []
    for i in range(length - 1):
        edges_src.append(i)
        edges_dst.append(i+1)
        # Undirected underlying graph? Or directed?
        # Graph Laplacian is usually undirected.
        # But logical implication is directed.
        # The "Obstacle" handles the direction. 
        # The physics (smoothness) is isotropic.
        edges_src.append(i+1)
        edges_dst.append(i)
        
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index, length

def apply_implication_constraint(u, chain_length):
    # Rule: Node i implies Node i+1
    # u[i+1] >= u[i] (Hard Implication?)
    # Or "If A then B" -> u_B cannot be less than u_A?
    # Yes, let's try u[i+1] = max(u[i+1], u[i]).
    # But wait, if u[i] is 0.3 (noise), this forces u[i+1] to be 0.3.
    # This propagates noise!
    # Ah, the Constraint is "If A is TRUE".
    # But `u` is continuous. 
    # Maybe the constraint is: u[B] >= u[A] - margin?
    # Or simply: The PDE *is* the reasoning. The obstacle is just consistency.
    # If we enforce u[B] >= u[A], then yes, noise propagates.
    # UNLESS the PDE kills the noise in A faster than it propagates to B?
    # Yes! TV Flow crushes u[A]=0.3 -> 0.0. 
    # So u[B] (constrained >= u[A]) is allowed to drop to 0.
    # Heat Flow preserves u[A]=0.3 (smooths it). So u[B] stays up.
    
    # We apply constraint sequentially for the chain
    # Note: This is an "Active Constraint" updated every step
    for i in range(chain_length - 1):
        # u[i+1] must be at least u[i] (Implication)
        # We use a soft mask or hard check?
        if u[i] > u[i+1]:
            u[i+1] = u[i]
    return u

def run_experiment(name, seed_val, description):
    print(f"\n--- Experiment: {name} (Seed={seed_val}) ---")
    print(description)
    
    length = 10
    edge_index, num_nodes = create_chain_graph(length)
    engine = SymbolicReasoningEngine(edge_index, num_nodes)
    
    # Setup
    u0 = torch.zeros(num_nodes)
    u0[0] = seed_val # Seed at start of chain
    
    # Static Constraints (None, we use Dynamic Implication)
    mask = None 
    
    # 1. Symbolic Engine
    u_sym = u0.clone()
    for t in range(200): # Steps
        # Physics Step
        # Interleave TV and MCF?
        # engine.solve does loop internally. We need manual loop for dynamic constraint.
        u_sym = engine.tv_flow_step(u_sym, dt=0.05)
        # u_sym = engine.mean_curvature_flow_step(u_sym, dt=0.05) # Optional
        if t % 2 == 0:
             u_sym = engine.mean_curvature_flow_step(u_sym, dt=0.05)
             
        # Dynamic Constraint: Propagate Implication
        u_sym = apply_implication_constraint(u_sym, length)
        
        # Also enforce Seed if it's "Fact"? 
        # If Seed is "Hypothesis" (0.3), we don't fix it.
        # If Seed is "Axiom" (1.0), we might fix u[0]=1.
        # Let's assume Seed is INITIAL state, not fixed BC.
        # So 0.3 should decay. 1.0 should stay (if stable)? 
        # Actually TV flow preserves 1.0 flat regions. 
        pass
        
    # 2. Heat Engine
    u_heat = u0.clone()
    for t in range(200):
        grad = engine.gradient(u_heat)
        div = engine.divergence(grad)
        u_heat += 0.05 * div
        u_heat = apply_implication_constraint(u_heat, length)

    # Report
    print(f"{'Node' :<5} | {'Sym':<8} | {'Heat':<8}")
    for i in range(length):
        print(f"{i:<5} | {u_sym[i]:.4f}   | {u_heat[i]:.4f}")
        
    # Metrics
    # "Digital Integrity": End of chain value
    end_sym = u_sym[-1].item()
    end_heat = u_heat[-1].item()
    
    print(f"End of Chain (u[{length-1}]): Sym={end_sym:.4f}, Heat={end_heat:.4f}")
    
    return end_sym, end_heat

def run_shock_experiment():
    print(f"\n--- Experiment: Logic Separation (Shock Interface) ---")
    length = 20
    edge_index, num_nodes = create_chain_graph(length)
    engine = SymbolicReasoningEngine(edge_index, num_nodes)
    
    # Setup BCs: Conflict
    # Node 0 = TRUE (1.0)
    # Node 19 = FALSE (-1.0)
    # Middle is Unknown (0.0)
    u0 = torch.zeros(num_nodes)
    u0[0] = 1.0
    u0[length-1] = -1.0
    
    # Mask to enforce BCs properly
    mask = torch.zeros(num_nodes)
    mask[0] = 1.0
    mask[length-1] = -1.0 
    # Technically mask=0 means free. We need a way to say "Fixed".
    # obstacle_projection in engine forces 1 if mask=1, 0 if mask=-1.
    # We want -1. So engine needs update?
    # Engine logic: if mask=1 -> 1. if mask=-1 -> 0.
    # User requested -1 BC.
    # Let's handle BCs manually in loop for this demo to be safe.
    
    # 1. Symbolic (TV Flow)
    # TV Flow minimizes variation. Should produce flat plateaus with 1 jump.
    u_sym = u0.clone()
    for t in range(500):
        u_sym = engine.tv_flow_step(u_sym, dt=0.01)
        # Enforce BCs
        u_sym[0] = 1.0
        u_sym[length-1] = -1.0
        
    # 2. Heat (Diffusion)
    # Should produce linear ramp
    u_heat = u0.clone()
    for t in range(500):
        grad = engine.gradient(u_heat)
        div = engine.divergence(grad)
        u_heat += 0.01 * div
        u_heat[0] = 1.0
        u_heat[length-1] = -1.0

    # Report
    print(f"{'Node' :<5} | {'Sym':<8} | {'Heat':<8}")
    for i in range(length):
        print(f"{i:<5} | {u_sym[i]:.4f}   | {u_heat[i]:.4f}")
        
    # Metric: Width of Ambiguity
    # Nodes where |u| < 0.9 (Not clearly True or False)
    sym_ambiguous = ((u_sym.abs() < 0.9)).sum().item()
    heat_ambiguous = ((u_heat.abs() < 0.9)).sum().item()
    
    print(f"\nAmbiguous Nodes (|u|<0.9): Sym={sym_ambiguous}, Heat={heat_ambiguous}")
    
    # Expectation: Sym should be small (ideally 1-2 interface nodes). Heat should be large (~18).
    
    if sym_ambiguous < heat_ambiguous:
        print("SUCCESS: Symbolic Engine created sharp logical boundary.")
    else:
        print("FAILURE: Symbolic Engine failed to sharpen.")

if __name__ == "__main__":
    run_shock_experiment()
