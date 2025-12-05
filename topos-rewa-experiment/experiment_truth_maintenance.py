"""
Experiment Phase 2: Truth Maintenance via KL-Projection
Demonstrate geometric resolution of logical contradictions
"""

import numpy as np
import torch
import torch.optim as optim
from config import CONFIG
from data_generation import generate_clevr_lite
from witness_manifold import WitnessManifold
from utils import kl_divergence


def run_truth_maintenance():
    """
    Simulate logic update and measure geometric recovery
    Scenario: Object is "Red", but must become "Blue"
    
    Returns:
        Dictionary with convergence history
    """
    print("\n" + "="*60)
    print("EXPERIMENT PHASE 2: Truth Maintenance (KL-Projection)")
    print("="*60)
    
    # Setup
    print("\n[1/4] Generating dataset and manifold...")
    dataset = generate_clevr_lite()
    manifold = WitnessManifold(dataset.data)
    
    # Scenario: System thinks Object X is "Red"
    # New Logic Rule: "Object X must be Blue"
    print("\n[2/4] Setting up contradiction scenario...")
    print("Current state: Object is 'Red'")
    print("Target state: Object must be 'Blue'")
    
    # Get a red object
    red_indices = dataset.get_ground_truth(color="red")
    obj_x = dataset.data[red_indices[0]]
    
    # Current distribution
    p_current = manifold.get_distribution(obj_x)
    
    # Target distribution (Blue prototype)
    blue_prototype = dataset.get_prototype(color="blue")
    p_target_region = manifold.get_distribution(blue_prototype)
    
    print(f"Initial KL divergence: {kl_divergence(p_current, p_target_region):.4f}")
    
    # Perform KL-Projection (Natural Gradient Descent)
    print("\n[3/4] Performing KL-projection...")
    print("Minimizing KL(p || p_target) via gradient descent...")
    
    # Convert to torch tensors
    p_new = torch.tensor(p_current, dtype=torch.float32, requires_grad=True)
    p_target = torch.tensor(p_target_region, dtype=torch.float32)
    
    optimizer = optim.SGD([p_new], lr=CONFIG["TM_LEARNING_RATE"])
    
    history_kl = []
    history_fisher_dist = []
    history_step = []
    
    for step in range(CONFIG["TM_MAX_STEPS"]):
        optimizer.zero_grad()
        
        # Normalize to ensure valid probability distribution
        p_new_normalized = torch.softmax(p_new, dim=0)
        
        # KL divergence loss
        epsilon = 1e-10
        p_new_clipped = torch.clamp(p_new_normalized, epsilon, 1.0)
        p_target_clipped = torch.clamp(p_target, epsilon, 1.0)
        
        loss = torch.sum(p_new_clipped * torch.log(p_new_clipped / p_target_clipped))
        
        loss.backward()
        optimizer.step()
        
        # Measure geometric path
        p_new_np = p_new_normalized.detach().numpy()
        d_F = manifold.fisher_distance(p_current, p_new_np)
        
        history_kl.append(loss.item())
        history_fisher_dist.append(d_F)
        history_step.append(step)
        
        if step % 10 == 0:
            print(f"Step {step:3d}: KL = {loss.item():.4f}, Fisher Distance = {d_F:.4f}")
    
    print(f"\n[4/4] Truth maintenance complete!")
    print(f"Final KL divergence: {history_kl[-1]:.4f}")
    print(f"Total Fisher distance moved: {history_fisher_dist[-1]:.4f}")
    print(f"Reduction in inconsistency: {(history_kl[0] - history_kl[-1]) / history_kl[0] * 100:.1f}%")
    
    return {
        'history_kl': history_kl,
        'history_fisher_dist': history_fisher_dist,
        'history_step': history_step,
        'initial_kl': history_kl[0],
        'final_kl': history_kl[-1],
        'total_fisher_dist': history_fisher_dist[-1]
    }


if __name__ == "__main__":
    results = run_truth_maintenance()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Initial logical inconsistency (KL): {results['initial_kl']:.4f}")
    print(f"Final logical inconsistency (KL): {results['final_kl']:.4f}")
    print(f"Total geometric movement (Fisher): {results['total_fisher_dist']:.4f}")
    print(f"Convergence steps: {len(results['history_step'])}")
    
    # Check if converged
    if results['final_kl'] < 0.1:
        print("\n✓ Successfully resolved contradiction via geometric projection")
    else:
        print("\n⚠ Partial resolution - may need more steps or tuning")
