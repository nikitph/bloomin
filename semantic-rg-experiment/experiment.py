"""Main experiment: Semantic RG Flow."""

import numpy as np
import matplotlib.pyplot as plt
from data_generation import generate_hierarchical_data
from witness_init import extract_micro_witnesses
from rg_operator import renormalization_step
from measurements import measure_couplings, evaluate_retrieval_accuracy
from config import CONFIG


def run_rg_experiment():
    """
    Run the complete Renormalization Group flow experiment.
    
    Returns:
        list: History of measurements at each RG scale
    """
    print("=" * 70)
    print("SEMANTIC RENORMALIZATION GROUP FLOW EXPERIMENT")
    print("=" * 70)
    
    # Generate hierarchical data
    print("\n[1/4] Generating hierarchical data...")
    data = generate_hierarchical_data()
    
    # Initialize microscopic witnesses
    print("\n[2/4] Initializing microscopic witnesses...")
    current_dist, _, _ = extract_micro_witnesses(data)
    
    # Track history
    history = []
    
    print("\n[3/4] Starting Renormalization Flow...")
    print("-" * 70)
    
    # Initial Measurement (s=0)
    print(f"\nScale 0 (Microscopic):")
    obs = measure_couplings(current_dist, data)
    acc = evaluate_retrieval_accuracy(current_dist, data, k=10)
    
    print(f"  L={obs['L']}, rho={obs['rho']:.6f}, K={obs['K']:.6f}, Chi={obs['Chi']:.6f}")
    print(f"  Recall@10: {acc:.4f}")
    
    history.append({**obs, "scale": 0, "acc": acc})
    
    # Flow loop
    for step in range(1, CONFIG["RG_STEPS"] + 1):
        current_L = current_dist.shape[1]
        target_L = int(current_L * CONFIG["COMPRESSION_FACTOR"])
        
        # Ensure we don't go below a minimum size
        if target_L < 4:
            print(f"\nStopping: target size {target_L} too small")
            break
        
        print(f"\nScale {step}:")
        
        # Apply Blocking Operator
        current_dist = renormalization_step(current_dist, target_L)
        
        # Measure Observables
        obs = measure_couplings(current_dist, data)
        acc = evaluate_retrieval_accuracy(current_dist, data, k=10)
        
        print(f"  L={obs['L']}, rho={obs['rho']:.6f}, K={obs['K']:.6f}, Chi={obs['Chi']:.6f}")
        print(f"  Recall@10: {acc:.4f}")
        
        history.append({**obs, "scale": step, "acc": acc})
    
    print("\n[4/4] Generating visualizations...")
    visualize_rg_flow(history)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return history


def visualize_rg_flow(history):
    """
    Create RG flow diagrams.
    
    Args:
        history: List of measurement dictionaries
    """
    scales = [h["scale"] for h in history]
    L_vals = [h["L"] for h in history]
    rho_vals = [h["rho"] for h in history]
    Chi_vals = [h["Chi"] for h in history]
    acc_vals = [h["acc"] for h in history]
    K_vals = [h["K"] for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Semantic Renormalization Group Flow", fontsize=16, fontweight='bold')
    
    # 1. L(s) vs Scale
    ax = axes[0, 0]
    ax.plot(scales, L_vals, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel("RG Scale", fontsize=12)
    ax.set_ylabel("Witness Count L", fontsize=12)
    ax.set_title("Witness Multiplicity Flow", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. rho(s) vs Scale
    ax = axes[0, 1]
    ax.plot(scales, rho_vals, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel("RG Scale", fontsize=12)
    ax.set_ylabel("Signal Strength ρ", fontsize=12)
    ax.set_title("Signal Concentration", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Chi(s) vs Scale
    ax = axes[0, 2]
    ax.plot(scales, Chi_vals, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.axhline(y=np.mean(Chi_vals), color='red', linestyle='--', 
               label=f'Mean: {np.mean(Chi_vals):.4f}', linewidth=2)
    ax.set_xlabel("RG Scale", fontsize=12)
    ax.set_ylabel("χ (Thermodynamic Variable)", fontsize=12)
    ax.set_title("Fixed Point Behavior", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Accuracy vs Scale
    ax = axes[1, 0]
    ax.plot(scales, acc_vals, 'o-', linewidth=2, markersize=8, color='#6A994E')
    ax.set_xlabel("RG Scale", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Retrieval Accuracy Preservation", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 5. Compression Ratio vs Accuracy
    ax = axes[1, 1]
    compression_ratios = [L_vals[0] / L for L in L_vals]
    ax.plot(compression_ratios, acc_vals, 'o-', linewidth=2, markersize=8, color='#BC4B51')
    ax.set_xlabel("Compression Ratio (L₀/L)", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("One-Shot Generalization", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 6. Curvature vs Scale
    ax = axes[1, 2]
    ax.plot(scales, K_vals, 'o-', linewidth=2, markersize=8, color='#8B5A3C')
    ax.set_xlabel("RG Scale", fontsize=12)
    ax.set_ylabel("Curvature Proxy K", fontsize=12)
    ax.set_title("Geometric Curvature Flow", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rg_flow_results.png', dpi=300, bbox_inches='tight')
    print("  Saved: rg_flow_results.png")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nFixed Point Analysis:")
    print(f"  Chi mean: {np.mean(Chi_vals):.6f}")
    print(f"  Chi std:  {np.std(Chi_vals):.6f}")
    print(f"  Chi stability: {np.std(Chi_vals) / np.mean(Chi_vals):.4f} (lower is better)")
    
    print(f"\nCompression Analysis:")
    print(f"  Initial L: {L_vals[0]}")
    print(f"  Final L:   {L_vals[-1]}")
    print(f"  Total compression: {L_vals[0] / L_vals[-1]:.2f}x")
    
    print(f"\nRetrieval Preservation:")
    print(f"  Initial accuracy: {acc_vals[0]:.4f}")
    print(f"  Final accuracy:   {acc_vals[-1]:.4f}")
    print(f"  Accuracy drop:    {acc_vals[0] - acc_vals[-1]:.4f}")
    
    # Find scale where accuracy drops below 90% of initial
    threshold = 0.9 * acc_vals[0]
    critical_scale = None
    for i, acc in enumerate(acc_vals):
        if acc < threshold:
            critical_scale = i
            break
    
    if critical_scale is not None:
        print(f"\nCritical Scale (90% accuracy threshold):")
        print(f"  Scale: {critical_scale}")
        print(f"  L: {L_vals[critical_scale]}")
        print(f"  Compression: {L_vals[0] / L_vals[critical_scale]:.2f}x")
    else:
        print(f"\nNo critical scale found (accuracy remains > 90%)")


if __name__ == "__main__":
    history = run_rg_experiment()
