"""
Main orchestration script for Topos-REWA experiments
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG
from experiment_composition import run_composition_test
from experiment_truth_maintenance import run_truth_maintenance
from experiment_hallucination import run_hallucination_test
from experiment_entanglement import run_entanglement_test


def setup_results_dir():
    """Create results directory if it doesn't exist"""
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)


def plot_composition_results(results):
    """
    Plot comparison of baseline vs topos for composition task
    
    Args:
        results: Dictionary from run_composition_test()
    """
    methods = ['Vector\nArithmetic', 'Sheaf\nGluing']
    precision = [results['baseline']['precision'], results['topos']['precision']]
    recall = [results['baseline']['recall'], results['topos']['recall']]
    f1 = [results['baseline']['f1'], results['topos']['f1']]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision
    axes[0].bar(methods, precision, color=['#ff6b6b', '#4ecdc4'])
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Compositional Query Precision', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precision):
        axes[0].text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Recall
    axes[1].bar(methods, recall, color=['#ff6b6b', '#4ecdc4'])
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Compositional Query Recall', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recall):
        axes[1].text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold')
    
    # F1
    axes[2].bar(methods, f1, color=['#ff6b6b', '#4ecdc4'])
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Compositional Query F1', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1):
        axes[2].text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.suptitle(f'Phase 1: Composition via Gluing - Query: "{results["query"]}"',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(CONFIG["RESULTS_DIR"], "composition_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved composition results to: {output_path}")
    plt.close()


def plot_truth_maintenance_results(results):
    """
    Plot truth maintenance dynamics
    
    Args:
        results: Dictionary from run_truth_maintenance()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KL divergence over steps
    axes[0].plot(results['history_step'], results['history_kl'], 
                 linewidth=2, color='#e74c3c', marker='o', markersize=3)
    axes[0].set_xlabel('Optimization Step', fontsize=12)
    axes[0].set_ylabel('KL Divergence (Logical Inconsistency)', fontsize=12)
    axes[0].set_title('Convergence of Logical Inconsistency', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Consistency Threshold')
    axes[0].legend()
    
    # Fisher distance vs KL (geometric path)
    axes[1].plot(results['history_fisher_dist'], results['history_kl'],
                 linewidth=2, color='#3498db', marker='o', markersize=3)
    axes[1].set_xlabel('Fisher Distance Moved', fontsize=12)
    axes[1].set_ylabel('KL Divergence (Logical Inconsistency)', fontsize=12)
    axes[1].set_title('Geometric Path to Consistency', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Consistency Threshold')
    
    # Add arrow showing direction
    if len(results['history_fisher_dist']) > 1:
        mid_idx = len(results['history_fisher_dist']) // 2
        dx = results['history_fisher_dist'][mid_idx + 1] - results['history_fisher_dist'][mid_idx]
        dy = results['history_kl'][mid_idx + 1] - results['history_kl'][mid_idx]
        axes[1].annotate('', xy=(results['history_fisher_dist'][mid_idx + 1], 
                                  results['history_kl'][mid_idx + 1]),
                        xytext=(results['history_fisher_dist'][mid_idx], 
                               results['history_kl'][mid_idx]),
                        arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    axes[1].legend()
    
    plt.suptitle('Phase 2: Truth Maintenance via KL-Projection',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(CONFIG["RESULTS_DIR"], "truth_maintenance_dynamics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved truth maintenance results to: {output_path}")
    plt.close()


def plot_hallucination_results(results):
    """
    Plot hallucination rate comparison
    
    Args:
        results: Dictionary from run_hallucination_test()
    """
    methods = ['Vector\nArithmetic', 'Sheaf\nGluing']
    hallucination_rates = [
        results['baseline_hallucination_rate'],
        results['topos_hallucination_rate']
    ]
    retrieved_counts = [
        results['baseline_retrieved'],
        results['topos_retrieved']
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hallucination Rate
    colors = ['#e74c3c', '#2ecc71']  # Red for baseline, green for topos
    bars = axes[0].bar(methods, hallucination_rates, color=colors)
    axes[0].set_ylabel('Hallucination Rate (%)', fontsize=12)
    axes[0].set_title('Logical Safety: Contradiction Detection', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 110])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Safe (0%)')
    
    for i, (bar, v) in enumerate(zip(bars, hallucination_rates)):
        axes[0].text(i, v + 3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
        if v == 0:
            axes[0].text(i, v + 10, '✓ SAFE', ha='center', color='green', fontweight='bold')
        else:
            axes[0].text(i, v + 10, '✗ UNSAFE', ha='center', color='red', fontweight='bold')
    
    # Items Retrieved
    axes[1].bar(methods, retrieved_counts, color=colors)
    axes[1].set_ylabel('Items Retrieved', fontsize=12)
    axes[1].set_title(f'Query: "{results["query"]}" (Impossible)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, max(retrieved_counts) * 1.2 if max(retrieved_counts) > 0 else 15])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Correct (0)')
    
    for i, v in enumerate(retrieved_counts):
        axes[1].text(i, v + 0.3, f'{v}', ha='center', fontweight='bold', fontsize=11)
    
    plt.suptitle('Phase 3: The Hallucination Trap (Logical Safety)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(CONFIG["RESULTS_DIR"], "hallucination_safety.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved hallucination results to: {output_path}")
    plt.close()


def plot_entanglement_results(results):
    """
    Plot entanglement detection comparison
    
    Args:
        results: Dictionary from run_entanglement_test()
    """
    methods = ['Vector\nArithmetic', 'Sheaf\nGluing']
    retrieved_counts = [
        results['baseline_retrieved'],
        results['topos_retrieved']
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Items Retrieved
    colors = ['#e74c3c', '#2ecc71']
    axes[0].bar(methods, retrieved_counts, color=colors)
    axes[0].set_ylabel('Items Retrieved', fontsize=12)
    axes[0].set_title(f'Query: "{results["query"]}" (Statistically Impossible)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, max(retrieved_counts) * 1.3 if max(retrieved_counts) > 0 else 15])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    for i, v in enumerate(retrieved_counts):
        axes[0].text(i, v + 0.3, f'{v}', ha='center', fontweight='bold', fontsize=11)
    
    # KL Divergence
    kl_values = [0, results['kl_divergence']]
    axes[1].bar(['Baseline\n(Not Measured)', 'Topos\n(Measured)'], kl_values, color=['#95a5a6', '#3498db'])
    axes[1].set_ylabel('KL Divergence (Anti-correlation)', fontsize=12)
    axes[1].set_title('Conditional Topology Awareness', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, max(kl_values) * 1.2])
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[1].text(1, kl_values[1] + 0.1, f'{kl_values[1]:.2f}', ha='center', fontweight='bold', fontsize=11)
    
    plt.suptitle('Phase 4: The Entanglement Test (Non-Orthogonal Logic)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(CONFIG["RESULTS_DIR"], "entanglement_detection.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved entanglement results to: {output_path}")
    plt.close()


def main():
    """Run all experiments and generate visualizations"""
    print("\n" + "="*60)
    print("TOPOS-REWA: Logic & Consistency Experiment")
    print("="*60)
    
    # Setup
    setup_results_dir()
    
    # Phase 1: Composition via Gluing
    composition_results = run_composition_test()
    plot_composition_results(composition_results)
    
    # Phase 2: Truth Maintenance
    truth_maintenance_results = run_truth_maintenance()
    plot_truth_maintenance_results(truth_maintenance_results)
    
    # Phase 3: Hallucination Trap (Logical Safety)
    hallucination_results = run_hallucination_test()
    plot_hallucination_results(hallucination_results)
    
    # Phase 4: Entanglement Test (Non-Orthogonal Logic)
    entanglement_results = run_entanglement_test()
    plot_entanglement_results(entanglement_results)
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {CONFIG['RESULTS_DIR']}/")
    print("\nKey Findings:")
    print(f"1. Sheaf Gluing Precision: {composition_results['topos']['precision']:.3f} "
          f"vs Vector Arithmetic: {composition_results['baseline']['precision']:.3f}")
    print(f"2. Truth Maintenance: Reduced inconsistency by "
          f"{(1 - truth_maintenance_results['final_kl'] / truth_maintenance_results['initial_kl']) * 100:.1f}%")
    print(f"3. Logical Safety: Topos {hallucination_results['topos_hallucination_rate']:.0f}% hallucination "
          f"vs Baseline {hallucination_results['baseline_hallucination_rate']:.0f}% hallucination")
    print(f"4. Contradiction Detection: {'✓ PASSED' if hallucination_results['safety_success'] else '✗ FAILED'}")
    print(f"5. Entanglement Detection: {'✓ PASSED' if entanglement_results['entanglement_detected'] else '✗ FAILED'} "
          f"(KL anti-correlation: {entanglement_results['kl_divergence']:.2f})")
    
    print("\n✓ All 4 experiments completed successfully!")


if __name__ == "__main__":
    main()
