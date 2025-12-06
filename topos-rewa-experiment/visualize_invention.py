"""
Visualization utilities for autopoietic concept invention experiment
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from config import CONFIG


def plot_free_energy_trace(free_energy_history, step_history, output_dir="results"):
    """
    Plot Free Energy evolution over time
    Shows the "Aha!" moment when concept is invented
    
    Args:
        free_energy_history: List of Free Energy values
        step_history: List of step numbers
        output_dir: Output directory for plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot Free Energy trace
    ax.plot(step_history, free_energy_history, linewidth=3, color='#e74c3c', marker='o', markersize=6)
    
    # Mark key phases
    # Start: Equilibrium
    ax.axhline(y=free_energy_history[0], color='green', linestyle='--', alpha=0.3, label='Initial Equilibrium')
    
    # Dream injection (step 0)
    ax.annotate('Dream Injection\n(Confusion)', xy=(0, free_energy_history[0]),
                xytext=(20, free_energy_history[0] + 0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    # Find the "Aha!" moment (steepest drop)
    if len(free_energy_history) > 2:
        drops = [free_energy_history[i] - free_energy_history[i+1] 
                for i in range(len(free_energy_history)-1)]
        max_drop_idx = np.argmax(drops) + 1
        
        ax.annotate('"Aha!" Moment\n(Invention)', 
                   xy=(step_history[max_drop_idx], free_energy_history[max_drop_idx]),
                   xytext=(step_history[max_drop_idx] + 30, free_energy_history[max_drop_idx] - 0.3),
                   arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                   fontsize=11, fontweight='bold', color='gold')
    
    # Final equilibrium
    ax.axhline(y=free_energy_history[-1], color='blue', linestyle='--', alpha=0.3, label='New Equilibrium')
    
    ax.set_xlabel('Optimization Step', fontsize=13)
    ax.set_ylabel('Free Energy (F = E - TS)', fontsize=13)
    ax.set_title('Thermodynamic Learning: Free Energy Trace', fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add phase annotations
    ax.text(0.02, 0.98, 'Phase 1: Dreaming', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.02, 0.90, 'Phase 2: Dissonance', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(0.02, 0.82, 'Phase 3: Invention', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "free_energy_trace.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Free Energy trace to: {output_path}")
    plt.close()


def plot_geometry_evolution(agent, output_dir="results"):
    """
    Plot geometry evolution using t-SNE
    Shows manifold warping to accommodate new concept
    
    Args:
        agent: ConsciousAgent instance
        output_dir: Output directory for plot
    """
    # Get all concept prototypes
    concepts = agent.ontology
    prototypes = [agent.concept_prototypes[c].detach().numpy() for c in concepts]
    
    # Stack into matrix
    X = np.stack(prototypes)
    
    # Apply t-SNE if we have enough points
    if len(prototypes) >= 2:
        # For 3 points, use 2D t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(2, len(prototypes)-1))
        X_embedded = tsne.fit_transform(X)
    else:
        # Just use first 2 dimensions
        X_embedded = X[:, :2]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Color map
    colors = {'Red': '#e74c3c', 'Blue': '#3498db', 'Purple': '#9b59b6'}
    
    # Plot each concept
    for i, concept in enumerate(concepts):
        color = colors.get(concept, '#95a5a6')
        ax.scatter(X_embedded[i, 0], X_embedded[i, 1], 
                  s=500, c=color, alpha=0.7, edgecolors='black', linewidth=2,
                  label=concept)
        ax.text(X_embedded[i, 0], X_embedded[i, 1], concept,
               fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Draw connections
    if 'Purple' in concepts:
        red_idx = concepts.index('Red')
        blue_idx = concepts.index('Blue')
        purple_idx = concepts.index('Purple')
        
        # Red to Purple
        ax.plot([X_embedded[red_idx, 0], X_embedded[purple_idx, 0]],
               [X_embedded[red_idx, 1], X_embedded[purple_idx, 1]],
               'k--', alpha=0.3, linewidth=1.5)
        
        # Blue to Purple
        ax.plot([X_embedded[blue_idx, 0], X_embedded[purple_idx, 0]],
               [X_embedded[blue_idx, 1], X_embedded[purple_idx, 1]],
               'k--', alpha=0.3, linewidth=1.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.set_title('Geometry Evolution: Manifold Warping for New Concept', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(alpha=0.2)
    
    # Add annotation
    ax.text(0.02, 0.02, 'Manifold expanded to accommodate Purple\n(Red + Blue superposition)',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "geometry_evolution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved geometry evolution to: {output_path}")
    plt.close()


def generate_all_visualizations(results, output_dir="results"):
    """
    Generate all visualizations for invention experiment
    
    Args:
        results: Dictionary from run_invention_experiment()
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Free Energy trace
    plot_free_energy_trace(
        results['free_energy_history'],
        results['step_history'],
        output_dir
    )
    
    # Geometry evolution
    plot_geometry_evolution(results['agent'], output_dir)
    
    print("✓ All visualizations generated!")
