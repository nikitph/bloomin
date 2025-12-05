"""
Visualizations for Experiment 7: Insight Chains
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA


def plot_concept_evolution_tree(results, save_path='results/concept_evolution_tree.png'):
    """
    Plot concept evolution tree showing parent-child relationships
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes by stage
    stage_1 = results['stage_1']['concepts']
    stage_2 = results['stage_2']['concepts']
    stage_3 = results['stage_3']['concepts']
    
    # Add edges (parent → child relationships)
    # Stage 2 (secondaries from primaries)
    if 'Purple' in stage_2:
        G.add_edge('Red', 'Purple')
        G.add_edge('Blue', 'Purple')
    if 'Orange' in stage_2:
        G.add_edge('Red', 'Orange')
        G.add_edge('Yellow', 'Orange')
    if 'Green' in stage_2:
        G.add_edge('Blue', 'Green')
        G.add_edge('Yellow', 'Green')
    
    # Stage 3 (tertiaries from secondaries)
    if 'Mauve' in stage_3:
        G.add_edge('Purple', 'Mauve')
        G.add_edge('Orange', 'Mauve')
    if 'Chartreuse' in stage_3:
        G.add_edge('Orange', 'Chartreuse')
        G.add_edge('Green', 'Chartreuse')
    if 'Teal' in stage_3:
        G.add_edge('Green', 'Teal')
        G.add_edge('Purple', 'Teal')
    
    # Position nodes by stage
    pos = {}
    
    # Stage 1: Top row
    for i, concept in enumerate(stage_1):
        pos[concept] = (i * 3, 3)
    
    # Stage 2: Middle row
    for i, concept in enumerate(stage_2):
        pos[concept] = (i * 3 + 1, 2)
    
    # Stage 3: Bottom row
    for i, concept in enumerate(stage_3):
        pos[concept] = (i * 3 + 1, 1)
    
    # Draw nodes by stage with different colors
    nx.draw_networkx_nodes(G, pos, nodelist=stage_1, node_color='#FF6B6B', 
                          node_size=2000, alpha=0.9, ax=ax, label='Stage 1: Primaries')
    nx.draw_networkx_nodes(G, pos, nodelist=stage_2, node_color='#4ECDC4', 
                          node_size=2000, alpha=0.9, ax=ax, label='Stage 2: Secondaries')
    nx.draw_networkx_nodes(G, pos, nodelist=stage_3, node_color='#95E1D3', 
                          node_size=2000, alpha=0.9, ax=ax, label='Stage 3: Tertiaries')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, width=2, alpha=0.6, ax=ax,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title('Concept Evolution Tree: Cascading Creativity', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved concept evolution tree to: {save_path}")
    plt.close()


def plot_free_energy_cascade(results, save_path='results/free_energy_cascade.png'):
    """
    Plot Free Energy reduction at each invention stage
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Free Energy reductions by stage
    stage_2_F = results['stage_2']['F_reductions']
    stage_3_F = results['stage_3']['F_reductions']
    
    concepts_2 = results['stage_2']['concepts']
    concepts_3 = results['stage_3']['concepts']
    
    # Bar plot
    x_2 = np.arange(len(concepts_2))
    x_3 = np.arange(len(concepts_3)) + len(concepts_2) + 0.5
    
    ax1.bar(x_2, stage_2_F, color='#4ECDC4', alpha=0.8, label='Stage 2 (Secondaries)')
    ax1.bar(x_3, stage_3_F, color='#95E1D3', alpha=0.8, label='Stage 3 (Tertiaries)')
    
    ax1.set_xticks(list(x_2) + list(x_3))
    ax1.set_xticklabels(concepts_2 + concepts_3, rotation=45, ha='right')
    ax1.set_ylabel('Free Energy Reduction (ΔF)', fontsize=12)
    ax1.set_title('Free Energy Reduction per Invention', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Cumulative Free Energy
    total_stage_2 = results['stage_2']['total_F_reduction']
    total_stage_3 = results['stage_3']['total_F_reduction']
    
    stages = ['Stage 1\n(Primaries)', 'Stage 2\n(Secondaries)', 'Stage 3\n(Tertiaries)']
    cumulative_F = [0, total_stage_2, total_stage_2 + total_stage_3]
    
    ax2.plot(stages, cumulative_F, marker='o', markersize=12, linewidth=3, 
            color='#FF6B6B', label='Cumulative ΔF')
    ax2.fill_between(range(len(stages)), cumulative_F, alpha=0.3, color='#FF6B6B')
    
    # Annotate values
    for i, (stage, val) in enumerate(zip(stages, cumulative_F)):
        ax2.annotate(f'{val:.2f}', (i, val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Cumulative Free Energy Reduction', fontsize=12)
    ax2.set_title('Cascading Free Energy Reduction', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Free Energy cascade to: {save_path}")
    plt.close()


def plot_color_space_embedding(results, save_path='results/color_space_embedding.png'):
    """
    Plot 2D/3D embedding of color space
    """
    agent = results['agent']
    pca = results['stage_4']['pca']
    
    # Get all color prototypes
    all_colors = results['final_ontology']
    color_prototypes = [agent.concept_prototypes[c].detach().numpy() for c in all_colors]
    X = np.stack(color_prototypes)
    
    # Project to 2D
    X_2d = pca.transform(X)[:, :2]
    
    # Separate by stage
    stage_1 = results['stage_1']['concepts']
    stage_2 = results['stage_2']['concepts']
    stage_3 = results['stage_3']['concepts']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot by stage
    for i, concept in enumerate(all_colors):
        if concept in stage_1:
            color, marker, size = '#FF6B6B', 'o', 200
            label = 'Primaries' if i == 0 else ''
        elif concept in stage_2:
            color, marker, size = '#4ECDC4', 's', 150
            label = 'Secondaries' if concept == stage_2[0] else ''
        elif concept in stage_3:
            color, marker, size = '#95E1D3', '^', 150
            label = 'Tertiaries' if concept == stage_3[0] else ''
        else:
            color, marker, size = 'gray', 'x', 100
            label = ''
        
        ax.scatter(X_2d[i, 0], X_2d[i, 1], c=color, marker=marker, s=size, 
                  alpha=0.8, edgecolors='black', linewidths=1.5, label=label)
        ax.annotate(concept, (X_2d[i, 0], X_2d[i, 1]), 
                   textcoords="offset points", xytext=(5,5), fontsize=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Color Space Embedding (PCA)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved color space embedding to: {save_path}")
    plt.close()


def generate_all_visualizations(results):
    """Generate all visualizations for Experiment 7"""
    print("\nGenerating visualizations...")
    plot_concept_evolution_tree(results)
    plot_free_energy_cascade(results)
    plot_color_space_embedding(results)
    print("✓ All visualizations generated!")
