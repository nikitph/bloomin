import torch
import numpy as np
from scipy.stats import pearsonr
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")
from models_with_losses import TacitReasonerWithLosses
from data_gen import HierarchicalGraphGenerator, GraphConfig


def analyze_witness_structure(model, graph_gen, save_path="witness_analysis.png"):
    """
    Diagnostic: Analyze correlation between witness similarity and graph distance.
    
    If witnesses encode distance structure, similar witnesses should correspond
    to nearby nodes in the graph.
    """
    model.eval()
    num_nodes = len(graph_gen.graph.nodes())
    
    print("\n" + "="*60)
    print("DIAGNOSTIC: Witness-Distance Correlation Analysis")
    print("="*60)
    
    # Extract witnesses for all nodes (using self-pairs)
    all_witnesses = []
    with torch.no_grad():
        for node_id in range(num_nodes):
            w = model.get_node_witness(node_id)
            all_witnesses.append(w.squeeze())
    
    all_witnesses = torch.stack(all_witnesses)  # [num_nodes, witness_dim]
    
    # Compute pairwise witness cosine similarity
    witness_norm = all_witnesses / (all_witnesses.norm(dim=1, keepdim=True) + 1e-8)
    witness_sim = torch.mm(witness_norm, witness_norm.t())  # [num_nodes, num_nodes]
    
    # Get graph distances
    graph_dist = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j in graph_gen.shortest_paths[i]:
                graph_dist[i, j] = graph_gen.shortest_paths[i][j]
            else:
                graph_dist[i, j] = -1  # Unreachable
    
    # Filter out unreachable pairs
    mask = graph_dist >= 0
    witness_sim_flat = witness_sim.numpy()[mask]
    graph_dist_flat = graph_dist[mask]
    
    # Separate by distance ranges
    short_mask = (graph_dist_flat >= 1) & (graph_dist_flat <= 4)
    long_mask = (graph_dist_flat >= 5) & (graph_dist_flat <= 8)
    
    # Compute correlations
    if short_mask.sum() > 0:
        corr_short = pearsonr(witness_sim_flat[short_mask], -graph_dist_flat[short_mask])[0]
        print(f"Correlation (distance 1-4): {corr_short:.4f}")
    else:
        corr_short = None
        print("No pairs with distance 1-4")
    
    if long_mask.sum() > 0:
        corr_long = pearsonr(witness_sim_flat[long_mask], -graph_dist_flat[long_mask])[0]
        print(f"Correlation (distance 5-8): {corr_long:.4f}")
    else:
        corr_long = None
        print("No pairs with distance 5-8")
    
    # Overall correlation
    corr_all = pearsonr(witness_sim_flat, -graph_dist_flat)[0]
    print(f"Correlation (all distances): {corr_all:.4f}")
    
    # Visualization
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Witness similarity vs Graph distance (short)
        if short_mask.sum() > 0:
            axes[0].scatter(graph_dist_flat[short_mask], witness_sim_flat[short_mask], 
                           alpha=0.3, s=10)
            axes[0].set_xlabel('Graph Distance')
            axes[0].set_ylabel('Witness Similarity')
            axes[0].set_title(f'Short Paths (1-4)\nCorr: {corr_short:.3f}')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Witness similarity vs Graph distance (long)
        if long_mask.sum() > 0:
            axes[1].scatter(graph_dist_flat[long_mask], witness_sim_flat[long_mask], 
                           alpha=0.3, s=10, color='orange')
            axes[1].set_xlabel('Graph Distance')
            axes[1].set_ylabel('Witness Similarity')
            axes[1].set_title(f'Long Paths (5-8)\nCorr: {corr_long:.3f}')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        print("Skipping visualization (matplotlib not available)")

    
    return {
        'corr_short': corr_short,
        'corr_long': corr_long,
        'corr_all': corr_all
    }

if __name__ == "__main__":
    # Load a trained model and analyze
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    model = TacitReasonerWithLosses(num_nodes)
    
    # For testing, just analyze random model
    print("Analyzing untrained model (baseline)...")
    results = analyze_witness_structure(model, gen, save_path="witness_analysis_untrained.png")
    print("\nResults:", results)
