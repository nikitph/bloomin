"""Improved experiment with configurable variants."""

import numpy as np
import matplotlib.pyplot as plt
import sys
from config_improved import VARIANTS


def generate_hierarchical_data(config):
    """Generate hierarchical data with config parameters."""
    np.random.seed(42)
    
    roots = np.random.randn(config["CLUSTER_BRANCHING"], config["DIM_INPUT"]) * 5.0
    
    def recurse(centers, current_depth):
        if current_depth == config["HIERARCHY_DEPTH"]:
            return centers
        
        new_centers = []
        for center in centers:
            sigma = 1.0 / (current_depth + 1)
            children = center + np.random.randn(
                config["CLUSTER_BRANCHING"], 
                config["DIM_INPUT"]
            ) * sigma
            new_centers.extend(children)
        
        return recurse(new_centers, current_depth + 1)
    
    leaf_centers = recurse(roots, 0)
    data = []
    points_per_leaf = config["N_SAMPLES"] // len(leaf_centers)
    
    for lc in leaf_centers:
        points = lc + np.random.randn(points_per_leaf, config["DIM_INPUT"]) * 0.1
        data.append(points)
    
    remainder = config["N_SAMPLES"] - len(leaf_centers) * points_per_leaf
    if remainder > 0:
        extra_points = leaf_centers[0] + np.random.randn(remainder, config["DIM_INPUT"]) * 0.1
        data.append(extra_points)
    
    data = np.vstack(data)
    
    print(f"Generated {len(data)} data points with {len(leaf_centers)} leaf clusters")
    return data


def compute_affinities(data, prototypes, temperature=1.0):
    """Compute soft affinities with temperature parameter."""
    distances = np.sum(data**2, axis=1, keepdims=True) + \
                np.sum(prototypes**2, axis=1) - \
                2 * np.dot(data, prototypes.T)
    
    affinities = np.exp(-distances / temperature)
    affinities = affinities / (np.sum(affinities, axis=1, keepdims=True) + 1e-10)
    
    return affinities


def extract_micro_witnesses(data, config):
    """Initialize witnesses with config parameters."""
    from sklearn.cluster import KMeans
    
    print(f"\nInitializing {config['L_MICRO']} microscopic witnesses...")
    
    kmeans = KMeans(
        n_clusters=config["L_MICRO"],
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(data)
    prototypes = kmeans.cluster_centers_
    
    witness_distributions = compute_affinities(
        data, prototypes, 
        temperature=config["AFFINITY_TEMPERATURE"]
    )
    
    print(f"Witness initialization complete")
    print(f"  Mean affinity: {np.mean(witness_distributions):.6f}")
    print(f"  Max affinity: {np.max(witness_distributions):.6f}")
    
    return witness_distributions, prototypes


def renormalization_step(current_distributions, target_size):
    """RG blocking operator."""
    from sklearn.cluster import AgglomerativeClustering
    
    n_samples, n_witnesses = current_distributions.shape
    print(f"  Renormalizing: {n_witnesses} -> {target_size} witnesses")
    
    # Compute witness-witness similarity
    norms = np.linalg.norm(current_distributions, axis=0, keepdims=True) + 1e-10
    normalized = current_distributions / norms
    gram = np.dot(normalized.T, normalized)
    
    # Cluster witnesses
    distance_matrix = 1.0 - gram
    np.fill_diagonal(distance_matrix, 0)
    
    clustering = AgglomerativeClustering(
        n_clusters=target_size,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # Create macroscopic packets
    new_distributions = np.zeros((n_samples, target_size))
    
    for cluster_id in range(target_size):
        indices = np.where(labels == cluster_id)[0]
        new_distributions[:, cluster_id] = np.sum(
            current_distributions[:, indices], 
            axis=1
        )
    
    # Normalize
    row_sums = np.sum(new_distributions, axis=1, keepdims=True) + 1e-10
    new_distributions = new_distributions / row_sums
    
    return new_distributions


def measure_couplings(distributions, data):
    """Measure RG flow observables."""
    n_samples, L = distributions.shape
    
    # Signal strength
    pairwise_overlaps = np.dot(distributions, distributions.T)
    mask = ~np.eye(n_samples, dtype=bool)
    rho = np.mean(pairwise_overlaps[mask])
    
    # Curvature proxy
    eigenvalues = np.linalg.svd(pairwise_overlaps, compute_uv=False)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    PR = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
    K_proxy = 1.0 / (PR + 1e-10)
    
    # Thermodynamic variable
    Delta = rho
    m = L
    Chi = (m * Delta**2) / (np.log(n_samples) + 1e-10)
    
    return {
        "L": L,
        "rho": rho,
        "K": K_proxy,
        "Chi": Chi,
        "PR": PR
    }


def evaluate_retrieval_accuracy(distributions, data, k=10):
    """Evaluate Recall@k."""
    n_samples = len(data)
    
    # Ground truth neighbors
    data_distances = np.sum(data**2, axis=1, keepdims=True) + \
                     np.sum(data**2, axis=1) - \
                     2 * np.dot(data, data.T)
    true_neighbors = np.argsort(data_distances, axis=1)[:, :k+1]
    
    # Witness space neighbors
    witness_similarities = np.dot(distributions, distributions.T)
    witness_distances = -witness_similarities
    pred_neighbors = np.argsort(witness_distances, axis=1)[:, :k+1]
    
    # Compute recall
    recalls = []
    for i in range(n_samples):
        true_set = set(true_neighbors[i, 1:])
        pred_set = set(pred_neighbors[i, 1:])
        recall = len(true_set & pred_set) / k
        recalls.append(recall)
    
    return np.mean(recalls)


def run_variant(variant_name, config):
    """Run experiment with specific configuration."""
    print("\n" + "=" * 70)
    print(f"VARIANT: {variant_name.upper()}")
    print("=" * 70)
    print(f"Config: L={config['L_MICRO']}, compression={config['COMPRESSION_FACTOR']}, "
          f"temp={config['AFFINITY_TEMPERATURE']}, depth={config['HIERARCHY_DEPTH']}")
    
    # Generate data
    data = generate_hierarchical_data(config)
    
    # Initialize witnesses
    current_dist, _ = extract_micro_witnesses(data, config)
    
    # Track history
    history = []
    
    print("\nStarting RG Flow...")
    
    # Initial measurement
    obs = measure_couplings(current_dist, data)
    acc = evaluate_retrieval_accuracy(current_dist, data, k=10)
    print(f"Scale 0: L={obs['L']}, rho={obs['rho']:.6f}, Chi={obs['Chi']:.6f}, Acc={acc:.4f}")
    history.append({**obs, "scale": 0, "acc": acc})
    
    # RG flow
    for step in range(1, config["RG_STEPS"] + 1):
        current_L = current_dist.shape[1]
        target_L = int(current_L * config["COMPRESSION_FACTOR"])
        
        if target_L < 4:
            break
        
        print(f"\nScale {step}:")
        current_dist = renormalization_step(current_dist, target_L)
        
        obs = measure_couplings(current_dist, data)
        acc = evaluate_retrieval_accuracy(current_dist, data, k=10)
        print(f"  L={obs['L']}, rho={obs['rho']:.6f}, Chi={obs['Chi']:.6f}, Acc={acc:.4f}")
        
        history.append({**obs, "scale": step, "acc": acc})
    
    return history


def compare_variants(results):
    """Generate comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RG Flow Variant Comparison", fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#BC4B51', '#8B5A3C']
    
    for idx, (variant_name, history) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        scales = [h["scale"] for h in history]
        
        # Chi stability
        ax = axes[0, 0]
        Chi_vals = [h["Chi"] for h in history]
        ax.plot(scales, Chi_vals, 'o-', label=variant_name, 
                linewidth=2, markersize=6, color=color)
        
        # Accuracy preservation
        ax = axes[0, 1]
        acc_vals = [h["acc"] for h in history]
        ax.plot(scales, acc_vals, 'o-', label=variant_name,
                linewidth=2, markersize=6, color=color)
        
        # Signal concentration
        ax = axes[1, 0]
        rho_vals = [h["rho"] for h in history]
        ax.plot(scales, rho_vals, 'o-', label=variant_name,
                linewidth=2, markersize=6, color=color)
        
        # Compression vs Accuracy
        ax = axes[1, 1]
        L_vals = [h["L"] for h in history]
        compression = [L_vals[0] / L for L in L_vals]
        ax.plot(compression, acc_vals, 'o-', label=variant_name,
                linewidth=2, markersize=6, color=color)
    
    # Configure axes
    axes[0, 0].set_xlabel("RG Scale")
    axes[0, 0].set_ylabel("χ (Thermodynamic Variable)")
    axes[0, 0].set_title("Fixed Point Stability")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("RG Scale")
    axes[0, 1].set_ylabel("Recall@10")
    axes[0, 1].set_title("Accuracy Preservation")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    axes[1, 0].set_xlabel("RG Scale")
    axes[1, 0].set_ylabel("Signal Strength ρ")
    axes[1, 0].set_title("Signal Concentration")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel("Compression Ratio")
    axes[1, 1].set_ylabel("Recall@10")
    axes[1, 1].set_title("Compression-Accuracy Trade-off")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('variant_comparison.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: variant_comparison.png")


def print_summary(results):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("VARIANT COMPARISON SUMMARY")
    print("=" * 70)
    
    for variant_name, history in results.items():
        Chi_vals = [h["Chi"] for h in history]
        acc_vals = [h["acc"] for h in history]
        L_vals = [h["L"] for h in history]
        
        chi_mean = np.mean(Chi_vals)
        chi_std = np.std(Chi_vals)
        chi_stability = chi_std / chi_mean if chi_mean > 0 else float('inf')
        
        print(f"\n{variant_name.upper()}:")
        print(f"  Initial accuracy: {acc_vals[0]:.4f}")
        print(f"  Final accuracy:   {acc_vals[-1]:.4f}")
        print(f"  Accuracy drop:    {acc_vals[0] - acc_vals[-1]:.4f}")
        print(f"  Chi stability:    {chi_stability:.4f} (std/mean)")
        print(f"  Compression:      {L_vals[0] / L_vals[-1]:.1f}x")


if __name__ == "__main__":
    # Select variants to run
    if len(sys.argv) > 1:
        variant_names = sys.argv[1:]
    else:
        # Run all variants
        variant_names = ["baseline", "optimized", "more_witnesses", "gentle_compression"]
    
    results = {}
    
    for variant_name in variant_names:
        if variant_name not in VARIANTS:
            print(f"Unknown variant: {variant_name}")
            continue
        
        config = VARIANTS[variant_name]
        history = run_variant(variant_name, config)
        results[variant_name] = history
    
    # Generate comparison
    if len(results) > 1:
        compare_variants(results)
    
    print_summary(results)
