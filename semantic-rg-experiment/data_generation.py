"""Hierarchical data generation for RG experiment."""

import numpy as np
from config import CONFIG


def generate_hierarchical_data():
    """
    Generate tree-structured Gaussian clusters.
    This simulates "Concepts -> Sub-concepts -> Instances".
    
    Returns:
        np.ndarray: Data points of shape (N_SAMPLES, DIM_INPUT)
    """
    np.random.seed(42)
    
    # Generate root centers
    roots = np.random.randn(CONFIG["CLUSTER_BRANCHING"], CONFIG["DIM_INPUT"]) * 5.0
    
    def recurse(centers, current_depth):
        """Recursively generate hierarchical clusters."""
        if current_depth == CONFIG["HIERARCHY_DEPTH"]:
            return centers  # Leaf centers
        
        new_centers = []
        for center in centers:
            # Spawn smaller, tighter clusters around parent
            # Variance decreases with depth
            sigma = 1.0 / (current_depth + 1)
            children = center + np.random.randn(
                CONFIG["CLUSTER_BRANCHING"], 
                CONFIG["DIM_INPUT"]
            ) * sigma
            new_centers.extend(children)
        
        return recurse(new_centers, current_depth + 1)
    
    # Get leaf centers
    leaf_centers = recurse(roots, 0)
    
    # Sample actual points around leaf centers
    data = []
    points_per_leaf = CONFIG["N_SAMPLES"] // len(leaf_centers)
    
    for lc in leaf_centers:
        # Sample points around this leaf center
        points = lc + np.random.randn(points_per_leaf, CONFIG["DIM_INPUT"]) * 0.1
        data.append(points)
    
    # Handle remainder
    remainder = CONFIG["N_SAMPLES"] - len(leaf_centers) * points_per_leaf
    if remainder > 0:
        extra_points = leaf_centers[0] + np.random.randn(remainder, CONFIG["DIM_INPUT"]) * 0.1
        data.append(extra_points)
    
    data = np.vstack(data)
    
    print(f"Generated {len(data)} data points with {len(leaf_centers)} leaf clusters")
    print(f"Hierarchy: {CONFIG['CLUSTER_BRANCHING']}^{CONFIG['HIERARCHY_DEPTH']} = {len(leaf_centers)} leaves")
    
    return data


if __name__ == "__main__":
    # Test data generation
    data = generate_hierarchical_data()
    print(f"Data shape: {data.shape}")
    print(f"Data mean: {np.mean(data):.4f}, std: {np.std(data):.4f}")
