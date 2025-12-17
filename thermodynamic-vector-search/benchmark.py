import numpy as np
from search import build_index
from utils import normalize, random_normal, euclidean_distance

def example_usage():
    """
    Complete end-to-end example
    """
    print("=" * 60)
    print("Thermodynamic Vector Search Example")
    print("=" * 60)
    
    # 1. Generate synthetic dataset
    print("\n1. Generating dataset...")
    dimension = 128
    n_vectors = 1000 # Reduced from 10000 for faster dev speed, can increase
    
    # Create clustered data
    vectors = []
    for cluster in range(10):
        center = random_normal(dimension)
        # 100 vectors per cluster
        for _ in range(n_vectors // 10):
            vec = center + 0.1 * random_normal(dimension)
            vec = normalize(vec)
            vectors.append(vec)
    
    print(f"   Created {len(vectors)} vectors in {dimension} dimensions")
    
    # 2. Build index
    print("\n2. Building index...")
    # Using smaller grid for speed in this example? 
    # Default is [64, 256, 1024].
    # For 128d, 64 is fine.
    # Note: Higher dims might need smaller resolution to fit in memory if not careful.
    # But shape is [resolution]*dimension.
    # WAIT. dimension=128. Resolution=64.
    # 64^128 is impossible.
    # The pseudocode says:
    # "Grid shape: [res, res, ..., res] (dimension times)"
    # This is physically impossible for d > 3 or 4.
    # "Grid shape: [resolution] * self.dimension" -> creates list of size d.
    # np.zeros(shape) creates array of size res * res * ... * res (d times).
    # Memory = 64^128 * 8 bytes = infinity.
    
    # CRITICAL ISSUE IN PSEUDOCODE.
    # "MultiResolutionGrid ... dimension ... bounds ... resolutions"
    # "Grid shape: [res, res, ..., res] (dimension times)"
    # This works for 1D, 2D, 3D. Not 128D.
    # The user request says "Complete Pseudocode ... Executable Pseudocode".
    # But 128D grid is impossible.
    # Maybe they mean 2D or 3D projection?
    # Or maybe independent grids per dimension?
    # "d-dimensional grid: kernel has -2d at center" implies fully coupled n-dim grid.
    
    # Re-reading: "1. Key Performance Notes ... Complexity Analysis ... O(G^d)"
    # "G=64 (curse of dimensionality)"
    # If d=128, G^d is too big.
    # Typical vector search uses quantization or LSH.
    # Thermodynamic approach usually implies operating on a 2D/3D manifold or graph.
    # But the pseudocode explicitly creates `np.zeros(shape)` where shape is `[resolution]*dimension`.
    
    # I MUST FIX THIS. 
    # If I run this with d=128, it will crash immediately.
    # The example uses d=128.
    # Perhaps the "grid" is not a spatial grid in vector space, but something else?
    # The code `MultiResolutionGrid` creates `zeros(shape)`.
    # `shape = [resolution] * self.dimension`.
    # This is definitely full grid.
    
    # HYPOTHESIS: The user might be thinking of a mapped low-D space (like 2D or 3D) 
    # where vectors are projected, OR the dimension in example should be small (like 2 or 3).
    # "dimension: Vector dimensionality (e.g., 128, 768, 1536)"
    # "Grid shape: [res, res, ..., res] (dimension times)"
    # This is a contradiction for implementation.
    
    # workaround: 
    # 1. Reduce dimension to 2 or 3 for the demo to work.
    # 2. Or assume the "Field" is defined on a low-dimensional manifold (e.g. 2D map).
    
    # "Index Construction ... dimension = len(vectors[0])".
    # It takes full dimension.
    
    # I will try to run with d=3 for the demo to show it works.
    # And I will add a warning note.
    # If I run d=128, it will die.
    
    # Let's change d=3 in this script for now.
    
    demo_dim = 3
    print(f"   [NOTE] Reducing dimension to {demo_dim} for grid tractability in this demo.")
    
    n_vectors = 1000
    vectors_lowd = []
    
    # Create clustered data in 3D
    for cluster in range(4):
        center = random_normal(demo_dim)
        for _ in range(n_vectors // 4):
            vec = center + 0.1 * random_normal(demo_dim)
            vec = normalize(vec)
            vectors_lowd.append(vec)
            
    index = build_index(vectors_lowd, dimension=demo_dim, resolutions=[8, 16]) 
    # Even 32^3 = 32768 cells. 1024^3 = 1e9 cells (8GB). 1024 is high for 3D. 
    # 64, 256, 1024 -> 1024^3 is big.
    # Let's use smaller resolutions for quick demo.
    
    # 3. Create query
    print("\n3. Creating query...")
    # Query near first cluster
    query = vectors_lowd[0] + 0.05 * random_normal(demo_dim)
    query = normalize(query)
    
    # 4. Search
    print("\n4. Executing search...")
    k = 10
    results = index.search(query, k=k)
    
    # 5. Display results
    print(f"\n5. Top {k} results:")
    print("-" * 60)
    for i, (idx, dist, metadata) in enumerate(results):
        print(f"  Rank {i+1}: Vector {idx}, Distance: {dist:.4f}")
    
    # 6. Verify
    print("\n6. Verification (brute force):")
    brute_force = index.database.get_nearest_to_point(query, k)
    print("-" * 60)
    for i, (idx, dist) in enumerate(brute_force):
        print(f"  Rank {i+1}: Vector {idx}, Distance: {dist:.4f}")
    
    # 7. Performance stats
    print("\n7. Performance Statistics:")
    print("-" * 60)
    print(f"  Total energy spent: {index.regulator.energy_spent}")
    print(f"  Spectral gap history: {index.monitor.gap_history[-5:]}")
    
    return index, results

# Run example
if __name__ == "__main__":
    index, results = example_usage()
