import numpy as np
import hashlib

def make_synthetic(N, k, W, L, rho, seed=42):
    """
    Generates synthetic witness sets with controlled overlap.
    
    Args:
        N: Number of items (concepts)
        k: Neighborhood size (cluster size)
        W: Witness universe size
        L: Number of witnesses per concept
        rho: Overlap fraction (0 to 1)
        seed: Random seed
        
    Returns:
        items: List of sets (or arrays) of witness indices
        cluster_labels: List of cluster IDs for each item
    """
    np.random.seed(seed)
    clusters = N // k
    items = []
    cluster_labels = []
    
    # Pre-generate shared sets for each cluster
    shared_sets = []
    for c in range(clusters):
        # Shared witnesses for this cluster
        shared_size = int(rho * L)
        if shared_size > 0:
            shared = np.random.choice(W, size=shared_size, replace=False)
        else:
            shared = np.array([], dtype=int)
        shared_sets.append(shared)
        
    for c in range(clusters):
        shared = shared_sets[c]
        shared_size = len(shared)
        
        for i in range(k):
            # Unique witnesses for this item
            # To ensure uniqueness, we pick from W excluding shared
            # Ideally we should also exclude other items' unique sets to be perfect, 
            # but for large W >> L, random choice is sufficient.
            # Let's be slightly more rigorous:
            
            needed_unique = L - shared_size
            
            # Simple rejection sampling for unique part to avoid overlap with shared
            # (Assuming W is large enough that we don't run out)
            candidates = np.random.choice(W, size=needed_unique * 2, replace=False)
            unique = []
            for cand in candidates:
                if cand not in shared:
                    unique.append(cand)
                if len(unique) == needed_unique:
                    break
            
            if len(unique) < needed_unique:
                # Fallback if rejection failed (shouldn't happen with W=10k, L=100)
                remaining = list(set(range(W)) - set(shared))
                unique = np.random.choice(remaining, size=needed_unique, replace=False)
            
            item_witnesses = np.concatenate([shared, np.array(unique, dtype=int)])
            items.append(item_witnesses)
            cluster_labels.append(c)
            
    return items, np.array(cluster_labels)

def rewa_hash_encode(items, m, seed=42):
    """
    Encodes witness sets into m-bit binary vectors using hashing.
    
    Args:
        items: List of witness sets
        m: Number of bits in the encoding
        seed: Random seed for reproducibility
        
    Returns:
        B: (N, m) binary matrix
    """
    N = len(items)
    B = np.zeros((N, m), dtype=np.uint8)
    
    # We use a simple hash function: hash(w) % m
    # To make it robust and "independent" for different m, we can salt it.
    # But for this experiment, simple modulo is often enough if witnesses are random integers.
    # Let's use a slightly better mix to avoid structural artifacts if W is ordered.
    
    # Using a fixed set of salts could simulate k-wise independence, 
    # but here we just map each witness to ONE bit (Bloom filter with k=1 hash functions per element? 
    # No, usually REWA maps each witness to one position, or multiple. 
    # The prompt says: "hash each witness -> positions". 
    # Let's assume 1 hash per witness for simplicity unless specified otherwise.
    # "hash each witness into m positions" - usually means w maps to h(w).
    
    for i, S in enumerate(items):
        for w in S:
            # Simple deterministic hash:
            # (w * large_prime + seed) % m
            h = (w * 2654435761 + seed) % m 
            B[i, h] = 1
            
    return B

def compute_overlap_stats(items, cluster_labels):
    """Helper to verify the generated overlaps."""
    N = len(items)
    # Check a few intra-cluster pairs
    intra_overlaps = []
    inter_overlaps = []
    
    # Sample some pairs
    for _ in range(1000):
        i, j = np.random.choice(N, 2, replace=False)
        overlap = len(set(items[i]) & set(items[j]))
        if cluster_labels[i] == cluster_labels[j]:
            intra_overlaps.append(overlap)
        else:
            inter_overlaps.append(overlap)
            
    return np.mean(intra_overlaps), np.mean(inter_overlaps)
