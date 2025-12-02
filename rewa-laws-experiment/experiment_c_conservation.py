import numpy as np
from synthetic_data import make_synthetic

def compute_pairwise_mi_estimate(items, W):
    """
    Estimates total pairwise MI for a subset of items.
    Using the closed form for Boolean sets:
    I(A; B) = log( |A|*|B| / |A u B| * W ) ? 
    Wait, prompt says: I = log( |A||B| / |A u B| ). 
    Actually, standard MI for sets A, B in universe U is:
    p11 = |A n B|/W, p10 = |A-B|/W, p01 = |B-A|/W, p00 = |U-(AuB)|/W
    I = sum p_xy log(p_xy / (p_x * p_y))
    
    The prompt simplifies: "I(Wx;Wy) = log( |A||B| / |A u B| )" 
    This looks like a specific approximation or definition used in their theory (Overlap-Information Isomorphism).
    Let's stick to the prompt's formula: I = log( L^2 / (2L - Delta) ) for fixed sizes.
    
    We will compute sum of I(Wi, Wj) for all pairs.
    """
    N = len(items)
    total_mi = 0.0
    count = 0
    
    # We'll sample pairs to estimate the sum, as N^2 is large
    num_pairs = 5000
    
    for _ in range(num_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        A = set(items[i])
        B = set(items[j])
        
        # Intersection
        intersect = len(A & B)
        union = len(A | B)
        
        # Prompt formula: log( |A||B| / |A u B| )
        # Note: This formula assumes specific conditions, but let's use it as the metric to test invariance.
        # If T is a permutation, |A|, |B|, |A u B| are ALL preserved.
        # So this metric is trivially invariant under permutation.
        
        if union > 0:
            val = np.log2( (len(A) * len(B)) / union )
            total_mi += val
        count += 1
        
    return total_mi / count

def run_experiment_c():
    print("Running Experiment C: Conservation...")
    
    N = 1000
    k = 10
    W = 10000
    L = 100
    rho = 0.1
    
    # 1. Generate Base System
    items, _ = make_synthetic(N, k, W, L, rho, seed=42)
    print("System generated.")
    
    # 2. Measure MI
    mi_original = compute_pairwise_mi_estimate(items, W)
    print(f"Original Mean Pairwise MI: {mi_original:.4f}")
    
    # 3. Apply Permutation T (Bijective)
    # T: W -> W (shuffle witness IDs)
    perm = np.random.permutation(W)
    items_permuted = []
    for item in items:
        items_permuted.append(perm[item]) # Map indices
        
    mi_permuted = compute_pairwise_mi_estimate(items_permuted, W)
    print(f"Permuted Mean Pairwise MI: {mi_permuted:.4f}")
    
    diff = abs(mi_original - mi_permuted)
    print(f"Difference: {diff:.6f}")
    if diff < 0.01:
        print("SUCCESS: Invariance confirmed.")
    else:
        print("FAILURE: MI changed!")
        
    # 4. Apply Non-Invertible Map (Collapse)
    # Map first 5000 witnesses to 0..2499 (2-to-1)
    # This reduces the witness universe effective size and causes collisions
    print("\nApplying Non-Invertible Map (Collapse)...")
    items_collapsed = []
    for item in items:
        new_item = []
        for w in item:
            # Aggressive collapse: modulo 100
            new_item.append(w % 100)
        items_collapsed.append(np.unique(new_item)) # Sets merge
        
    mi_collapsed = compute_pairwise_mi_estimate(items_collapsed, 100) # W is now effectively 100
    print(f"Collapsed Mean Pairwise MI: {mi_collapsed:.4f}")
    
    if abs(mi_collapsed - mi_original) > 0.01:
        print("SUCCESS: MI changed under non-invertible map.")
    else:
        print("FAILURE: MI did not change significantly.")

if __name__ == "__main__":
    run_experiment_c()
