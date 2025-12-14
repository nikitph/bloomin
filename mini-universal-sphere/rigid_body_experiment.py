import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_distances
import os
from gensim.models import KeyedVectors
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# CONFIGURATION
# ==========================================
# We use 300d models for both to ensure a rigorous "Rigid Body" test.
# This requires ~2GB RAM.
REF_MODEL_NAME = "word2vec-google-news-300" 
TGT_MODEL_NAME = "glove-wiki-gigaword-300"
VOCAB_LIMIT = 15000  # Limit vocab size for speed
NUM_ANCHORS = 500    # The "Grid Points" (was 5 in previous exp)
NUM_TEST = 1000      # Held-out words for validation

# ==========================================
# 1. HELPER: LOAD & NORMALIZE
# ==========================================
def load_and_norm(model_name, limit):
    print(f"Loading {model_name}...")
    
    model = None
    # Manual load fallback logic
    manual_base = os.path.expanduser(f"~/gensim-data/{model_name}/{model_name}.gz")
    if os.path.exists(manual_base):
        print(f"Found local file at: {manual_base}")
        # Word2Vec binary vs GloVe text format check
        is_binary = "word2vec" in model_name or model_name.endswith(".bin.gz")
        # GloVe is usually not binary in gensim-data, but check extension
        # Typically gensim-data gloves are converted to w2v format txt
        try:
            model = KeyedVectors.load_word2vec_format(manual_base, binary=is_binary)
        except Exception as e:
            # Fallback for glove which might be non-binary
            print(f"Binary load failed, trying text: {e}")
            model = KeyedVectors.load_word2vec_format(manual_base, binary=False)
            
    if model is None:
        print("Local file not found or failed. Attempting gensim download...")
        try:
            model = api.load(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None, None

    # Filter to common English words (simple heuristic: lowercase, alpha)
    # We rely on the model's sort order (usually freq)
    words = []
    vecs = []
    count = 0
    # Optimizing this loop
    # model.index_to_key gives ordered list
    for w in model.index_to_key:
        if w.isalpha() and w.islower():
            words.append(w)
            vecs.append(model[w])
            count += 1
            if count >= limit:
                break
    
    matrix = np.array(vecs)
    # Axiom 0.2: Spherical Normalization
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms
    
    return words, matrix, model

# ==========================================
# 2. ALGORITHM: MAXIMAL VOLUMETRIC DISPERSION
# ==========================================
def select_anchors_fps(matrix, words, k=500):
    """
    Selects k anchors using Furthest Point Sampling (FPS).
    This ensures the anchors span the convex hull of the semantic space,
    locking the geometry from the 'outside in'.
    """
    print(f"Selecting {k} Anchors via Maximal Volumetric Dispersion...")
    n_samples = matrix.shape[0]
    indices = [0] # Start with the most frequent word (usually 'the')
    
    # Initialize distances to the first point
    # We work with Cosine Distance (1 - dot) since we are on sphere
    # dists[i] = min distance from point i to any selected anchor
    
    # Efficient computation:
    # 1. Calculate dist matrix from current selection to all points
    # 2. Update min_dists
    
    # Pre-compute all pairwise is too heavy. We do it iteratively.
    min_dists = cosine_distances(matrix[indices], matrix).flatten()
    
    for _ in range(1, k):
        # The next anchor is the one furthest from the current set
        new_idx = np.argmax(min_dists)
        indices.append(new_idx)
        
        # Update distances: min(existing_dist, dist_to_new_point)
        new_dists = cosine_distances(matrix[[new_idx]], matrix).flatten()
        min_dists = np.minimum(min_dists, new_dists)
        
        if len(indices) % 50 == 0:
            print(f"  Selected {len(indices)}/{k}: '{words[new_idx]}'")
            
    return [words[i] for i in indices]

# ==========================================
# 3. CORE: RIGID BODY ALIGNMENT (PROCRUSTES)
# ==========================================
def align_spaces(ref_vecs, tgt_vecs):
    """
    Solves for Orthogonal Matrix Q such that Tgt @ Q approx Ref.
    Minimizes Frobenius norm || Ref - Tgt @ Q ||
    """
    # R @ T.T = U @ S @ Vt
    # Q = U @ Vt
    Q, scale = orthogonal_procrustes(ref_vecs, tgt_vecs)
    
    # Note: scipy returns R such that A = B @ R. 
    # Check docs: orthogonal_procrustes(A, B) maps B to A?
    # Actually: "min || A - B @ R ||". So maps B (target) -> A (ref).
    return Q.T # We return the matrix to multiply Tgt by.

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # A. Load Models
    ref_words, ref_mat, _ = load_and_norm(REF_MODEL_NAME, VOCAB_LIMIT)
    tgt_words, tgt_mat, _ = load_and_norm(TGT_MODEL_NAME, VOCAB_LIMIT)

    if ref_mat is None or tgt_mat is None:
        print("Failed to load models. Exiting.")
        exit(1)

    # B. Find Intersection (Common Vocabulary)
    common_set = set(ref_words).intersection(set(tgt_words))
    common_list = sorted(list(common_set))
    print(f"Common Vocabulary Size: {len(common_list)}")

    # Build Common Matrices
    w2i_ref = {w: i for i, w in enumerate(ref_words)}
    w2i_tgt = {w: i for i, w in enumerate(tgt_words)}

    ref_common = np.array([ref_mat[w2i_ref[w]] for w in common_list])
    tgt_common = np.array([tgt_mat[w2i_tgt[w]] for w in common_list])

    # C. Select Anchors (Training Set)
    # We select indices from the common list
    anchors = select_anchors_fps(ref_common, common_list, k=NUM_ANCHORS)
    anchor_indices = [common_list.index(w) for w in anchors]

    # Separate into Train (Anchors) and Test (Held-out)
    mask_anchor = np.zeros(len(common_list), dtype=bool)
    mask_anchor[anchor_indices] = True

    X_train_ref = ref_common[mask_anchor]
    X_train_tgt = tgt_common[mask_anchor]

    X_test_ref = ref_common[~mask_anchor][:NUM_TEST] # Take first N held-out
    X_test_tgt = tgt_common[~mask_anchor][:NUM_TEST]

    print(f"\nAlignment Task: Map GloVe -> Word2Vec using {NUM_ANCHORS} anchors.")

    # D. Compute Alignment
    # 1. Baseline: No Alignment (Identity)
    diff_raw = X_test_ref - X_test_tgt
    residual_raw = np.mean(np.linalg.norm(diff_raw, axis=1))

    # 2. Alignment: Procrustes 2.0
    # Solve Q using Anchors
    R, _ = orthogonal_procrustes(X_train_ref, X_train_tgt)
    # Transform Test Set
    X_test_aligned = X_test_tgt @ R

    diff_aligned = X_test_ref - X_test_aligned
    residual_aligned = np.mean(np.linalg.norm(diff_aligned, axis=1))

    # E. Results
    print("\n" + "="*40)
    print("RESULTS: RIGID BODY CONVERGENCE TEST")
    print("="*40)
    print(f"Baseline Divergence (No Align): {residual_raw:.4f}")
    print(f"Aligned Residual (500 Anchors): {residual_aligned:.4f}")

    improvement = (residual_raw - residual_aligned) / residual_raw * 100
    print(f"Geometric Convergence Improvement: {improvement:.1f}%")

    if residual_aligned < 0.6:
        print("\n✅ STATUS: CONVERGENCE.")
        print("   The latent topologies are compatible under rotation.")
        print("   The 'Universal Sphere' Protocol is viable.")
    else:
        print("\n⚠️ STATUS: PARTIAL DIVERGENCE.")
        print("   Significant topological differences remain.")

    # F. Visualization
    # Let's verify a few random words
    print("\n--- Spot Check (Cosine Similarity after Align) ---")
    test_samples = ["science", "music", "war", "water", "system"]
    for w in test_samples:
        if w in common_list and w not in anchors:
            idx = common_list.index(w)
            v_ref = ref_common[idx]
            v_tgt_raw = tgt_common[idx]
            v_tgt_aligned = v_tgt_raw @ R
            
            sim_before = np.dot(v_ref, v_tgt_raw)
            sim_after = np.dot(v_ref, v_tgt_aligned)
            
            print(f"Word: '{w}' | Sim Before: {sim_before:.3f} | Sim After: {sim_after:.3f}")

    # Plotting the Residual Distribution
    norms_aligned = np.linalg.norm(diff_aligned, axis=1)
    
    plt.figure(figsize=(10,6))
    plt.hist(norms_aligned, bins=50, color='green', alpha=0.7, label='Aligned')
    plt.axvline(residual_aligned, color='k', linestyle='dashed', linewidth=1)
    plt.title(f"Distribution of Alignment Residuals (Mean: {residual_aligned:.3f})")
    plt.xlabel("Euclidean Distance between Word2Vec and Aligned-GloVe")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("rigid_body_residuals.png")
    print("Plot saved to 'rigid_body_residuals.png'")
