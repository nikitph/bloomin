import numpy as np
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import os
from gensim.models import KeyedVectors
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# CONFIGURATION
# ==========================================
REF_MODEL_NAME = "word2vec-google-news-300" 
TGT_MODEL_NAME = "glove-wiki-gigaword-300"
VOCAB_LIMIT = 15000 
NUM_ANCHORS = 1000   # Increased density for the Lattice
K_NEIGHBORS = 10     # How many witnesses define a concept?

# ==========================================
# 1. HELPER: LOAD & NORMALIZE
# ==========================================
def load_and_norm(model_name, limit):
    print(f"Loading {model_name}...")
    
    model = None
    # Manual load fallback logic (copied from rigid_body_experiment.py)
    manual_base = os.path.expanduser(f"~/gensim-data/{model_name}/{model_name}.gz")
    if os.path.exists(manual_base):
        print(f"Found local file at: {manual_base}")
        is_binary = "word2vec" in model_name or model_name.endswith(".bin.gz")
        try:
            model = KeyedVectors.load_word2vec_format(manual_base, binary=is_binary)
        except Exception as e:
            print(f"Binary load failed, trying text: {e}")
            model = KeyedVectors.load_word2vec_format(manual_base, binary=False)
            
    if model is None:
        print("Local file not found or failed. Attempting gensim download...")
        try:
            model = api.load(model_name)
        except Exception as e:
             print(f"Error loading {model_name}: {e}")
             return None, None, None

    words = []
    vecs = []
    count = 0
    for w in model.index_to_key:
        if w.isalpha() and w.islower():
            words.append(w)
            vecs.append(model[w])
            count += 1
            if count >= limit:
                break
    
    matrix = np.array(vecs)
    # Spherical Normalization
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms
    return words, matrix, {w: i for i, w in enumerate(words)}

# ==========================================
# 2. LATTICE MAPPING (BARYCENTRIC)
# ==========================================
def learn_barycentric_mapping(source_vecs, target_vecs, query_vecs, k=10):
    """
    1. Finds k-nearest anchors in Source for each Query.
    2. Learns linear weights to reconstruct Query from those anchors.
    3. Applies those weights to the Target anchors to hallucinate the result.
    """
    print(f"Constructing Lattice with k={k} neighbors...")
    
    # A. Fit NN on Source Anchors
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nbrs.fit(source_vecs)
    
    # B. Find neighbors for queries
    distances, indices = nbrs.kneighbors(query_vecs)
    
    reconstructed_vecs = []
    
    # C. Solve for weights locally (LLE style)
    # For each query word, we want: w_source approx sum(c_i * anchor_source_i)
    # Then: w_target = sum(c_i * anchor_target_i)
    
    for i in range(len(query_vecs)):
        # Get the k neighbors for this specific word
        neighbor_idxs = indices[i]
        
        # Source Anchors (The basis for this word)
        S_local = source_vecs[neighbor_idxs].T  # Shape (300, k)
        target_word_source = query_vecs[i]      # Shape (300,)
        
        # Solve S_local * c = target_word_source
        # We use Least Squares
        reg = LinearRegression(fit_intercept=False)
        reg.fit(S_local, target_word_source)
        weights = reg.coef_
        
        # D. Project into Universe (Target Space)
        # Get the SAME anchors but in the Target Space
        T_local = target_vecs[neighbor_idxs].T  # Shape (300, k)
        
        # Synthesize: w_target = T_local * weights
        projected_word = np.dot(T_local, weights)
        
        # Renormalize (Project back to sphere)
        if np.linalg.norm(projected_word) > 1e-9:
            projected_word /= np.linalg.norm(projected_word)
            
        reconstructed_vecs.append(projected_word)
        
    return np.array(reconstructed_vecs)

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    # A. Load
    ref_words, ref_mat, w2i_ref = load_and_norm(REF_MODEL_NAME, VOCAB_LIMIT)
    tgt_words, tgt_mat, w2i_tgt = load_and_norm(TGT_MODEL_NAME, VOCAB_LIMIT)

    if ref_mat is None or tgt_mat is None:
        print("Failed to load models. Exiting.")
        exit(1)

    # B. Common Vocab
    common_set = set(ref_words).intersection(set(tgt_words))
    common_list = sorted(list(common_set))
    print(f"Common Vocab: {len(common_list)}")

    # C. Select Anchors & Test Set
    # We just take the first N as anchors (frequency based) for the Lattice
    anchors = common_list[:NUM_ANCHORS]
    test_words = common_list[NUM_ANCHORS:NUM_ANCHORS+1000]

    # Build Matrices
    X_src_anchors = np.array([tgt_mat[w2i_tgt[w]] for w in anchors]) # Source = GloVe
    X_tgt_anchors = np.array([ref_mat[w2i_ref[w]] for w in anchors]) # Target = Word2Vec (Universe)

    X_src_test = np.array([tgt_mat[w2i_tgt[w]] for w in test_words])
    X_tgt_test_truth = np.array([ref_mat[w2i_ref[w]] for w in test_words])

    print(f"Lattice: Mapping {len(test_words)} words from GloVe -> Word2Vec using {NUM_ANCHORS} anchors.")

    # D. Run Lattice Mapping
    X_projected = learn_barycentric_mapping(X_src_anchors, X_tgt_anchors, X_src_test, k=K_NEIGHBORS)

    # E. Evaluate
    # We measure Cosine Similarity between Projected and Truth
    cos_sims = np.sum(X_projected * X_tgt_test_truth, axis=1)
    mean_sim = np.mean(cos_sims)

    print("\n" + "="*40)
    print("RESULTS: ELASTIC LATTICE (BARYCENTRIC)")
    print("="*40)
    print(f"Mean Cosine Similarity: {mean_sim:.4f}")
    print("(1.0 = Perfect, 0.0 = Orthogonal/Fail)")

    # F. Spot Check
    print("\n--- Spot Check (Lattice Reconstruction) ---")
    check_indices = [0, 10, 50, 100, 200]
    for idx in check_indices:
        if idx < len(test_words):
            w = test_words[idx]
            sim = cos_sims[idx]
            print(f"Word: '{w:<15}' | Recovered Sim: {sim:.3f}")

    if mean_sim > 0.6:
        print("\n✅ STATUS: SUCCESS.")
        print("   The Lattice preserves semantic topology locally.")
        print("   This proves the Universal Sphere is a MESH, not a Matrix.")
    else:
        print("\n⛔ STATUS: FAILURE.")
        print("   Local topology is also divergent.")
