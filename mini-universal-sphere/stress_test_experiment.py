import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
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
VOCAB_LIMIT = 20000 
NUM_ANCHORS = 1000   
K_NEIGHBORS = 15     # Slight increase to stabilize the mesh

# ==========================================
# 1. SETUP (Robust Loader)
# ==========================================
def load_and_norm(model_name, limit):
    print(f"Loading {model_name}...")
    
    model = None
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
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms
    return words, matrix, {w: i for i, w in enumerate(words)}

# ==========================================
# 2. BARYCENTRIC MAPPING (LLE)
# ==========================================
def calculate_stress(source_vecs, target_vecs, query_vecs, truth_vecs, k=15):
    """
    Returns the stress (1 - cos_sim) for each query word.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nbrs.fit(source_vecs)
    distances, indices = nbrs.kneighbors(query_vecs)
    
    stresses = []
    
    for i in range(len(query_vecs)):
        # Solve weights in Source
        neighbor_idxs = indices[i]
        S_local = source_vecs[neighbor_idxs].T
        target_word_source = query_vecs[i]
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(S_local, target_word_source)
        weights = reg.coef_
        
        # Project to Target
        T_local = target_vecs[neighbor_idxs].T
        projected_word = np.dot(T_local, weights)
        
        if np.linalg.norm(projected_word) > 0:
            projected_word /= np.linalg.norm(projected_word)
        
        # Calculate Stress (1 - Cosine Similarity)
        # Cosine Sim is just dot product because normalized
        sim = np.dot(projected_word, truth_vecs[i])
        stresses.append(1.0 - sim)
        
    return np.array(stresses)

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    # A. Load
    ref_words, ref_mat, w2i_ref = load_and_norm(REF_MODEL_NAME, VOCAB_LIMIT)
    tgt_words, tgt_mat, w2i_tgt = load_and_norm(TGT_MODEL_NAME, VOCAB_LIMIT)

    if ref_mat is None or tgt_mat is None:
        print("Failed to load models.")
        exit(1)

    # B. Intersection
    common_set = set(ref_words).intersection(set(tgt_words))
    # Sort by frequency (heuristic: index in ref model ~ frequency)
    common_list = sorted(list(common_set), key=lambda w: w2i_ref[w])
    print(f"Common Vocab: {len(common_list)}")

    # C. Partition
    # Anchors: Top 1000 most frequent words (The Grid)
    anchors = common_list[:NUM_ANCHORS]
    # Test: Next 5000 words (The Candidates)
    test_words = common_list[NUM_ANCHORS:NUM_ANCHORS+5000]

    X_src_anchors = np.array([tgt_mat[w2i_tgt[w]] for w in anchors]) # GloVe
    X_tgt_anchors = np.array([ref_mat[w2i_ref[w]] for w in anchors]) # Word2Vec

    X_src_test = np.array([tgt_mat[w2i_tgt[w]] for w in test_words])
    X_tgt_truth = np.array([ref_mat[w2i_ref[w]] for w in test_words])

    print(f"Stress Test: Analyzing {len(test_words)} words...")

    # D. Run Stress Calculation
    stress_scores = calculate_stress(X_src_anchors, X_tgt_anchors, X_src_test, X_tgt_truth, k=K_NEIGHBORS)

    # ==========================================
    # ANALYSIS
    # ==========================================

    # 1. The Stable Core (Lowest Stress)
    sorted_indices = np.argsort(stress_scores)
    print("\n--- THE STABLE CORE (Universals) ---")
    for i in range(20):
        idx = sorted_indices[i]
        print(f"{i+1}. {test_words[idx]:<15} (Stress: {stress_scores[idx]:.3f})")

    # 2. The Torn Periphery (Highest Stress)
    print("\n--- THE TORN PERIPHERY (Context-Dependent) ---")
    for i in range(20):
        idx = sorted_indices[-(i+1)]
        print(f"{i+1}. {test_words[idx]:<15} (Stress: {stress_scores[idx]:.3f})")

    # 3. Frequency vs Stress Correlation
    # Is stress just because words are rare?
    # Frequency rank is just the index in our sorted list + offset
    freq_ranks = np.arange(len(test_words)) + NUM_ANCHORS
    correlation, _ = pearsonr(freq_ranks, stress_scores)
    print(f"\nCorrelation (Freq Rank vs Stress): {correlation:.3f}")
    print("(Positive = Rarer words have higher stress. Low value = Bias is structural, not just paucity.)")

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    # Plot scatter
    plt.scatter(freq_ranks, stress_scores, alpha=0.1, s=2, c='blue')

    # Highlight specific interesting words
    highlight_words = ["apple", "run", "bank", "charge", "democracy", "freedom", "physics", "paris"]
    for w in highlight_words:
        if w in test_words:
            idx = test_words.index(w)
            rank = idx + NUM_ANCHORS
            stress = stress_scores[idx]
            plt.scatter(rank, stress, c='red', s=50)
            plt.annotate(w, (rank, stress), fontsize=12, fontweight='bold')

    plt.axhline(0.4, color='k', linestyle='--', label='Stability Threshold')
    plt.xlabel("Frequency Rank (Lower = More Common)")
    plt.ylabel("Semantic Stress (Higher = More Torn)")
    plt.title("Map of the Universal Sphere: Stability vs Frequency")
    plt.legend()
    plt.ylim(0, 1.2)
    plt.savefig("stress_map.png")
    print("Plot saved to 'stress_map.png'")

    # 5. The "Swiss Cheese" Ratio
    stable_count = np.sum(stress_scores < 0.4)
    total_count = len(stress_scores)
    print(f"\nUniversal Sphere Coverage: {stable_count}/{total_count} ({stable_count/total_count:.1%})")
    print("This is the percentage of concepts that are geometrically invariant across corpora.")
