import numpy as np
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
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
VOCAB_LIMIT = 20000 
NUM_ANCHORS = 1000   
K_NEIGHBORS = 15     # For Lattice Mapping

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

def get_lattice_projection(source_vecs, target_vecs, query_vec):
    """
    Projects a single vector from Source -> Target using Lattice Interpolation.
    """
    # 1. Find neighbors in Source
    # We brute force this for single query speed (no pre-fit tree needed for demo)
    sims = np.dot(source_vecs, query_vec)
    neighbor_idxs = np.argsort(sims)[-K_NEIGHBORS:]
    
    S_local = source_vecs[neighbor_idxs].T
    T_local = target_vecs[neighbor_idxs].T
    
    # 2. Learn Weights
    reg = LinearRegression(fit_intercept=False)
    reg.fit(S_local, query_vec)
    weights = reg.coef_
    
    # 3. Project
    projected = np.dot(T_local, weights)
    if np.linalg.norm(projected) > 0:
        projected /= np.linalg.norm(projected)
        
    return projected

# ==========================================
# 2. ATOMIC FISSION LOGIC
# ==========================================
def perform_atomic_fission(word, src_mat, tgt_mat, w2i_src, src_anchors, tgt_anchors, ref_model, k=2):
    """
    Splits a word into 'k' atoms based on Source Neighborhood, 
    maps atoms to Target, and identifies them.
    """
    if word not in w2i_src: return
    
    vec = src_mat[w2i_src[word]]
    
    # A. Get Local Neighborhood in Source
    # Get top 50 neighbors to form a context cloud
    sims = np.dot(src_mat, vec)
    context_idxs = np.argsort(sims)[-50:] 
    context_vecs = src_mat[context_idxs]
    
    # B. Cluster the Context (Fission)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(context_vecs)
    
    print(f"\n🔬 FISSION: '{word.upper()}' (Split into {k} Atoms)")
    
    # C. Process Each Atom
    for i in range(k):
        # 1. Define Atom Centroid in Source
        atom_mask = labels == i
        if np.sum(atom_mask) < 2: continue # Skip noise
        
        atom_src_vec = np.mean(context_vecs[atom_mask], axis=0)
        atom_src_vec /= np.linalg.norm(atom_src_vec)
        
        # 2. Transport Atom to Universe (Target)
        atom_tgt_vec = get_lattice_projection(src_anchors, tgt_anchors, atom_src_vec)
        
        # 3. Identify Atom in Universe
        # Find closest word in Target Reference Model
        try:
            matches = ref_model.similar_by_vector(atom_tgt_vec, topn=3)
            match_str = ", ".join([f"{m[0]} ({m[1]:.2f})" for m in matches])
            print(f"   ATOM {i+1}: Maps to -> [{match_str}]")
        except Exception as e:
            print(f"   ATOM {i+1}: Alignment Failed ({e})")

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    # A. Load Matrices
    ref_words, ref_mat, w2i_ref = load_and_norm(REF_MODEL_NAME, VOCAB_LIMIT)
    tgt_words, tgt_mat, w2i_tgt = load_and_norm(TGT_MODEL_NAME, VOCAB_LIMIT) # Src = GloVe

    if ref_mat is None or tgt_mat is None:
        print("Failed to load matrices.")
        exit(1)

    # B. Anchors (The Grid)
    common_set = set(ref_words).intersection(set(tgt_words))
    common_list = sorted(list(common_set), key=lambda w: w2i_ref[w])
    anchors = common_list[:NUM_ANCHORS]

    X_src_anchors = np.array([tgt_mat[w2i_tgt[w]] for w in anchors])
    X_tgt_anchors = np.array([ref_mat[w2i_ref[w]] for w in anchors])

    # C. Reload Full Reference Model for Lookup
    print("Loading full Reference Model for lookup...")
    ref_model_raw = None
    manual_base = os.path.expanduser(f"~/gensim-data/{REF_MODEL_NAME}/{REF_MODEL_NAME}.gz")
    if os.path.exists(manual_base):
        try:
            print(f"Loading raw model from {manual_base}")
            ref_model_raw = KeyedVectors.load_word2vec_format(manual_base, binary=True)
        except Exception as e:
            print(f"Failed to load raw binary: {e}")

    if ref_model_raw is None:
        try:
             ref_model_raw = api.load(REF_MODEL_NAME)
        except Exception as e:
            print(f"Failed to load via api: {e}")
            exit(1)

    print("\n" + "="*50)
    print("EXPERIMENT 5: ATOMIC FISSION (POLYSEMY REPAIR)")
    print("="*50)
    print("Attempting to split 'Torn' concepts into Universal Atoms.")

    # D. Test Cases (High Stress Words + Polysemes)
    test_cases = [
        ("bank", 2),   # Financial vs River
        ("apple", 2),  # Fruit vs Tech
        ("run", 2),    # Motion vs Business/Abstract
        ("cell", 2),   # Biology vs Prison/Phone
        ("star", 2),   # Astronomy vs Celebrity
        ("degree", 2), # Temperature vs Education
        ("charge", 3)  # Electrical vs Criminal vs Cost
    ]

    for w, k in test_cases:
        if w in w2i_tgt:
            perform_atomic_fission(w, tgt_mat, ref_mat, w2i_tgt, 
                                   X_src_anchors, X_tgt_anchors, 
                                   ref_model_raw, k=k)

    print("\n" + "="*50)
    print("ANALYSIS")
    print("If the atoms map to distinct, coherent concepts in the Universe,")
    print("then 'Semantic Tearing' is solved by 'Sheaf Expansion'.")
