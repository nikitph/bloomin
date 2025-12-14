import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from scipy.linalg import orthogonal_procrustes
from mini_sphere import UniversalSemanticSphere
import os
from gensim.models import KeyedVectors
import ssl

# Bypass SSL verification for gensim downloader just in case
ssl._create_default_https_context = ssl._create_unverified_context

def load_word2vec_google_news():
    """
    Robust loader for the large Google News model.
    Checks local manual path first, then tries gensim API.
    """
    print("Loading Word2Vec (Google News 300d)... this is ~1.5GB and may take time.")
    
    # Check common manual paths
    manual_paths = [
        os.path.expanduser("~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"),
        "/Volumes/External/models/GoogleNews-vectors-negative300.bin.gz",
        "./GoogleNews-vectors-negative300.bin.gz"
    ]
    
    for path in manual_paths:
        if os.path.exists(path):
            print(f"Found local file at: {path}")
            return KeyedVectors.load_word2vec_format(path, binary=True)
            
    print("Local file not found. Attempting gensim download (this is huge)...")
    try:
        return api.load("word2vec-google-news-300")
    except Exception as e:
        print(f"Gensim download failed: {e}")
        print("Please manually download 'word2vec-google-news-300.gz' to ~/gensim-data/word2vec-google-news-300/")
        raise

def load_glove_local_or_api():
    """ Reuse logic from mini_sphere.py for GloVe """
    print("Loading GloVe (50d)...")
    try:
        return api.load("glove-wiki-gigaword-50")
    except Exception:
        manual_path = os.path.expanduser("~/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
        if os.path.exists(manual_path):
             return KeyedVectors.load_word2vec_format(manual_path, binary=False)
        raise

def align_spheres(sphere_target, sphere_source, common_vocab):
    """
    Aligns source sphere to target sphere using Procrustes on common vocab.
    Returns: aligned_source_vectors, residual
    """
    # Extract vectors for common vocabulary
    X_target = []
    X_source = []
    
    for w in common_vocab:
        v_t = sphere_target.embed(w)
        v_s = sphere_source.embed(w)
        if v_t is not None and v_s is not None:
             X_target.append(v_t)
             X_source.append(v_s)
             
    X_target = np.array(X_target)
    X_source = np.array(X_source)
    
    # Orthogonal Procrustes: Find R such that ||X_target - X_source @ R|| is minimized
    # Note: scipy's orthogonal_procrustes solves for R s.t. ||A - B @ R|| is min
    # Here A = X_target, B = X_source
    R, scale = orthogonal_procrustes(X_target, X_source)
    
    # Align source
    X_source_aligned = X_source @ R
    
    # Calculate Residual (Frobenius Norm / N)
    diff = X_target - X_source_aligned
    residual = np.linalg.norm(diff, 'fro') / np.sqrt(len(common_vocab))
    
    return X_source_aligned, X_target, residual, R

if __name__ == "__main__":
    # 1. Load Models
    glove_model = load_glove_local_or_api()
    try:
        w2v_model = load_word2vec_google_news()
    except Exception as e:
        print(f"CRITICAL: Could not load Word2Vec. {e}")
        exit(1)

    # 2. Build Independent Universes
    print("\n--- BUILDING SPHERES ---")
    anchors = [
        ("good", "bad"),       # Axis 0: Valence
        ("real", "imaginary"), # Axis 1: Existence
        ("future", "past"),    # Axis 2: Time
        ("active", "passive"), # Axis 3: Agency
        ("known", "unknown")   # Axis 4: Epistemic
    ]
    
    # Sphere A: GloVe 50d
    print("Constructing GloVe Sphere...")
    univ_glove = UniversalSemanticSphere(glove_model, target_dim=50)
    univ_glove.build_universal_basis(anchors)
    
    # Sphere B: Word2Vec 50d (Downsizing 300d -> 50d via the Universal Projector)
    print("Constructing Word2Vec Sphere...")
    univ_w2v = UniversalSemanticSphere(w2v_model, target_dim=50) # Projecting 300 -> 50
    univ_w2v.build_universal_basis(anchors)
    
    # 3. Convergence Test
    print("\n--- RUNNING CONVERGENCE TEST ---")
    # Find common vocabulary (top 1000 overlap)
    vocab_glove = set(list(glove_model.key_to_index.keys())[:5000])
    vocab_w2v = set(list(w2v_model.key_to_index.keys())[:5000])
    common_vocab = list(vocab_glove.intersection(vocab_w2v))
    print(f"Found {len(common_vocab)} common words in top 5000.")
    
    aligned_w2v, target_glove, residual, R_matrix = align_spheres(univ_glove, univ_w2v, common_vocab)
    
    print(f"\n✅ CONVERGENCE RESIDUAL: {residual:.4f}")
    if residual < 0.2:
        print("RESULT: STRONG CONVERGENCE. The spheres are geometrically isomorphic!")
    elif residual < 0.4:
        print("RESULT: MODERATE CONVERGENCE. Similar topology.")
    else:
        print("RESULT: DIVERGENCE. The corpora define different realities.")
        
    # 4. Visualization
    print("\n--- PLOTTING ---")
    plot_words = ["love", "hate", "freedom", "prison", "truth", "lie", "flower", "gun", "science", "magic"]
    
    coords_g = []
    coords_w = []
    labels = []
    
    for w in plot_words:
        if w in glove_model and w in w2v_model:
            v_g = univ_glove.embed(w)
            v_w = univ_w2v.embed(w)
            
            # Align individual vector w manually using R
            v_w_aligned = v_w @ R_matrix
            
            coords_g.append(v_g[:2]) # Valence vs Reality
            coords_w.append(v_w_aligned[:2])
            labels.append(w)
            
    coords_g = np.array(coords_g)
    coords_w = np.array(coords_w)
    
    plt.figure(figsize=(10, 10))
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    
    # Plot GloVe (Blue)
    plt.scatter(coords_g[:,0], coords_g[:,1], c='blue', label='GloVe Sphere', marker='o', s=100, alpha=0.6)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (coords_g[i,0], coords_g[i,1]), color='blue', fontsize=10, xytext=(5, 5), textcoords='offset points')
        
    # Plot Word2Vec (Red)
    plt.scatter(coords_w[:,0], coords_w[:,1], c='red', label='Word2Vec Sphere (Aligned)', marker='x', s=100, alpha=0.6)
    for i, txt in enumerate(labels):
        # Draw line connecting them
        plt.plot([coords_g[i,0], coords_w[i,0]], [coords_g[i,1], coords_w[i,1]], 'k-', alpha=0.2)

    plt.title(f"Universal Convergence: GloVe vs Word2Vec\nResidual: {residual:.4f}")
    plt.xlabel("Axis 0: Valence (Bad <-> Good)")
    plt.ylabel("Axis 1: Existence (Imaginary <-> Real)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("convergence_plot.png")
    print("Plot saved to 'convergence_plot.png'")
