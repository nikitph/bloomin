import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Attempt import or mock
try:
    from sentence_transformers import SentenceTransformer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False
    print("Warning: sentence_transformers not found. Using synthetic spherical data.")

import urllib.request

def get_embeddings():
    if HAS_BERT:
        try:
            print("Downloading 10,000 word list...")
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            
            articles = text.splitlines()
            # articles = articles[:10000] # Use all (usually ~10k)
            print(f"Encoding {len(articles)} topics...")
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(articles, batch_size=64, show_progress_bar=True)
            
            return articles, embeddings
        except Exception as e:
            print(f"Error loading model or data: {e}")
            print("Falling back to synthetic data.")
            return create_synthetic_data()
    else:
        return create_synthetic_data()

def create_synthetic_data():
    # Synthetic data on a high-dim sphere
    n_samples = 10000
    dim = 384
    # Random Gaussian
    data = np.random.randn(n_samples, dim)
    # Normalize to sphere
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    embeddings = data / norms
    articles = [f"Topic_{i}" for i in range(n_samples)]
    return articles, embeddings

def estimate_intrinsic_dimension(embeddings):
    # PCA
    pca = PCA().fit(embeddings)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    d_95 = np.searchsorted(cumsum, 0.95) + 1
    d_99 = np.searchsorted(cumsum, 0.99) + 1
    
    return d_95, d_99

def measure_curvature(u, v, w):
    # 1. Edge lengths (geodesic dist on sphere = angle)
    # Ensure normalized
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)
    
    # Dot products
    uv = np.clip(np.dot(u, v), -1.0, 1.0)
    vw = np.clip(np.dot(v, w), -1.0, 1.0)
    wu = np.clip(np.dot(w, u), -1.0, 1.0)
    
    a = np.arccos(vw) # opposite u
    b = np.arccos(wu) # opposite v
    c = np.arccos(uv) # opposite w
    
    # 2. Angles using Spherical Law of Cosines
    # cos a = cos b cos c + sin b sin c cos alpha
    # cos alpha = (cos a - cos b cos c) / (sin b sin c)
    epsilon = 1e-9
    
    def get_angle(side, adj1, adj2):
        denom = np.sin(adj1) * np.sin(adj2)
        if hasattr(denom, 'item'): denom = denom.item()
        if abs(denom) < epsilon: return 0.0 # Degenerate
        
        cos_ang = (np.cos(side) - np.cos(adj1)*np.cos(adj2)) / denom
        return np.arccos(np.clip(cos_ang, -1.0, 1.0))
        
    alpha = get_angle(a, b, c)
    beta = get_angle(b, a, c)
    gamma = get_angle(c, a, b)
    
    # 3. Excess
    angle_sum = alpha + beta + gamma
    excess = angle_sum - np.pi
    
    # 4. Planar Area (Heron's Formula on side lengths)
    # This represents the area of a flat triangle with same side lengths
    # Used as reference to compute K
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq < 0: area_sq = 0
    area = np.sqrt(area_sq)
    
    # 5. K = Excess / Area
    if area < epsilon:
        K = 0.0 # Undefined/Flat
    else:
        K = excess / area
        
    return K, excess, area

def run_experiments():
    print("Loading data...")
    articles, embeddings = get_embeddings()
    print(f"Loaded {len(articles)} articles. Embedding shape: {embeddings.shape}")
    
    # --- Experiment 1: Dimension ---
    print("\n--- Experiment 1: Intrinsic Dimension ---")
    d_95, d_99 = estimate_intrinsic_dimension(embeddings)
    print(f"Wikipedia intrinsic dimension:")
    print(f"  95% variance: {d_95}D")
    print(f"  99% variance: {d_99}D")
    
    # --- Experiment 2: Curvature ---
    print("\n--- Experiment 2: Curvature ---")
    results = []
    
    for _ in range(100):
        idx = np.random.choice(len(articles), 3, replace=False)
        K, excess, area = measure_curvature(
            embeddings[idx[0]],
            embeddings[idx[1]],
            embeddings[idx[2]]
        )
        if area > 1e-6: # Filter degenerate
            results.append(K)
            
    K_mean = np.mean(results)
    K_std = np.std(results)
    
    print(f"Global Curvature:")
    print(f"  Mean K: {K_mean:.4f} ± {K_std:.4f}")
    
    if abs(K_mean - 1.0) < 0.1:
        print("  ✓ Consistent with K=1 (spherical)")
    elif K_mean > 0:
        print(f"  → Positive curvature K={K_mean:.2f} (elliptic)")
    else:
        print(f"  → Negative curvature K={K_mean:.2f} (hyperbolic)")
        
    # --- Experiment 3: Local vs Global ---
    print("\n--- Experiment 3: Local Curvature ---")
    kmeans = KMeans(n_clusters=5, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    for cluster_id in range(5):
        mask = labels == cluster_id
        cluster_emb = embeddings[mask]
        
        if len(cluster_emb) < 5: 
            continue
            
        K_local = []
        for _ in range(20):
            idx = np.random.choice(len(cluster_emb), 3, replace=False)
            K, _, area = measure_curvature(
                cluster_emb[idx[0]],
                cluster_emb[idx[1]],
                cluster_emb[idx[2]]
            )
            if area > 1e-6:
                K_local.append(K)
                
        if K_local:
            print(f"Cluster {cluster_id} (n={len(cluster_emb)}): K = {np.mean(K_local):.4f}")

if __name__ == "__main__":
    run_experiments()
