import numpy as np
from model_loaders import GensimLoader
from curvature_measurement import measure_curvature

def measure_euclidean_stats(embeddings, n_triangles=1000):
    """
    Measure 'Curvature' treating vectors as points in Euclidean space R^n.
    For any triangle in R^n, Angle Sum = Pi, so Excess = 0, K = 0.
    This serves as a control to show Raw vectors form flat triangles.
    """
    print(f"Sampling {n_triangles} triangles in Euclidean space...")
    
    excesses = []
    
    for _ in range(n_triangles):
        idx = np.random.choice(len(embeddings), 3, replace=False)
        p1, p2, p3 = embeddings[idx]
        
        # Edge vectors
        u = p2 - p1
        v = p3 - p2
        w = p1 - p3
        
        # Norms
        a = np.linalg.norm(u)
        b = np.linalg.norm(v)
        c = np.linalg.norm(w)
        
        # Law of Cosines for angles
        # A (at p1): angle betw (p2-p1) and (p3-p1)
        # vec_A1 = p2-p1 (u)
        # vec_A2 = p3-p1 (-w)
        
        def get_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos, -1.0, 1.0))
            
        A = get_angle(p2-p1, p3-p1)
        B = get_angle(p1-p2, p3-p2) # p1-p2 is -u, p3-p2 is v
        C = get_angle(p1-p3, p2-p3) 
        
        angle_sum = A + B + C
        excess = angle_sum - np.pi
        excesses.append(excess)
        
    mean_excess = np.mean(excesses)
    print(f"Euclidean Excess: {mean_excess:.6f} (Expected: 0.0)")
    
    # Heuristically, K = Excess / Area. If Excess ~ 0, K ~ 0.
    return 0.0 if abs(mean_excess) < 1e-4 else mean_excess

def run_experiment():
    print("Loading Word2Vec...")
    loader = GensimLoader()
    model = loader.load_model('word2vec-google-news-300')
    
    # Get subset
    try:
        keys = list(model.key_to_index.keys())[:10000]
    except:
        keys = list(model.vocab.keys())[:10000]
        
    embeddings_raw = loader.get_embeddings(keys)
    
    print(f"\nExperiment 1: Raw Word2Vec (Euclidean Geometry)")
    K_raw = measure_euclidean_stats(embeddings_raw)
    print(f"Raw Word2Vec: K ≈ {K_raw:.4f} (Flat Volume)")
    
    print(f"\nExperiment 2: Normalized Word2Vec (Spherical Geometry)")
    # Note: measure_curvature auto-normalizes internally, so we can pass raw or normalized.
    # We pass raw and let it project to the sphere to measure the SURFACE curvature.
    results = measure_curvature(embeddings_raw, n_triangles=1000, verbose=False)
    K_norm = results['K_mean']
    print(f"Normalized Word2Vec: K ≈ {K_norm:.4f} (Curved Surface)")
    
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print(f"Raw Vectors (Volume):    K ≈ {K_raw:.2f}")
    print(f"Norm Vectors (Surface):  K ≈ {K_norm:.2f}")
    print("Normalization PROJOECTS the Eucliean cloud onto the Semantic Hypersphere.")
    print("The 'Meaning' is in the Surface (Angles), not the Volume (Lengths).")

if __name__ == "__main__":
    run_experiment()
