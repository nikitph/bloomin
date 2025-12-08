import numpy as np
import matplotlib.pyplot as plt
from model_loaders import GensimLoader
from curvature_measurement import measure_curvature
import os

def measure_euclidean_stats(embeddings, n_triangles=1000):
    excesses = []
    
    for _ in range(n_triangles):
        idx = np.random.choice(len(embeddings), 3, replace=False)
        p1, p2, p3 = embeddings[idx]
        
        # Law of Cosines angles in Euclidean space (Flat)
        def get_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos, -1.0, 1.0))
            
        A = get_angle(p2-p1, p3-p1)
        B = get_angle(p1-p2, p3-p2) 
        C = get_angle(p1-p3, p2-p3) 
        
        excess = (A + B + C) - np.pi
        excesses.append(excess)
        
    mean_excess = np.mean(excesses)
    return 0.0 if abs(mean_excess) < 1e-4 else mean_excess

def run_glove_experiment():
    print("Loading GloVe (glove-wiki-gigaword-100)...")
    loader = GensimLoader()
    try:
        model = loader.load_model('glove-wiki-gigaword-100')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get sample
    try:
        keys = list(model.key_to_index.keys())[:10000]
    except:
        keys = list(model.vocab.keys())[:10000]
        
    embeddings_raw = loader.get_embeddings(keys)
    
    # 1. Norm Statistics
    norms = np.linalg.norm(embeddings_raw, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    cv = std_norm / mean_norm
    
    print(f"\n1. Norm Statistics (n={len(embeddings_raw)}):")
    print(f"   Mean: {mean_norm:.4f}")
    print(f"   Std:  {std_norm:.4f}")
    print(f"   CV:   {cv:.4f} (Coefficient of Variation)")
    print(f"   Max:  {np.max(norms):.4f}")
    
    # 2. Raw Curvature
    print(f"\n2. Raw Experiment (Euclidean Geometry)")
    K_raw = measure_euclidean_stats(embeddings_raw)
    print(f"   Raw GloVe: K ≈ {K_raw:.4f} (Flat Volume)")
    
    # 3. Normalized Curvature
    print(f"\n3. Normalized Experiment (Spherical Geometry)")
    results = measure_curvature(embeddings_raw, n_triangles=1000, verbose=False)
    K_norm = results['K_mean']
    print(f"   Normalized GloVe: K ≈ {K_norm:.4f} (Curved Surface)")

    # Plot Norms
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(mean_norm, color='red', linestyle='--', label=f'Mean: {mean_norm:.2f}')
    plt.title('Distribution of GloVe Vector Norms')
    plt.xlabel('Euclidean Norm ||v||')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'results/glove_norm_dist.png'
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_glove_experiment()
