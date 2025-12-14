import numpy as np
import matplotlib.pyplot as plt
from universal_sphere_v2 import UniversalSemanticSphereV2
from sklearn.decomposition import PCA

# ==========================================
# 1. WITNESS EXTRACTOR
# ==========================================
class WitnessExtractor:
    def __init__(self, model):
        self.model = model

    def get_witnesses(self, word, context=None, top_k=50):
        """
        Extracts witness vectors.
        If context is provided, uses vector addition (word + context) to bias the neighborhood.
        This simulates 'Contextual Shrinkage'.
        """
        if word not in self.model:
            return [], []

        if context:
            # Simple additive context (standard w2v arithmetic)
            # vector = word + context
            target_vec = self.model[word] + self.model[context]
        else:
            target_vec = self.model[word]

        # Get nearest neighbors as witnesses
        # We need both words and their vectors
        neighbors = self.model.similar_by_vector(target_vec, topn=top_k)
        
        witness_words = [n[0] for n in neighbors]
        # Robustly handle read-only arrays
        witness_vecs = np.array([self.model[w].copy() for w in witness_words])
        
        # Normalize
        norms = np.linalg.norm(witness_vecs, axis=1, keepdims=True)
        witness_vecs = witness_vecs / norms
        
        return witness_words, witness_vecs

# ==========================================
# 2. GEOMETRIC MEANING (Volume)
# ==========================================
def compute_semantic_volume(projected_witnesses):
    """
    Computes Generalized Variance (Determinant of Covariance) 
    as a proxy for the 'Volume' of the admissible region.
    """
    if len(projected_witnesses) < 2:
        return 0.0
        
    # Covariance Matrix (K x K)
    cov_matrix = np.cov(projected_witnesses.T)
    # Pseudo-determinant (product of non-zero eigenvalues) or Trace (Total Variance)
    # Determinant is better for "Volume", Trace is better for "Spread".
    # Since dimensionality (24) might be high relative to witnesses (50),
    # determinant approaches zero. We'll use TRACE (Total Variance) as a robust metric for Ambiguity.
    
    total_variance = np.trace(cov_matrix)
    return total_variance

# ==========================================
# 3. EXPERIMENT: CONTEXT SHRINKAGE
# ==========================================
def run_context_experiment():
    # A. Init Sphere & Model
    sphere = UniversalSemanticSphereV2()
    sphere.build_basis() # Load w2v and build axes
    
    extractor = WitnessExtractor(sphere.seed_model)
    
    # B. Test Case: "BANK"
    target = "bank"
    contexts = [None, "money", "river"]
    
    results = {}
    
    print(f"\nScanning Semantic Volume for '{target.upper()}'...")
    
    for ctx in contexts:
        label = f"{target} + {ctx}" if ctx else target
        print(f"\n--- Context: {label} ---")
        
        # 1. Extract
        words, vecs = extractor.get_witnesses(target, context=ctx, top_k=50)
        print(f"Witnesses: {words[:7]}...")
        
        # 2. Project to Universal Sphere (24-dim)
        projected = sphere.project(vecs.T).T # (N, K)
        
        # 3. Compute Volume
        volume = compute_semantic_volume(projected)
        results[label] = {
            "vol": volume,
            "centroid": np.mean(projected, axis=0),
            "words": words,
            "proj": projected
        }
        print(f"Semantic Volume (Trace): {volume:.4f}")

    # C. Verification
    base_vol = results[target]["vol"]
    fin_vol = results[f"{target} + money"]["vol"]
    geo_vol = results[f"{target} + river"]["vol"]
    
    print("\n" + "="*40)
    print("RESULTS: CONTEXTUAL HULL SHRINKAGE")
    print("="*40)
    print(f"Base Volume ('{target}'): {base_vol:.4f}")
    print(f"Financial Volume (+money): {fin_vol:.4f} ({(fin_vol/base_vol)*100:.1f}%)")
    print(f"Geographic Volume (+river): {geo_vol:.4f} ({(geo_vol/base_vol)*100:.1f}%)")
    
    # Distance between centroids
    c_fin = results[f"{target} + money"]["centroid"]
    c_geo = results[f"{target} + river"]["centroid"]
    dist = np.linalg.norm(c_fin - c_geo)
    print(f"Distance between Context Centroids: {dist:.4f}")
    
    if fin_vol < base_vol and geo_vol < base_vol:
        print("\n✅ STATUS: SUCCESS.")
        print("   Adding context reduces Semantic Entropy (Volume).")
        print("   Ambiguity shrinks via Witness Accumulation.")
    else:
        print("\n⚠️ STATUS: FAILURE.")
        print("   Context did not strictly reduce volume.")

    # D. Visualization (PCA to 2D)
    # Combine all points to fit PCA
    all_proj = np.vstack([r["proj"] for r in results.values()])
    pca = PCA(n_components=2)
    pca.fit(all_proj)
    
    plt.figure(figsize=(10, 8))
    colors = {'bank': 'gray', 'bank + money': 'green', 'bank + river': 'blue'}
    
    for label, res in results.items():
        coords = pca.transform(res["proj"])
        plt.scatter(coords[:,0], coords[:,1], label=f"{label} (Vol={res['vol']:.2f})", 
                    c=colors.get(label, 'red'), alpha=0.6, edgecolors='none')
        
        # Annotate centroid
        cent = np.mean(coords, axis=0)
        plt.scatter(cent[0], cent[1], c="black", marker="x", s=100)
    
    plt.title(f"Dynamic Meaning of '{target.upper()}' in Universal Sphere (PCA proj)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("context_shrinkage.png")
    print("Plot saved to 'context_shrinkage.png'")

if __name__ == "__main__":
    run_context_experiment()
