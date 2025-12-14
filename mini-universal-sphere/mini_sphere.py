import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from scipy.linalg import orth
from sklearn.decomposition import PCA
from typing import List, Tuple
import ssl

# Bypass SSL verification for gensim downloader
ssl._create_default_https_context = ssl._create_unverified_context


# ==========================================
# PART 1: THE UNIVERSAL SPHERE KERNEL
# ==========================================

class UniversalSemanticSphere:
    def __init__(self, raw_model, target_dim=100):
        """
        Initializes the Universal Sphere.
        raw_model: A standard embedding model (Word2Vec/GloVe) acting as 'Raw Perception'
        target_dim: The fixed dimension 'd' of our universe.
        """
        self.raw = raw_model
        self.d = target_dim
        self.basis_matrix = None # The transformation Q
        self.vocab = list(raw_model.key_to_index.keys())[:20000] # Limit vocab for speed
        
        # We start with the raw matrix of the top 20k words
        self.X_raw = np.array([raw_model[w] for w in self.vocab])
        
        # Normalize raw inputs (Axiom 0.2: Norms don't matter yet, but helps stability)
        self.X_raw = self.X_raw / np.linalg.norm(self.X_raw, axis=1, keepdims=True)

    def build_universal_basis(self, anchor_definitions: List[Tuple[str, str]]):
        """
        Constructs the 'Semantic Axes' (Basis B).
        Instead of random init, we lock specific dimensions to human universals.
        Algorithm: Constrained Symbolic Gram-Schmidt + Spectral Residuals.
        """
        print(f"Constructing Universe with {len(anchor_definitions)} Symbolic Axes...")
        
        symbolic_vectors = []
        
        # 1. Extract Symbolic Directions
        for (pos_word, neg_word) in anchor_definitions:
            try:
                # Direction = v(positive) - v(negative)
                v = self.raw[pos_word] - self.raw[neg_word]
                v = v / np.linalg.norm(v)
                symbolic_vectors.append(v)
            except KeyError:
                print(f"Warning: Seeds '{pos_word}/{neg_word}' not found.")
        
        symbolic_matrix = np.array(symbolic_vectors).T 
        
        # 2. Orthonormalize the Symbolic Axes (Gram-Schmidt)
        # This ensures 'Time' and 'Valence' are mathematically independent axes
        # WE USE MANUAL GRAM-SCHMIDT to preserve the order (Axis 0 = Valence, etc.)
        # scipy.linalg.orth() uses SVD and does not preserve column order/meaning.
        
        def stable_gram_schmidt(V):
            # V is (D, N) where N is number of axes
            basis = []
            for i in range(V.shape[1]):
                v = V[:, i]
                # Subtract projections onto existing basis vectors
                for b in basis:
                    v = v - np.dot(v, b) * b
                
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v = v / norm
                    basis.append(v)
                else:
                    # If dependent, we essentially skip or can add random ortho (but here we skip)
                    pass
            return np.array(basis).T

        Q_sym = stable_gram_schmidt(symbolic_matrix)
        
        num_symbolic = Q_sym.shape[1]

        print(f" -> Stabilized {num_symbolic} orthogonal symbolic axes.")

        # 3. Spectral Initialization for Latent Dimensions
        # We project the raw data into the Null Space of the symbolic axes
        # to find variance NOT explained by our symbols.
        
        # Projector onto Symbolic Space
        P_sym = Q_sym @ Q_sym.T
        # Projector onto Null Space (I - P)
        P_null = np.eye(self.raw.vector_size) - P_sym
        
        # Project data into null space
        X_resid = self.X_raw @ P_null
        
        # Run PCA on residuals to find the 'Latent Statistical Axes'
        pca = PCA(n_components = self.d - num_symbolic)
        pca.fit(X_resid)
        Q_lat = pca.components_.T # These are the remaining basis vectors
        
        # 4. Fuse the Basis
        # The Universal Basis Q is [Symbolic | Latent]
        self.basis_matrix = np.hstack([Q_sym, Q_lat])
        
        print(f" -> Universe Construction Complete. Basis Shape: {self.basis_matrix.shape}")

    def embed(self, word_or_vec):
        """
        Maps a token or vector into the Universal Sphere.
        No retraining. Just projection onto the fixed basis B.
        """
        if isinstance(word_or_vec, str):
            if word_or_vec not in self.raw:
                return None
            v_raw = self.raw[word_or_vec]
        else:
            v_raw = word_or_vec
            
        # 1. Linear Projection (Transform to Universal Coordinates)
        v_univ = v_raw @ self.basis_matrix
        
        # 2. Sphericalization (Axiom 0.2)
        v_sphere = v_univ / np.linalg.norm(v_univ)
        
        return v_sphere

# ==========================================
# PART 2: REWA GEOMETRY LOGIC
# ==========================================

def get_hemisphere_center(witnesses):
    """
    Finds a vector 'c' such that w_i . c > 0 for all witnesses.
    If this exists, the Hemisphere Constraint is satisfied.
    Simple heuristic: The normalized sum (Frechet mean).
    """
    mean_vec = np.sum(witnesses, axis=0)
    if np.linalg.norm(mean_vec) < 1e-9:
        return None # Vectors perfectly cancel out (antipodal)
    return mean_vec / np.linalg.norm(mean_vec)

def check_admissibility(witnesses):
    """
    Returns True if witnesses define a valid semantic region.
    Returns False (Refusal) if they are contradictory (violate hemisphere).
    """
    if len(witnesses) == 0: return False
    
    # 1. Calculate Candidate Mean
    mu = get_hemisphere_center(witnesses)
    if mu is None: return False
    
    # 2. Verify all witnesses are in the hemisphere defined by mu
    # (Axiom 4.2 / Theorem 4.3 Impossibility Detection)
    dots = witnesses @ mu
    min_dot = np.min(dots)
    
    # If min_dot <= 0, one witness is >= 90 degrees away from the mean
    # strictly speaking, convexity allows spread up to <180, 
    # but robust coherence usually demands <90 (Strict Policy) or <180 (Relaxed)
    # Here we implement the strict hemisphere constraint (>0).
    return min_dot > 0, mu

# ==========================================
# PART 3: EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Load Raw Perception (Simulating a pre-trained model)
    print("Downloading GloVe (Raw Perception)... this may take a minute.")
    try:
        glove = api.load("glove-wiki-gigaword-50") # Small 50d model for demo
    except Exception as e:
        print(f"Warning: gensim api.load failed ({e}). Falling back to manual load.")
        from gensim.models import KeyedVectors
        import os
        # Path where we manually downloaded it
        manual_path = os.path.expanduser("~/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
        if os.path.exists(manual_path):
            print(f"Loading from local file: {manual_path}")
            glove = KeyedVectors.load_word2vec_format(manual_path, binary=False)
        else:
            raise RuntimeError(f"Could not load model via api.load and manual file not found at {manual_path}")


    # 2. Initialize the Immutable Sphere
    universe = UniversalSemanticSphere(glove, target_dim=50)

    # 3. Define the Fundamental Axes of Humanity (The "Big Bang")
    # These definitions will NEVER change.
    anchors = [
        ("good", "bad"),       # Axis 0: Valence
        ("real", "imaginary"), # Axis 1: Existence
        ("future", "past"),    # Axis 2: Time
        ("active", "passive"), # Axis 3: Agency
        ("known", "unknown")   # Axis 4: Epistemic
    ]

    universe.build_universal_basis(anchors)

    # ==========================================
    # DEMO 1: INTERPRETABILITY (Zero-Shot)
    # ==========================================
    print("\n--- DEMO 1: AXIS INSPECTION ---")
    print("Because we locked the geometry, Axis 0 IS 'Valence' and Axis 2 IS 'Time'.")
    print("We can check coordinates directly without a classifier.")

    test_words = ["excellent", "murder", "tomorrow", "yesterday", "fantasy"]

    for w in test_words:
        v = universe.embed(w)
        if v is None:
            print(f"Word '{w}' not in vocabulary.")
            continue
        # Coordinate 0 is Good/Bad, Coordinate 2 is Future/Past
        valence = v[0] 
        time_dim = v[2]
        
        val_str = "POSITIVE" if valence > 0 else "NEGATIVE"
        time_str = "FUTURE" if time_dim > 0 else "PAST"
        
        print(f"Word: '{w:<12}' | Valence: {valence:+.3f} ({val_str}) | Time: {time_dim:+.3f} ({time_str})")


    # ==========================================
    # DEMO 2: ADMISSIBLE REGIONS & REFUSAL
    # ==========================================
    print("\n--- DEMO 2: REWA ADMISSIBILITY CHECKS ---")

    def run_rewa_check(concept_name, evidence_words):
        print(f"\nQuery: Interpret '{concept_name}'")
        print(f"Evidence (Witnesses): {evidence_words}")
        
        # Map witnesses to the sphere
        valid_words = [w for w in evidence_words if w in universe.raw]
        if len(valid_words) != len(evidence_words):
             print(f"Warning: Some words skipped due to OOV: {set(evidence_words) - set(valid_words)}")

        if not valid_words:
            print("No valid witnesses found.")
            return

        w_vecs = np.array([universe.embed(w) for w in valid_words])
        
        # Check Geometry
        is_valid, mean_direction = check_admissibility(w_vecs)
        
        if is_valid:
            print(f"✅ STATUS: ADMISSIBLE.")
            print(f"   The witnesses form a coherent convex hull.")
            # In a real app, we would now apply Policy rho(mean_direction)
        else:
            print(f"⛔ STATUS: REFUSAL (Topological Contradiction).")
            print(f"   Witnesses violate the Hemisphere Constraint.")

    # Case A: Coherent Concept
    run_rewa_check("Bank (Financial)", ["money", "deposit", "finance", "loan"])

    # Case B: Coherent Concept (Ambiguity, but valid)
    run_rewa_check("Bank (River)", ["river", "water", "flow", "fish"])

    # Case C: Impossible Concept (Hallucination Attempt)
    # User asks: "Imagine a hot freezing fire"
    run_rewa_check("Hot Ice", ["fire", "magma", "ice", "freezing"])

    # ==========================================
    # DEMO 3: VISUALIZATION
    # ==========================================
    print("\n--- DEMO 3: VISUALIZING THE MANIFOLD ---")
    # Let's project some words onto our first two Symbolic Axes
    words_to_plot = ["love", "hate", "happy", "sad", "fact", "myth", "science", "ghost"]
    coords = []
    labels = []

    for w in words_to_plot:
        v = universe.embed(w)
        if v is not None:
            coords.append([v[0], v[1]]) # Axis 0 (Valence), Axis 1 (Reality)
            labels.append(w)

    coords = np.array(coords)

    if len(coords) > 0:
        plt.figure(figsize=(8, 8))
        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(0, color='grey', linestyle='--')
        plt.scatter(coords[:,0], coords[:,1], c='blue')

        for i, txt in enumerate(labels):
            plt.annotate(txt, (coords[i,0]+0.02, coords[i,1]+0.02), fontsize=12)

        plt.xlabel("Axis 0: Valence (Bad <---> Good)")
        plt.ylabel("Axis 1: Existence (Imaginary <---> Real)")
        plt.title("The Immutable Universal Semantic Sphere (Axes 0 & 1)")
        plt.grid(True, alpha=0.3)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        # Draw the unit circle (boundary of the 2D slice of the sphere)
        circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle=':')
        plt.gca().add_patch(circle)
        
        print("Saving plot to 'manifold_visualization.png'...")
        plt.savefig("manifold_visualization.png")
        # plt.show() # Commented out to avoid blocking execution in headless env
        print("Plot saved.")
