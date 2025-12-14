import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
import os
import ssl

# Bypass SSL
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. GROUNDED AXES DEFINITION
# ==========================================
# Ordered by primacy for Gram-Schmidt stability.
# Format: 'Axis Name': ('Negative Pole', 'Positive Pole')
GROUNDED_AXES = {
    # --- FUNDAMENTAL DIMENSIONS (Space/Time/State) ---
    'existence':    ('mythical', 'real'),  # Reality
    'time':         ('past', 'future'),    # Temporal Direction
    'valence':      ('bad', 'good'),       # Affect
    'spatial_x':    ('left', 'right'),     # Horizontal
    'spatial_y':    ('down', 'up'),        # Vertical
    'spatial_z':    ('far', 'near'),       # Depth
    'animacy':      ('inanimate', 'animate'), # Biological
    'agency':       ('passive', 'active'),    # Interaction
    
    # --- PHYSICAL PROPERTIES ---
    'size':         ('small', 'large'),
    'temperature':  ('cold', 'hot'),
    'weight':       ('light', 'heavy'),
    'hardness':     ('soft', 'hard'),
    'speed':        ('slow', 'fast'),
    'brightness':   ('dark', 'bright'),
    'state':        ('liquid', 'solid'),
    'strength':     ('weak', 'strong'),

    # --- ABSTRACT PROPERTIES ---
    'concreteness': ('abstract', 'concrete'),
    'complexity':   ('simple', 'complex'),
    'familiarity':  ('strange', 'familiar'),
    'safety':       ('dangerous', 'safe'),
    'certainty':    ('uncertain', 'certain'), # Epistemic
    'necessity':    ('possible', 'necessary'), # Modal
    'causality':    ('effect', 'cause'),      # Causal
    
    # --- SOCIAL/HUMAN ---
    'humanness':    ('object', 'human'),
    'gender':       ('feminine', 'masculine'), # Grammatical/Social
    'age':          ('old', 'young'),
    'wealth':       ('poor', 'rich'),
    'politics':     ('conservative', 'liberal'),
    'religion':     ('secular', 'religious'),
    'formality':    ('casual', 'formal'),
    'publicity':    ('private', 'public'),

    # --- EMOTIONAL ---
    'arousal':      ('calm', 'excited'),
    'dominance':    ('submissive', 'dominant'),
    
    # --- QUANTITY ---
    'quantity':     ('few', 'many'),
    'frequency':    ('rare', 'common'),
}

ORDERED_AXIS_NAMES = [
    'existence', 'valence', 'time', 'agency', 'animacy', 'concreteness',
    'spatial_x', 'spatial_y', 'spatial_z',
    'size', 'temperature', 'weight', 'speed', 'strength',
    'certainty', 'causality', 'safety',
    'humanness', 'gender', 'wealth', 'politics',
    'arousal', 'dominance', 'frequency'
] 
# (We select a subset or ordered list to ensure the core basis is robust)

# ==========================================
# 2. SPHERE CONSTRUCTION LOGIC
# ==========================================
class UniversalSemanticSphereV2:
    def __init__(self, axis_definitions=GROUNDED_AXES, seed_model_name="word2vec-google-news-300"):
        self.axis_definitions = axis_definitions
        self.basis_matrix = None # Shape (D, K) where K is num axes
        self.axis_names = []
        self.seed_model_name = seed_model_name
        self.seed_model = None

    def load_seed_model(self):
        """Robustly loads the seed model (Word2Vec) to initialize axes."""
        print(f"Loading seed model: {self.seed_model_name}...")
        manual_base = os.path.expanduser(f"~/gensim-data/{self.seed_model_name}/{self.seed_model_name}.gz")
        
        if os.path.exists(manual_base):
            print(f"Found local file at: {manual_base}")
            is_binary = "word2vec" in self.seed_model_name or self.seed_model_name.endswith(".bin.gz")
            self.seed_model = KeyedVectors.load_word2vec_format(manual_base, binary=is_binary)
        else:
            print("Downloading via API...")
            try:
                self.seed_model = api.load(self.seed_model_name)
            except Exception as e:
                print(f"Failed to load seed model: {e}")
                raise

    def build_basis(self, ordered_names=ORDERED_AXIS_NAMES):
        """
        Constructs the orthonormal universal basis using Stable Gram-Schmidt.
        """
        if self.seed_model is None:
            self.load_seed_model()

        print("Constructing Universal Basis from Grounded Axes...")
        raw_vectors = []
        valid_names = []

        for name in ordered_names:
            neg_word, pos_word = self.axis_definitions[name]
            
            # Check if exemplars exist
            if neg_word not in self.seed_model or pos_word not in self.seed_model:
                print(f"  Skipping axis '{name}': terms not found in seed.")
                continue

            # Axis Direction = Positive - Negative
            v_pos = self.seed_model[pos_word]
            v_neg = self.seed_model[neg_word]
            axis_vec = v_pos - v_neg
            
            # Normalize raw vector
            axis_vec /= np.linalg.norm(axis_vec)
            
            raw_vectors.append(axis_vec)
            valid_names.append(name)

        # Matrix shape: (D, N_axes) - we want axes as columns
        V = np.array(raw_vectors).T 
        
        # Gram-Schmidt Orthogonalization
        Q = self.stable_gram_schmidt(V)
        
        self.basis_matrix = Q
        self.axis_names = valid_names
        
        print(f"Basis Constructed. Shape: {self.basis_matrix.shape}")
        print(f"Axes: {self.axis_names}")
        return self.basis_matrix

    def stable_gram_schmidt(self, V):
        """
        Orthonormalizes column vectors in V.
        Preserves the direction of the first vector, 
        projects the second onto the complement of the first, etc.
        """
        Q = []
        for i in range(V.shape[1]):
            v = V[:, i]
            # Subtract projections onto existing basis vectors
            for q in Q:
                v = v - np.dot(v, q) * q
            
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                q = v / norm
                Q.append(q)
            else:
                # Linearly dependent axis (redundant)
                # For this application, we skip it or could add random perturbation
                pass
        
        return np.array(Q).T

    def project(self, vector):
        """
        Projects a raw vector (D-dim) onto the Universal Sphere (K-dim).
        Result is coords in terms of [Valence, Time, ...].
        """
        # project v onto basis columns: Coords = Q.T @ v
        if self.basis_matrix is None:
            raise ValueError("Basis not built.")
            
        coords = self.basis_matrix.T @ vector
        return coords

if __name__ == "__main__":
    # Test the construction
    sphere = UniversalSemanticSphereV2()
    sphere.build_basis()
    
    # Sanity Check: Project some concepts
    test_words = ['murder', 'puppy', 'tomorrow', 'stone', 'mathematics', 'running']
    
    print("\n--- Universal Coordinate Projection ---")
    headers = [n[:4] for n in sphere.axis_names[:8]] # First 8 axes
    print(f"{'WORD':<15} | " + " | ".join(headers))
    
    for w in test_words:
        if w in sphere.seed_model:
            vec = sphere.seed_model[w].copy()
            # Normalize input
            vec /= np.linalg.norm(vec)
            
            coords = sphere.project(vec)
            
            # Format first 8 coords
            vals = [f"{c:.2f}" for c in coords[:8]]
            print(f"{w:<15} | " + " | ".join(vals))
