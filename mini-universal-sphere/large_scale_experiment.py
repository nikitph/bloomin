import numpy as np
import matplotlib.pyplot as plt
from universal_sphere_v2 import UniversalSemanticSphereV2

# ==========================================
# 0. DATASET (50 Polysemous Words)
# ==========================================
POLYSEMOUS_DATASET = [
    # Format: (Word, Context A, Context B)
    ('bank', 'money', 'river'),
    ('apple', 'pie', 'computer'),
    ('star', 'sky', 'movie'),
    ('cell', 'prison', 'phone'),
    ('run', 'jog', 'program'),
    ('spring', 'season', 'coil'),
    ('bat', 'baseball', 'animal'),
    ('scale', 'weight', 'fish'),
    ('crane', 'bird', 'construction'),
    ('date', 'calendar', 'fruit'),
    ('right', 'correct', 'left'),
    ('match', 'game', 'fire'),
    ('light', 'lamp', 'heavy'),
    ('bark', 'dog', 'tree'),
    ('band', 'music', 'rubber'),
    ('bar', 'pub', 'chocolate'),
    ('bass', 'fish', 'guitar'),
    ('bow', 'arrow', 'tie'),
    ('capital', 'city', 'money'),
    ('chest', 'body', 'box'),
    ('chip', 'potato', 'computer'),
    ('clip', 'paper', 'video'),
    ('club', 'golf', 'party'),
    ('code', 'program', 'secret'),
    ('court', 'law', 'tennis'),
    ('deck', 'cards', 'ship'),
    ('fan', 'electric', 'sports'),
    ('file', 'document', 'tool'),
    ('glass', 'drink', 'window'),
    ('jam', 'traffic', 'strawberry'),
    ('key', 'lock', 'piano'),
    ('letter', 'mail', 'alphabet'),
    ('mole', 'animal', 'skin'),
    ('nail', 'finger', 'hammer'),
    ('net', 'internet', 'fish'),
    ('novel', 'book', 'new'),
    ('organ', 'body', 'instrument'),
    ('palm', 'hand', 'tree'),
    ('park', 'car', 'nature'),
    ('pitch', 'sound', 'baseball'),
    ('point', 'dot', 'opinion'),
    ('pole', 'north', 'rod'),
    ('pound', 'money', 'weight'),
    ('press', 'news', 'push'),
    ('pupil', 'student', 'eye'),
    ('racket', 'tennis', 'noise'),
    ('record', 'music', 'file'),
    ('ring', 'jewelry', 'bell'),
    ('rock', 'stone', 'music'),
    ('root', 'tree', 'math'),
    ('row', 'boat', 'line'),
    ('seal', 'animal', 'stamp'),
    ('sign', 'symbol', 'contract'),
    ('sink', 'kitchen', 'fall'),
    ('space', 'outer', 'room'),
    ('speaker', 'audio', 'person'),
    ('spot', 'place', 'dot'),
    ('stick', 'branch', 'glue'),
    ('table', 'furniture', 'data'),
    ('tank', 'fish', 'army'),
    ('tie', 'rope', 'suit'),
    ('toast', 'bread', 'speech'),
    ('track', 'run', 'music'),
    ('trip', 'travel', 'fall'),
    ('wave', 'ocean', 'hand'),
    ('well', 'good', 'water'),
]

# ==========================================
# 1. WITNESS EXTRACTOR (Simplified)
# ==========================================
class WitnessExtractor:
    def __init__(self, model):
        self.model = model

    def get_witnesses(self, word, context=None, top_k=50):
        if word not in self.model: return [], []
        if context and context not in self.model: return [], []

        if context:
            # Vector Addition for Context
            target_vec = self.model[word] + self.model[context]
        else:
            target_vec = self.model[word]

        neighbors = self.model.similar_by_vector(target_vec, topn=top_k)
        witness_words = [n[0] for n in neighbors]
        # Robust copy
        witness_vecs = np.array([self.model[w].copy() for w in witness_words])
        
        norms = np.linalg.norm(witness_vecs, axis=1, keepdims=True)
        witness_vecs = witness_vecs / norms
        
        return witness_words, witness_vecs

# ==========================================
# 2. EXPERIMENT LOGIC
# ==========================================
def run_validation():
    # Init
    sphere = UniversalSemanticSphereV2()
    sphere.build_basis() # Load w2v
    extractor = WitnessExtractor(sphere.seed_model)
    
    shifts = []
    separations = []
    details = []

    print(f"\nProcessing {len(POLYSEMOUS_DATASET)} words...")
    
    for word, ctx_a, ctx_b in POLYSEMOUS_DATASET:
        # A. Get Witnesses & Project
        _, vec_base = extractor.get_witnesses(word)
        _, vec_a = extractor.get_witnesses(word, ctx_a)
        _, vec_b = extractor.get_witnesses(word, ctx_b)
        
        if len(vec_base) == 0 or len(vec_a) == 0 or len(vec_b) == 0:
            print(f"Skipping {word}: missing vectors.")
            continue
            
        proj_base = sphere.project(vec_base.T).T
        proj_a = sphere.project(vec_a.T).T
        proj_b = sphere.project(vec_b.T).T
        
        # B. Compute Centroids
        cent_base = np.mean(proj_base, axis=0)
        cent_a = np.mean(proj_a, axis=0)
        cent_b = np.mean(proj_b, axis=0)
        
        # C. Metrics
        # Shift: Avg distance from Base to a Context
        dist_a = np.linalg.norm(cent_base - cent_a)
        dist_b = np.linalg.norm(cent_base - cent_b)
        avg_shift = (dist_a + dist_b) / 2.0
        
        # Separation: Distance between Context A and Context B
        separation = np.linalg.norm(cent_a - cent_b)
        
        shifts.append(avg_shift)
        separations.append(separation)
        
        details.append({
            'word': word, 'shift': avg_shift, 'separation': separation
        })
        
        # Print first few
        if len(shifts) <= 5:
            print(f"  {word.upper()}: Shift={avg_shift:.3f}, Sep={separation:.3f} ({ctx_a}/{ctx_b})")

    # ==========================================
    # 3. ANALYSIS
    # ==========================================
    shifts = np.array(shifts)
    separations = np.array(separations)
    
    mean_shift = np.mean(shifts)
    mean_sep = np.mean(separations)
    
    print("\n" + "="*40)
    print("RESULTS: LARGE SCALE VALIDATION")
    print("="*40)
    print(f"Count: {len(shifts)} words")
    print(f"Mean Centroid Shift: {mean_shift:.3f} (Base -> Context)")
    print(f"Mean Context Separation: {mean_sep:.3f} (Context A <-> Context B)")
    
    # Success Criteria check
    if mean_shift > 0.15 and mean_sep > 0.2:
        print("\n✅ STATUS: SUCCESS.")
        print("   Consistent Geometric Disambiguation observed.")
    else:
        print("\n⚠️ STATUS: WEAK EFFECT.")
    
    # Plot Histograms
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(shifts, bins=15, color='blue', alpha=0.7)
    plt.title(f"Distribution of Centroid Shifts\n(Mean: {mean_shift:.2f})")
    plt.xlabel("Euclidean Distance (Base -> Context)")
    
    plt.subplot(1, 2, 2)
    plt.hist(separations, bins=15, color='green', alpha=0.7)
    plt.title(f"Distribution of Context Separations\n(Mean: {mean_sep:.2f})")
    plt.xlabel("Euclidean Distance (Context A <-> Context B)")
    
    plt.tight_layout()
    plt.savefig("large_scale_stats.png")
    print("Plot saved to 'large_scale_stats.png'")

if __name__ == "__main__":
    run_validation()
