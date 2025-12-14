import numpy as np
import matplotlib.pyplot as plt
from universal_sphere_v2 import UniversalSemanticSphereV2
from large_scale_experiment import WitnessExtractor
from multilingual_data import SPANISH_AXES, SPANISH_DATASET, HINDI_AXES, HINDI_DATASET
import os
from gensim.models import KeyedVectors

# ==========================================
# EXPERIMENT CONFIG
# ==========================================
LANGUAGES = {
    'ES': {
        'model_path': '~/gensim-data/fasttext-es/cc.es.300.vec.gz',
        'axes': SPANISH_AXES,
        'dataset': SPANISH_DATASET
    },
    'HI': {
        'model_path': '~/gensim-data/fasttext-hi/cc.hi.300.vec.gz',
        'axes': HINDI_AXES,
        'dataset': HINDI_DATASET
    }
}

def load_local_fasttext(path):
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        print(f"Model not found at {expanded}")
        return None
    
    print(f"Loading FastText model from {expanded}...")
    try:
        # FastText .vec files are word2vec format (text)
        return KeyedVectors.load_word2vec_format(expanded, binary=False, limit=500000)
    except Exception as e:
        print(f"Failed to load: {e}")
        return None

def run_multilingual_validation():
    results_summary = {}

    for lang, config in LANGUAGES.items():
        print(f"\n" + "="*40)
        print(f"RUNNING VALIDATION FOR: {lang}")
        print("="*40)
        
        # 1. Load Model
        model = load_local_fasttext(config['model_path'])
        if model is None:
            print(f"Skipping {lang} (Model missing - please download cc.{lang.lower()}.300.vec.gz)")
            continue
            
        # 2. Build Sphere (Language Specific)
        # We inject the model as the 'seed_model' and use translated axes
        sphere = UniversalSemanticSphereV2(axis_definitions=config['axes'])
        sphere.seed_model = model # Inject manually
        sphere.build_basis(ordered_names=list(config['axes'].keys()))
        
        # 3. Setup Extractor
        extractor = WitnessExtractor(model)
        
        # 4. Run Polysemy Test
        shifts = []
        separations = []
        
        print(f"Processing {len(config['dataset'])} words...")
        
        for word, ctx_a, ctx_b in config['dataset']:
            # A. Extractor
            # Note: FastText handles OOV with n-grams usually, but gensim KeyedVectors.load_word2vec_format 
            # loads it as a static dict unless we load as FastText native. 
            # For this exp, we assume the words are in vocab.
            
            _, vec_base = extractor.get_witnesses(word)
            _, vec_a = extractor.get_witnesses(word, ctx_a)
            _, vec_b = extractor.get_witnesses(word, ctx_b)
            
            if len(vec_base) == 0:
                print(f"  Missing word: {word}")
                continue
                
            # B. Project
            proj_base = sphere.project(vec_base.T).T
            proj_a = sphere.project(vec_a.T).T
            proj_b = sphere.project(vec_b.T).T
            
            # C. Centroids
            cent_base = np.mean(proj_base, axis=0)
            cent_a = np.mean(proj_a, axis=0)
            cent_b = np.mean(proj_b, axis=0)
            
            # D. Metrics
            avg_shift = (np.linalg.norm(cent_base - cent_a) + np.linalg.norm(cent_base - cent_b)) / 2.0
            separation = np.linalg.norm(cent_a - cent_b)
            
            shifts.append(avg_shift)
            separations.append(separation)
            
            print(f"  {word}: Sep={separation:.3f}")
            
        # 5. Stats
        mean_sep = np.mean(separations)
        print(f"\n{lang} RESULT: Mean Separation = {mean_sep:.3f}")
        results_summary[lang] = mean_sep
        
        # Plot
        plt.figure()
        plt.hist(separations, bins=10, alpha=0.7)
        plt.title(f"{lang}: Context Separation (Mean {mean_sep:.3f})")
        plt.savefig(f"multilingual_{lang}_stats.png")

    print("\n" + "="*40)
    print("FINAL MULTILINGUAL RESULTS")
    print("="*40)
    for lang, score in results_summary.items():
        print(f"{lang}: {score:.4f} (> 0.2 is Success)")

if __name__ == "__main__":
    run_multilingual_validation()
