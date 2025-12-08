import numpy as np
import matplotlib.pyplot as plt
from model_loaders import GensimLoader
import sys
import os

def check_norms():
    # Load model
    loader = GensimLoader()
    try:
        model = loader.load_model('word2vec-google-news-300')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get sample vectors using keys present in the model
    # Note: gensim 4.0+ uses key_to_index
    try:
        vocab = list(model.key_to_index.keys())
    except AttributeError:
        # Fallback for older gensim
        vocab = list(model.vocab.keys())
        
    keys = vocab[:20000] # First 20k (frequent)
    
    import random
    random.shuffle(keys)
    sample_keys = keys[:10000]
    
    embeddings = loader.get_embeddings(sample_keys)
    
    # Compute Norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    cv = std_norm / mean_norm
    
    print(f"\nWord2Vec Norm Statistics (n={len(embeddings)}):")
    print(f"  Mean: {mean_norm:.4f}")
    print(f"  Std:  {std_norm:.4f}")
    print(f"  CV:   {cv:.4f} (Coefficient of Variation)")
    print(f"  Min:  {np.min(norms):.4f}")
    print(f"  Max:  {np.max(norms):.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(mean_norm, color='red', linestyle='--', label=f'Mean: {mean_norm:.2f}')
    plt.title('Distribution of Word2Vec Vector Norms')
    plt.xlabel('Euclidean Norm ||v||')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'w2vec_norm_dist.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # Theoretical Note
    print("\nNOTE: Unnormalized Curvature")
    print("If we assume the space is Euclidean (raw vectors), K=0 by definition.")
    print("If we assumes the space is Angular (Cosine), normalization is implicit.")
    print("This variance in norms represents 'Frequency/Importance' info, distinguishing the 'Cone' from the 'Sphere'.")

if __name__ == "__main__":
    check_norms()
