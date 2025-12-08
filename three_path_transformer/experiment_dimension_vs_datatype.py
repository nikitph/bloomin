import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import urllib.request
import os
import glob

def get_text_data(n_samples=5000):
    print(f"Stats: Fetching {n_samples} text samples...")
    try:
        url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
        lines = [l.strip() for l in text.splitlines() if len(l) > 3]
        return lines[:n_samples]
    except Exception as e:
        print(f"Error fetching text: {e}")
        return [f"sample text {i}" for i in range(n_samples)]

def get_code_data(root_dir, n_samples=5000):
    print(f"Stats: Fetching {n_samples} code samples from {root_dir}...")
    code_lines = []
    for root, dirs, files in os.walk(root_dir):
        if 'venv' in root or '.git' in root: continue
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r', errors='ignore') as f:
                        for line in f:
                            clean = line.strip()
                            # Filter for meaningful code (not just imports or empty)
                            if len(clean) > 15 and not clean.startswith('#') and 'import' not in clean:
                                code_lines.append(clean)
                                if len(code_lines) >= n_samples:
                                    return code_lines
                except:
                    continue
    return code_lines

def get_synthetic_linear(n_samples=5000, dim=384, intrinsic_dim=50):
    print(f"Stats: Generating {n_samples} Synthetic Linear ({intrinsic_dim}D) vectors...")
    # Create fixed basis
    basis = np.random.randn(intrinsic_dim, dim)
    # Generate coefficients
    coeffs = np.random.randn(n_samples, intrinsic_dim)
    # Project
    X = np.dot(coeffs, basis)
    # Normalize
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def get_synthetic_hierarchy(n_samples=5000, dim=384, depth=5, branching=4):
    print(f"Stats: Generating {n_samples} Synthetic Hierarchy vectors...")
    # Simple random tree walk
    # Root
    root = np.random.randn(dim)
    root /= np.linalg.norm(root)
    
    vectors = []
    
    def generate_children(parent, current_depth):
        if len(vectors) >= n_samples: return
        
        # Add parent (concept)
        vectors.append(parent)
        
        if current_depth >= depth: return
        
        for _ in range(branching):
            # Child is Parent + Noise
            noise = np.random.randn(dim) * 0.5 
            child = parent + noise
            child /= np.linalg.norm(child)
            generate_children(child, current_depth + 1)
            
    # Generate broad tree
    start_nodes = 50
    for _ in range(start_nodes):
        node = np.random.randn(dim)
        node /= np.linalg.norm(node)
        generate_children(node, 1)
        
    X = np.array(vectors[:n_samples])
    return X

def get_random_embeddings(n_samples=5000, dim=384):
    print(f"Stats: Generating {n_samples} random noise vectors...")
    # Generate on sphere
    X = np.random.randn(n_samples, dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def analyze_intrinsic_dimension(embeddings, label):
    pca = PCA()
    pca.fit(embeddings)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d_90 = np.searchsorted(cumsum, 0.90) + 1
    d_95 = np.searchsorted(cumsum, 0.95) + 1
    d_99 = np.searchsorted(cumsum, 0.99) + 1
    
    print(f"\ndataset: {label}")
    print(f"  D_90: {d_90}")
    print(f"  D_95: {d_95}")
    print(f"  D_99: {d_99}")
    
    return cumsum, d_95, d_99

def run_experiment():
    # 1. Setup
    model = SentenceTransformer('all-MiniLM-L6-v2') # 384 dim
    n = 5000
    
    # 2. Get Data
    texts = get_text_data(n)
    code = get_code_data('/Users/truckx/PycharmProjects/bloomin', n)
    
    # 3. Encode
    print("\nEncoding Text...")
    emb_text = model.encode(texts, show_progress_bar=True)
    
    print("Encoding Code...")
    emb_code = model.encode(code, show_progress_bar=True)
    
    emb_rand = get_random_embeddings(n, 384)
    emb_syn_linear = get_synthetic_linear(n, 384, intrinsic_dim=50)
    emb_syn_hier = get_synthetic_hierarchy(n, 384)
    
    # 4. Analyze
    results = {}
    results['Random Noise'] = analyze_intrinsic_dimension(emb_rand, 'Random Noise')
    results['Synthetic Linear (50D)'] = analyze_intrinsic_dimension(emb_syn_linear, 'Synthetic Linear (50D)')
    results['Synthetic Hierarchy'] = analyze_intrinsic_dimension(emb_syn_hier, 'Synthetic Hierarchy')
    results['Natural Language'] = analyze_intrinsic_dimension(emb_text, 'Natural Language')
    results['Python Code'] = analyze_intrinsic_dimension(emb_code, 'Python Code')
    
    # 5. Plot
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Random Noise': 'gray', 
        'Natural Language': 'blue', 
        'Python Code': 'green',
        'Synthetic Linear (50D)': 'red',
        'Synthetic Hierarchy': 'purple'
    }
    styles = {
        'Random Noise': ':', 
        'Natural Language': '-', 
        'Python Code': '--',
        'Synthetic Linear (50D)': '-.',
        'Synthetic Hierarchy': '-.'
    }
    
    for name, (curve, d95, d99) in results.items():
        if name not in colors: continue 
        label_text = f"{name} (95%={d95}D)"
        plt.plot(curve, label=label_text, color=colors[name], linestyle=styles[name], linewidth=2)
        # Mark 95% point
        plt.plot(d95, 0.95, 'o', color=colors[name])
        
    plt.axhline(0.95, color='black', linestyle='-', alpha=0.2, label='95% Variance')
    plt.axhline(0.99, color='black', linestyle='--', alpha=0.2, label='99% Variance')
    
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Geometry of Structure: Data Types vs Synthetic Baselines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 384)
    plt.ylim(0, 1.05)
    
    output_path = "dimension_vs_datatype_synthetic.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
