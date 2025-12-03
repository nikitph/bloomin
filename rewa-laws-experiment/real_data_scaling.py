import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import binarize
from scipy import sparse
import time

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12

class RealDataLab:
    def __init__(self, dataset_name, max_samples=5000):
        self.name = dataset_name
        print(f"Loading {dataset_name}...")
        start = time.time()
        
        if dataset_name == '20Newsgroups':
            # Load subset to keep it fast
            data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            X = vectorizer.fit_transform(data.data).tocsr()  # Keep sparse!
            # Take subset if too large
            if X.shape[0] > max_samples:
                idx = np.random.choice(X.shape[0], max_samples, replace=False)
                X = X[idx]
                self.y = data.target[idx]
            else:
                self.y = data.target
            # Binarize (faster on sparse)
            X.data = (X.data > 0).astype(np.int8)
            self.X = X
            
        elif dataset_name == 'Digits':
            data = load_digits()
            self.X = (data.data > 8).astype(np.int8)
            self.y = data.target
            if len(self.X) > max_samples:
                idx = np.random.choice(len(self.X), max_samples, replace=False)
                self.X = self.X[idx]
                self.y = self.y[idx]
            
        elif dataset_name == 'MNIST':
            # Use sklearn's smaller version or cache the download
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
            # Subsample immediately
            idx = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
            self.X = (X[idx] > 127).astype(np.int8)
            self.y = y[idx].astype(int)
            
        self.N, self.D = self.X.shape
        print(f"  Shape: {self.X.shape}, Time: {time.time()-start:.2f}s")
        
    def get_decimated_view(self, p, seed=42):
        """Returns view with fraction p of features"""
        rng = np.random.default_rng(seed)
        num_features = max(1, int(p * self.D))
        selected_indices = rng.choice(self.D, num_features, replace=False)
        selected_indices.sort()  # Sorting helps performance
        
        if sparse.issparse(self.X):
            return self.X[:, selected_indices]
        else:
            return self.X[:, selected_indices]

    def measure_delta(self, X_view, num_samples=500):
        """
        Measures the actual semantic gap Delta by computing
        average cosine similarity between nearest neighbors.
        Returns Delta as a proxy for signal strength.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Sample subset for speed
        N = X_view.shape[0]
        if N > num_samples:
            idx = np.random.choice(N, num_samples, replace=False)
            X_sample = X_view[idx] if sparse.issparse(X_view) else X_view[idx]
            y_sample = self.y[idx]
        else:
            X_sample = X_view
            y_sample = self.y
            
        # Compute pairwise cosine similarity
        Sim = cosine_similarity(X_sample)
        np.fill_diagonal(Sim, -np.inf)
        
        # Find nearest neighbors
        nearest = np.argmax(Sim, axis=1)
        
        # Compute average similarity for same-class pairs
        same_class_sims = []
        for i in range(len(y_sample)):
            if y_sample[i] == y_sample[nearest[i]]:
                same_class_sims.append(Sim[i, nearest[i]])
                
        if len(same_class_sims) > 0:
            avg_sim = np.mean(same_class_sims)
            # Convert similarity to a gap measure
            # Higher similarity = higher Delta (signal)
            # We use similarity directly as Delta proxy
            delta = avg_sim
        else:
            delta = 0.01  # Fallback
            
        return delta

    def encode_and_eval_vectorized(self, X_view, m, seed=42):
        """
        Vectorized version using sparse matrix operations
        """
        rng = np.random.default_rng(seed)
        D_view = X_view.shape[1]
        
        # Create hash mapping using sparse matrix
        # Each feature j maps to one random bit in [0, m)
        feature_to_bit = rng.integers(0, m, size=D_view)
        
        # Create projection matrix P (D_view x m) as sparse CSR
        rows = np.arange(D_view)
        cols = feature_to_bit
        data = np.ones(D_view, dtype=np.int8)
        P = sparse.csr_matrix((data, (rows, cols)), shape=(D_view, m))
        
        # Encode: B = (X_view * P) > 0  (sparse matrix multiplication)
        # Use logical OR operation (any non-zero)
        if sparse.issparse(X_view):
            B = (X_view @ P).astype(bool).astype(np.int8)
        else:
            B = (X_view @ P.toarray()).astype(bool).astype(np.int8)
        
        # For large N, we need to compute similarity in batches
        # Use matrix multiplication for similarity
        # For faster computation, we can use the fact that similarity
        # between binary vectors = sum of AND operations
        
        # Alternative: use sklearn's pairwise_distances for cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute cosine similarity (faster for sparse/dense)
        Sim = cosine_similarity(B)
        np.fill_diagonal(Sim, -np.inf)
        
        nearest = np.argmax(Sim, axis=1)
        acc = np.mean(self.y[nearest] == self.y)
        
        return acc

def run_experiment():
    # Use smaller datasets and fewer iterations
    datasets = ['Digits', '20Newsgroups']  # Remove MNIST or add later
    ps = [0.25, 0.5, 1.0]
    ms = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])  # Start very low to see the transition
    
    fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 6 * len(datasets)))
    
    for i, ds_name in enumerate(datasets):
        print(f"\n--- Processing {ds_name} ---")
        start_ds = time.time()
        lab = RealDataLab(ds_name, max_samples=2000)  # Limit samples
        
        results = {p: [] for p in ps}
        deltas = {}  # Store measured deltas
        
        for p in ps:
            print(f"  Decimation p={p}...")
            X_view = lab.get_decimated_view(p)
            
            # Measure actual Delta
            delta_measured = lab.measure_delta(X_view)
            deltas[p] = delta_measured
            print(f"    Measured Delta: {delta_measured:.4f}")
            
            for m_idx, m in enumerate(ms):
                print(f"    m={m} ({m_idx+1}/{len(ms)})", end='\r')
                acc = lab.encode_and_eval_vectorized(X_view, m)
                results[p].append(acc)
            print()  # New line after m loop
            
        # Plotting code remains the same...
        ax_raw = axes[i][0]
        for p in ps:
            ax_raw.plot(ms, results[p], 'o-', label=f'Features p={p}')
        ax_raw.set_xscale('log')
        ax_raw.set_title(f'{ds_name}: Raw Phase Transitions')
        ax_raw.set_ylabel('Top-1 Accuracy')
        ax_raw.legend()
        ax_raw.grid(True, which="both", ls="-")
        
        ax_scaled = axes[i][1]
        for p in ps:
            # Use measured Delta^2 for scaling
            delta = deltas[p]
            x_scaled = ms * (delta**2)
            # Normalize accuracy by max accuracy for this p
            max_acc = max(results[p])
            normalized_acc = np.array(results[p]) / max_acc if max_acc > 0 else np.array(results[p])
            ax_scaled.plot(x_scaled, normalized_acc, 'o-', 
                          label=f'p={p} ($\\Delta$={delta:.3f})')
            
        ax_scaled.set_xscale('log')
        ax_scaled.set_title(f'{ds_name}: Scaling Collapse ($x = m \\cdot \\Delta_{{measured}}^2$)')
        ax_scaled.set_xlabel('Thermodynamic Variable ($m \\cdot \\Delta^2$)')
        ax_scaled.set_ylabel('Normalized Accuracy (Acc / Max_Acc)')
        ax_scaled.legend()
        ax_scaled.grid(True, which="both", ls="-")
        ax_scaled.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Max Capacity')
        
        print(f"  Dataset time: {time.time()-start_ds:.2f}s")
        
    plt.tight_layout()
    plt.savefig('real_data_scaling_collapse.png')
    print("\nSaved real_data_scaling_collapse.png")

if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
    print(f"\nTotal execution time: {time.time()-start_time:.2f} seconds")
