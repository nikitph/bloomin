# -*- coding: utf-8 -*-
"""
UNIFIED INVESTIGATION SUITE: Probing the Rough Edges of Semantic Thermodynamics
WITH REAL NEWSGROUPS DATA

This script runs 5 targeted experiments to investigate:
1. K-Dependence: Does C_M increase with K? (Explaining the mismatch)
2. N-Scaling: Does chi_c scale logarithmically with N? (Core theory validation)
3. Rho-Dependence: Is universality robust across signal strengths?
4. L-Dependence: Does witness count matter?
5. Precision: High-resolution transition analysis.

Uses real text from 20 newsgroups dataset instead of synthetic data.
Runtime: ~3-4 hours total (with n_trials=3)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import time
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import re

# Create output directory
OUTPUT_DIR = 'newsgroups_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Publication Style ---
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

class REWALabNewsgroups:
    """REWA Laboratory using real newsgroups text data"""
    
    def __init__(self, N=2000, max_vocab_size=10000, seed=42):
        self.N = N
        self.max_vocab_size = max_vocab_size
        self.rng = np.random.default_rng(seed)
        self.items = None
        self.cluster_labels = None
        self.W_universe = 0
        self.L = 0  # Will be set based on actual data
        
    def load_newsgroups_data(self, subset='train', categories=None):
        """Load newsgroups data and convert to witness sets"""
        print(f"Loading 20 newsgroups dataset (subset={subset})...")
        
        # Fetch newsgroups data
        newsgroups = fetch_20newsgroups(
            subset=subset,
            categories=categories,
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        
        # Limit to N documents
        if len(newsgroups.data) > self.N:
            indices = self.rng.choice(len(newsgroups.data), size=self.N, replace=False)
            texts = [newsgroups.data[i] for i in indices]
            labels = newsgroups.target[indices]
        else:
            texts = newsgroups.data
            labels = newsgroups.target
            self.N = len(texts)
        
        print(f"Processing {self.N} documents from {len(np.unique(labels))} categories...")
        
        # Tokenize and build vocabulary
        # Use simple word tokenization
        vectorizer = CountVectorizer(
            max_features=self.max_vocab_size,
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8,  # Word must appear in at most 80% of documents
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Words with 3+ letters
        )
        
        # Fit and transform
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # Convert to witness sets
        self.items = []
        for i in range(self.N):
            # Get non-zero indices (words present in this document)
            witnesses = doc_term_matrix[i].nonzero()[1]
            self.items.append(set(witnesses))
        
        self.cluster_labels = labels
        self.W_universe = len(vectorizer.vocabulary_)
        
        # Calculate average document length (witnesses per item)
        self.L = int(np.mean([len(item) for item in self.items]))
        
        print(f"Vocabulary size (W_universe): {self.W_universe}")
        print(f"Average document length (L): {self.L}")
        print(f"Number of categories: {len(np.unique(labels))}")
        
        return self
    
    def measure_gap(self, n_samples=200):
        """Measure the gap between same-cluster and different-cluster overlaps"""
        same_cluster_overlaps = []
        diff_cluster_overlaps = []
        
        for _ in range(n_samples):
            i, j = self.rng.choice(self.N, size=2, replace=False)
            
            # Calculate overlap (Jaccard-like but normalized by L)
            intersection = len(self.items[i] & self.items[j])
            # Normalize by average length to get a proportion
            avg_len = (len(self.items[i]) + len(self.items[j])) / 2
            if avg_len > 0:
                overlap = intersection / avg_len
            else:
                overlap = 0
            
            if self.cluster_labels[i] == self.cluster_labels[j]:
                same_cluster_overlaps.append(overlap)
            else:
                diff_cluster_overlaps.append(overlap)
        
        Delta_same = np.mean(same_cluster_overlaps) if same_cluster_overlaps else 0
        Delta_diff = np.mean(diff_cluster_overlaps) if diff_cluster_overlaps else 0
        Delta_gap = Delta_same - Delta_diff
        
        print(f"  Delta_same: {Delta_same:.4f}, Delta_diff: {Delta_diff:.4f}, Gap: {Delta_gap:.4f}")
        
        return Delta_gap
    
    def encode(self, m, K_hashes=2, seed=42):
        """Encode witness sets using LSH"""
        rng_local = np.random.default_rng(seed)
        hash_map = rng_local.integers(0, m, size=(self.W_universe, K_hashes))
        
        encoded_matrix = np.zeros((self.N, m), dtype=np.int16)
        
        for i, witnesses in enumerate(self.items):
            w_indices = np.array(list(witnesses), dtype=int)
            if len(w_indices) > 0:
                indices = hash_map[w_indices].flatten()
                encoded_matrix[i, indices] = 1
        
        return encoded_matrix
    
    def evaluate_retrieval(self, encoded_matrix):
        """Evaluate retrieval accuracy using nearest neighbor"""
        Sim = np.dot(encoded_matrix, encoded_matrix.T).astype(float)
        np.fill_diagonal(Sim, -np.inf)
        
        nearest_indices = np.argmax(Sim, axis=1)
        pred_clusters = self.cluster_labels[nearest_indices]
        
        return np.mean(pred_clusters == self.cluster_labels)

def sigmoid(x, x_c, a, y_min, y_max):
    return y_min + (y_max - y_min) / (1 + np.exp(-a * (x - x_c)))

def fit_critical_point(m_values, accuracies, Delta):
    """Fit sigmoid to find critical point chi_c"""
    x_scaled = m_values * (Delta ** 2)
    try:
        popt, _ = curve_fit(sigmoid, x_scaled, accuracies, 
                           p0=[np.median(x_scaled), 0.05, 0.0, 1.0],
                           bounds=([0, 0, 0, 0.8], [np.inf, 1, 0.2, 1.0]),
                           maxfev=5000)
        return popt[0], popt[1] # chi_c, a
    except:
        return None, None

def run_experiment_config(lab, m_values, K, n_trials=3):
    """Run a single configuration on pre-loaded data"""
    Delta = lab.measure_gap(n_samples=500)
    
    accuracies = []
    for m in m_values:
        trial_accs = []
        for t in range(n_trials):
            enc = lab.encode(m=m, K_hashes=K, seed=42+t)
            trial_accs.append(lab.evaluate_retrieval(enc))
        accuracies.append(np.mean(trial_accs))
    
    chi_c, width = fit_critical_point(m_values, accuracies, Delta)
    return chi_c, width, accuracies, Delta

# ============================================================================
# EXPERIMENT 1: K-Dependence (The C_M Mystery)
# ============================================================================
def experiment_1_K_dependence(lab):
    print("\n" + "="*60)
    print(" EXPERIMENT 1: K-Dependence (The C_M Mystery)")
    print("="*60)
    
    K_values = [2, 3, 4, 6, 8]
    m_values = np.geomspace(16, 4096, num=15, dtype=int)
    results = {'K': [], 'chi_c': []}
    
    plt.figure(figsize=(10, 6))
    
    for K in K_values:
        print(f"Testing K={K}...")
        chi_c, _, accs, Delta = run_experiment_config(lab, m_values, K=K)
        if chi_c:
            results['K'].append(K)
            results['chi_c'].append(chi_c)
            plt.plot(m_values * Delta**2, accs, 'o-', label=f'K={K} (χ_c={chi_c:.1f})')
            print(f"  -> χ_c = {chi_c:.2f}")
    
    plt.xscale('log')
    plt.xlabel('χ = m·Δ²')
    plt.ylabel('Accuracy')
    plt.title('Effect of Hash Functions K on Critical Point (Newsgroups Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'investigation_exp1_K_dependence.png'))
    plt.close()
    
    # Plot C_M vs K
    plt.figure(figsize=(8, 5))
    plt.plot(results['K'], results['chi_c'], 's-', color='purple', markersize=8)
    plt.xlabel('Number of Hash Functions K')
    plt.ylabel('Critical Point χ_c')
    plt.title('Does C_M increase with K? (Newsgroups Data)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'investigation_exp1_chi_vs_K.png'))
    plt.close()
    
    return results

# ============================================================================
# EXPERIMENT 2: N-Scaling (The Core Theory Test)
# ============================================================================
def experiment_2_N_scaling():
    print("\n" + "="*60)
    print(" EXPERIMENT 2: N-Scaling (The Core Theory Test)")
    print("="*60)
    
    N_values = [500, 1000, 2000, 3000]
    m_values = np.geomspace(16, 8192, num=15, dtype=int)
    results = {'N': [], 'chi_c': []}
    
    for N in N_values:
        print(f"Testing N={N}...")
        lab_n = REWALabNewsgroups(N=N, max_vocab_size=10000, seed=42)
        lab_n.load_newsgroups_data(subset='train')
        
        chi_c, _, _, _ = run_experiment_config(lab_n, m_values, K=4)
        if chi_c:
            results['N'].append(N)
            results['chi_c'].append(chi_c)
            print(f"  -> χ_c = {chi_c:.2f}")

    # Plot χ_c vs log(N)
    plt.figure(figsize=(8, 6))
    log_N = np.log(results['N'])
    plt.plot(log_N, results['chi_c'], 'o', markersize=8, label='Observed')
    
    # Linear fit
    slope, intercept = np.polyfit(log_N, results['chi_c'], 1)
    plt.plot(log_N, slope*log_N + intercept, 'r--', 
             label=f'Fit: slope (C_M) = {slope:.2f}')
    
    plt.xlabel('log(N)')
    plt.ylabel('Critical Point χ_c')
    plt.title(f'N-Scaling Validation: χ_c ∝ log(N) (Newsgroups)\nEmpirical C_M = {slope:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'investigation_exp2_N_scaling.png'))
    plt.close()
    
    return results

# ============================================================================
# EXPERIMENT 3: Vocabulary Size Dependence (W_universe)
# ============================================================================
def experiment_3_vocab_size():
    print("\n" + "="*60)
    print(" EXPERIMENT 3: Vocabulary Size Dependence")
    print("="*60)
    
    vocab_sizes = [2000, 4000, 6000, 8000, 10000]
    m_values = np.geomspace(16, 8192, num=15, dtype=int)
    results = {'vocab_size': [], 'chi_c': []}
    
    for vocab_size in vocab_sizes:
        print(f"Testing vocab_size={vocab_size}...")
        lab_v = REWALabNewsgroups(N=2000, max_vocab_size=vocab_size, seed=42)
        lab_v.load_newsgroups_data(subset='train')
        
        chi_c, _, _, _ = run_experiment_config(lab_v, m_values, K=4)
        if chi_c:
            results['vocab_size'].append(vocab_size)
            results['chi_c'].append(chi_c)
            print(f"  -> χ_c = {chi_c:.2f}")
            
    plt.figure(figsize=(8, 5))
    plt.plot(results['vocab_size'], results['chi_c'], 'o-', color='green')
    plt.xlabel('Vocabulary Size (W_universe)')
    plt.ylabel('Critical Point χ_c')
    plt.title('Effect of Vocabulary Size on Critical Point')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(OUTPUT_DIR, 'investigation_exp3_vocab_dependence.png'))
    plt.close()
    
    return results

# ============================================================================
# EXPERIMENT 4: Category Subset (Different Cluster Structures)
# ============================================================================
def experiment_4_category_subsets():
    print("\n" + "="*60)
    print(" EXPERIMENT 4: Category Subset Analysis")
    print("="*60)
    
    # Test with different numbers of categories
    category_counts = [5, 10, 15, 20]
    m_values = np.geomspace(16, 8192, num=15, dtype=int)
    results = {'n_categories': [], 'chi_c': []}
    
    all_categories = fetch_20newsgroups(subset='train').target_names
    
    for n_cat in category_counts:
        print(f"Testing with {n_cat} categories...")
        categories = all_categories[:n_cat]
        
        lab_c = REWALabNewsgroups(N=2000, max_vocab_size=10000, seed=42)
        lab_c.load_newsgroups_data(subset='train', categories=categories)
        
        chi_c, _, _, _ = run_experiment_config(lab_c, m_values, K=4)
        if chi_c:
            results['n_categories'].append(n_cat)
            results['chi_c'].append(chi_c)
            print(f"  -> χ_c = {chi_c:.2f}")
            
    plt.figure(figsize=(8, 5))
    plt.plot(results['n_categories'], results['chi_c'], 'd-', color='orange')
    plt.xlabel('Number of Categories')
    plt.ylabel('Critical Point χ_c')
    plt.title('Effect of Cluster Count on Critical Point')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'investigation_exp4_category_dependence.png'))
    plt.close()
    
    return results

# ============================================================================
# EXPERIMENT 5: Precision (Transition Sharpness)
# ============================================================================
def experiment_5_precision(lab):
    print("\n" + "="*60)
    print(" EXPERIMENT 5: Precision (Transition Sharpness)")
    print("="*60)
    
    # Dense sampling around expected critical point
    m_dense = np.linspace(50, 400, num=30, dtype=int)
    n_trials_high = 10
    
    print(f"Running dense sweep with {n_trials_high} trials...")
    Delta = lab.measure_gap(n_samples=1000)
    
    accuracies = []
    std_devs = []
    
    for m in m_dense:
        trial_accs = []
        for t in range(n_trials_high):
            enc = lab.encode(m=m, K_hashes=4, seed=100+t)
            trial_accs.append(lab.evaluate_retrieval(enc))
        accuracies.append(np.mean(trial_accs))
        std_devs.append(np.std(trial_accs))
        
    # Fit with error bars
    x_scaled = m_dense * Delta**2
    popt, pcov = curve_fit(sigmoid, x_scaled, accuracies, 
                          sigma=std_devs, absolute_sigma=True,
                          p0=[np.median(x_scaled), 0.05, 0.0, 1.0])
    
    chi_c = popt[0]
    chi_c_err = np.sqrt(np.diag(pcov))[0]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_scaled, accuracies, yerr=std_devs, fmt='o', label='Data')
    
    x_smooth = np.linspace(x_scaled.min(), x_scaled.max(), 200)
    plt.plot(x_smooth, sigmoid(x_smooth, *popt), 'r-', 
             label=f'Fit: χ_c = {chi_c:.2f} ± {chi_c_err:.2f}')
    
    plt.xlabel('χ = m·Δ²')
    plt.ylabel('Accuracy')
    plt.title('High-Precision Transition Analysis (Newsgroups Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'investigation_exp5_precision.png'))
    plt.close()
    
    print(f"  -> χ_c = {chi_c:.4f} ± {chi_c_err:.4f}")
    
    return {'chi_c': chi_c, 'chi_c_err': chi_c_err}

def run_suite():
    print("="*80)
    print(" UNIFIED INVESTIGATION SUITE - NEWSGROUPS DATA")
    print("="*80)
    
    # Load main dataset once for experiments 1 and 5
    print("\nLoading main dataset...")
    lab_main = REWALabNewsgroups(N=2000, max_vocab_size=10000, seed=42)
    lab_main.load_newsgroups_data(subset='train')
    
    all_results = {}
    
    # Run all experiments
    all_results['exp1'] = experiment_1_K_dependence(lab_main)
    all_results['exp2'] = experiment_2_N_scaling()
    all_results['exp3'] = experiment_3_vocab_size()
    all_results['exp4'] = experiment_4_category_subsets()
    all_results['exp5'] = experiment_5_precision(lab_main)
    
    # Save all results
    # Convert numpy types to native python types for JSON serialization
    def convert(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    results_path = os.path.join(OUTPUT_DIR, 'investigation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
        
    print("\n" + "="*80)
    print(" INVESTIGATION COMPLETE")
    print("="*80)
    print(f"Generated 6 figures and results JSON in '{OUTPUT_DIR}/' folder")
    print("\nGenerated files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {fname}")

if __name__ == "__main__":
    run_suite()
