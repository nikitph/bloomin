"""
UNIFIED INVESTIGATION SUITE: Probing the Rough Edges of Semantic Thermodynamics

This script runs 5 targeted experiments to investigate:
1. K-Dependence: Does C_M increase with K? (Explaining the mismatch)
2. N-Scaling: Does χ_c scale logarithmically with N? (Core theory validation)
3. Rho-Dependence: Is universality robust across signal strengths?
4. L-Dependence: Does witness count matter?
5. Precision: High-resolution transition analysis.

Runtime: ~3-4 hours total (with n_trials=3)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import time
import os

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

class REWALabEnhanced:
    """Enhanced REWA Laboratory with precise gap measurement"""
    
    def __init__(self, N=2000, W_universe=10000, L=100, seed=42):
        self.N = N
        self.W_universe = W_universe
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.items = None
        self.cluster_labels = None
        
    def generate_clustered_data(self, k_clusters=20, rho=0.1):
        items_per_cluster = self.N // k_clusters
        shared_count = int(rho * self.L)
        unique_count = self.L - shared_count
        
        self.items = []
        self.cluster_labels = []
        
        required_witnesses = k_clusters * shared_count + self.N * unique_count
        if required_witnesses > self.W_universe:
            self.W_universe = int(required_witnesses * 1.2)
        
        witness_pool = np.arange(self.W_universe)
        self.rng.shuffle(witness_pool)
        witness_id = 0
        
        cluster_prototypes = []
        
        for cluster_idx in range(k_clusters):
            cluster_shared = witness_pool[witness_id:witness_id + shared_count]
            witness_id += shared_count
            cluster_prototypes.append(cluster_shared)
            
            for _ in range(items_per_cluster):
                unique_witnesses = witness_pool[witness_id:witness_id + unique_count]
                witness_id += unique_count
                item_witnesses = np.concatenate([cluster_shared, unique_witnesses])
                self.items.append(set(item_witnesses))
                self.cluster_labels.append(cluster_idx)
        
        self.cluster_labels = np.array(self.cluster_labels)
        return self
    
    def measure_gap(self, n_samples=200):
        same_cluster_overlaps = []
        diff_cluster_overlaps = []
        for _ in range(n_samples):
            i, j = self.rng.choice(self.N, size=2, replace=False)
            overlap = len(self.items[i] & self.items[j]) / self.L
            if self.cluster_labels[i] == self.cluster_labels[j]:
                same_cluster_overlaps.append(overlap)
            else:
                diff_cluster_overlaps.append(overlap)
        
        Delta_same = np.mean(same_cluster_overlaps)
        Delta_diff = np.mean(diff_cluster_overlaps)
        Delta_gap = Delta_same - Delta_diff
        return Delta_gap
    
    def encode(self, m, K_hashes=2, seed=42):
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

def run_experiment_config(N, L, K, rho, m_values, n_trials=3):
    """Run a single configuration"""
    lab = REWALabEnhanced(N=N, W_universe=300000, L=L, seed=42)
    lab.generate_clustered_data(k_clusters=20, rho=rho)
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
def experiment_1_K_dependence():
    print("\n" + "="*60)
    print(" EXPERIMENT 1: K-Dependence (The C_M Mystery)")
    print("="*60)
    
    K_values = [2, 3, 4, 6, 8]
    m_values = np.geomspace(16, 4096, num=15, dtype=int)
    results = {'K': [], 'chi_c': []}
    
    plt.figure(figsize=(10, 6))
    
    for K in K_values:
        print(f"Testing K={K}...")
        chi_c, _, accs, Delta = run_experiment_config(N=2000, L=75, K=K, rho=0.4, m_values=m_values)
        if chi_c:
            results['K'].append(K)
            results['chi_c'].append(chi_c)
            plt.plot(m_values * Delta**2, accs, 'o-', label=f'K={K} (χ_c={chi_c:.1f})')
            print(f"  -> χ_c = {chi_c:.2f}")
    
    plt.xscale('log')
    plt.xlabel('χ = m·Δ²')
    plt.ylabel('Accuracy')
    plt.title('Effect of Hash Functions K on Critical Point')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('investigation_exp1_K_dependence.png')
    
    # Plot C_M vs K
    plt.figure(figsize=(8, 5))
    plt.plot(results['K'], results['chi_c'], 's-', color='purple', markersize=8)
    plt.xlabel('Number of Hash Functions K')
    plt.ylabel('Critical Point χ_c')
    plt.title('Does C_M increase with K?')
    plt.grid(True, alpha=0.3)
    plt.savefig('investigation_exp1_chi_vs_K.png')
    
    return results

# ============================================================================
# EXPERIMENT 2: N-Scaling (The Core Theory Test)
# ============================================================================
def experiment_2_N_scaling():
    print("\n" + "="*60)
    print(" EXPERIMENT 2: N-Scaling (The Core Theory Test)")
    print("="*60)
    
    N_values = [500, 1000, 2000, 4000, 8000]
    m_values = np.geomspace(16, 8192, num=15, dtype=int)
    results = {'N': [], 'chi_c': []}
    
    for N in N_values:
        print(f"Testing N={N}...")
        # Adjust clusters to keep items/cluster constant (~100)
        k_clusters = max(5, N // 100)
        
        lab = REWALabEnhanced(N=N, W_universe=300000, L=75, seed=42)
        lab.generate_clustered_data(k_clusters=k_clusters, rho=0.4)
        Delta = lab.measure_gap()
        
        accuracies = []
        for m in m_values:
            trial_accs = []
            for t in range(3):
                enc = lab.encode(m=m, K_hashes=4, seed=42+t)
                trial_accs.append(lab.evaluate_retrieval(enc))
            accuracies.append(np.mean(trial_accs))
            
        chi_c, _ = fit_critical_point(m_values, accuracies, Delta)
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
    plt.title(f'N-Scaling Validation: χ_c ∝ log(N)\nEmpirical C_M = {slope:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('investigation_exp2_N_scaling.png')
    
    return results

# ============================================================================
# EXPERIMENT 3: Rho-Dependence (Universality Check)
# ============================================================================
def experiment_3_rho_fine():
    print("\n" + "="*60)
    print(" EXPERIMENT 3: Rho-Dependence (Universality Check)")
    print("="*60)
    
    rho_values = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    m_values = np.geomspace(16, 8192, num=15, dtype=int)
    results = {'rho': [], 'chi_c': []}
    
    for rho in rho_values:
        print(f"Testing rho={rho}...")
        chi_c, _, _, _ = run_experiment_config(N=2000, L=75, K=4, rho=rho, m_values=m_values)
        if chi_c:
            results['rho'].append(rho)
            results['chi_c'].append(chi_c)
            print(f"  -> χ_c = {chi_c:.2f}")
            
    plt.figure(figsize=(8, 5))
    plt.plot(results['rho'], results['chi_c'], 'o-', color='green')
    plt.xlabel('Signal Strength ρ')
    plt.ylabel('Critical Point χ_c')
    plt.title('Universality Check: Is χ_c independent of ρ?')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.savefig('investigation_exp3_rho_dependence.png')
    
    return results

# ============================================================================
# EXPERIMENT 4: L-Dependence (Witness Count)
# ============================================================================
def experiment_4_L_dependence():
    print("\n" + "="*60)
    print(" EXPERIMENT 4: L-Dependence (Witness Count)")
    print("="*60)
    
    L_values = [25, 50, 75, 100, 150, 200]
    m_values = np.geomspace(16, 8192, num=15, dtype=int)
    results = {'L': [], 'chi_c': []}
    
    for L in L_values:
        print(f"Testing L={L}...")
        chi_c, _, _, _ = run_experiment_config(N=2000, L=L, K=4, rho=0.4, m_values=m_values)
        if chi_c:
            results['L'].append(L)
            results['chi_c'].append(chi_c)
            print(f"  -> χ_c = {chi_c:.2f}")
            
    plt.figure(figsize=(8, 5))
    plt.plot(results['L'], results['chi_c'], 'd-', color='orange')
    plt.xlabel('Witness Count L')
    plt.ylabel('Critical Point χ_c')
    plt.title('Does Witness Count L affect Critical Point?')
    plt.grid(True, alpha=0.3)
    plt.savefig('investigation_exp4_L_dependence.png')
    
    return results

# ============================================================================
# EXPERIMENT 5: Precision (Transition Sharpness)
# ============================================================================
def experiment_5_precision():
    print("\n" + "="*60)
    print(" EXPERIMENT 5: Precision (Transition Sharpness)")
    print("="*60)
    
    # Dense sampling around expected critical point (~24 for K=4, rho=0.4)
    # chi = m * Delta^2  => m = chi / Delta^2
    # Delta approx 0.4 => Delta^2 = 0.16
    # m_c approx 24 / 0.16 = 150
    
    m_dense = np.linspace(50, 400, num=30, dtype=int)
    n_trials_high = 20
    
    print(f"Running dense sweep with {n_trials_high} trials...")
    lab = REWALabEnhanced(N=2000, W_universe=300000, L=75, seed=42)
    lab.generate_clustered_data(k_clusters=20, rho=0.4)
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
    plt.title('High-Precision Transition Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('investigation_exp5_precision.png')
    
    print(f"  -> χ_c = {chi_c:.4f} ± {chi_c_err:.4f}")
    
    return {'chi_c': chi_c, 'chi_c_err': chi_c_err}

def run_suite():
    all_results = {}
    
    # Run all experiments
    all_results['exp1'] = experiment_1_K_dependence()
    all_results['exp2'] = experiment_2_N_scaling()
    all_results['exp3'] = experiment_3_rho_fine()
    all_results['exp4'] = experiment_4_L_dependence()
    all_results['exp5'] = experiment_5_precision()
    
    # Save all results
    # Convert numpy types to native python types for JSON serialization
    def convert(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open('investigation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
        
    print("\n" + "="*80)
    print(" INVESTIGATION COMPLETE")
    print("="*80)
    print("Generated 5 figures and investigation_results.json")

if __name__ == "__main__":
    run_suite()
