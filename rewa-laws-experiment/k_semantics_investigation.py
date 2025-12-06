"""
K SEMANTICS INVESTIGATION: Resolving the Theoretical Ambiguity

This script runs 4 targeted experiments to clarify how K affects χ_c:
1. OR vs AND aggregation schemes
2. Per-hash mutual information (additivity test)
3. Finite-size correction C_M(K) analysis
4. Correlation stress test

Goal: Determine which theoretical regime we're in and provide defensible claims.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import entropy
import json

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
    """Enhanced REWA Laboratory with multiple aggregation schemes"""
    
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
        
        for cluster_idx in range(k_clusters):
            cluster_shared = witness_pool[witness_id:witness_id + shared_count]
            witness_id += shared_count
            
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
        return Delta_same - Delta_diff
    
    def encode_AND(self, m, K_hashes=2, seed=42):
        """Current scheme: concatenate all K hash outputs (AND-like)"""
        rng_local = np.random.default_rng(seed)
        hash_map = rng_local.integers(0, m, size=(self.W_universe, K_hashes))
        encoded_matrix = np.zeros((self.N, m), dtype=np.int16)
        
        for i, witnesses in enumerate(self.items):
            w_indices = np.array(list(witnesses), dtype=int)
            if len(w_indices) > 0:
                indices = hash_map[w_indices].flatten()
                encoded_matrix[i, indices] = 1
        return encoded_matrix
    
    def encode_OR(self, m, K_hashes=2, seed=42):
        """OR scheme: K independent hash functions, aggregate scores"""
        rng_local = np.random.default_rng(seed)
        
        # Store K separate encodings
        encodings = []
        for k in range(K_hashes):
            hash_map_k = rng_local.integers(0, m, size=self.W_universe)
            encoded_k = np.zeros((self.N, m), dtype=np.int16)
            
            for i, witnesses in enumerate(self.items):
                w_indices = np.array(list(witnesses), dtype=int)
                if len(w_indices) > 0:
                    indices = hash_map_k[w_indices]
                    encoded_k[i, indices] = 1
            encodings.append(encoded_k)
        
        return encodings  # Return list of K encodings
    
    def evaluate_AND(self, encoded_matrix):
        """Standard evaluation for AND scheme"""
        Sim = np.dot(encoded_matrix, encoded_matrix.T).astype(float)
        np.fill_diagonal(Sim, -np.inf)
        nearest_indices = np.argmax(Sim, axis=1)
        pred_clusters = self.cluster_labels[nearest_indices]
        return np.mean(pred_clusters == self.cluster_labels)
    
    def evaluate_OR(self, encodings):
        """OR evaluation: sum similarities across K channels"""
        K = len(encodings)
        Sim_total = np.zeros((self.N, self.N), dtype=float)
        
        for k in range(K):
            Sim_k = np.dot(encodings[k], encodings[k].T).astype(float)
            Sim_total += Sim_k
        
        np.fill_diagonal(Sim_total, -np.inf)
        nearest_indices = np.argmax(Sim_total, axis=1)
        pred_clusters = self.cluster_labels[nearest_indices]
        return np.mean(pred_clusters == self.cluster_labels)

def sigmoid(x, x_c, a, y_min, y_max):
    return y_min + (y_max - y_min) / (1 + np.exp(-a * (x - x_c)))

def fit_critical_point(m_values, accuracies, Delta):
    x_scaled = m_values * (Delta ** 2)
    try:
        popt, _ = curve_fit(sigmoid, x_scaled, accuracies, 
                           p0=[np.median(x_scaled), 0.05, 0.0, 1.0],
                           bounds=([0, 0, 0, 0.8], [np.inf, 1, 0.2, 1.0]),
                           maxfev=5000)
        return popt[0]
    except:
        return None

# ============================================================================
# EXPERIMENT 1: OR vs AND Aggregation Schemes
# ============================================================================
def experiment_1_OR_vs_AND():
    print("\n" + "="*80)
    print(" EXPERIMENT 1: OR vs AND Aggregation Schemes")
    print("="*80)
    
    K_values = [2, 3, 4, 6, 8]
    m_values = np.geomspace(16, 4096, num=15, dtype=int)
    n_trials = 3
    
    results_AND = {'K': [], 'chi_c': []}
    results_OR = {'K': [], 'chi_c': []}
    
    lab = REWALabEnhanced(N=2000, W_universe=300000, L=75, seed=42)
    lab.generate_clustered_data(k_clusters=20, rho=0.4)
    Delta = lab.measure_gap(n_samples=500)
    
    for K in K_values:
        print(f"\nTesting K={K}...")
        
        # AND scheme
        accs_AND = []
        for m in m_values:
            trial_accs = []
            for t in range(n_trials):
                enc = lab.encode_AND(m=m, K_hashes=K, seed=42+t)
                trial_accs.append(lab.evaluate_AND(enc))
            accs_AND.append(np.mean(trial_accs))
        
        chi_c_AND = fit_critical_point(m_values, accs_AND, Delta)
        if chi_c_AND:
            results_AND['K'].append(K)
            results_AND['chi_c'].append(chi_c_AND)
            print(f"  AND: χ_c = {chi_c_AND:.2f}")
        
        # OR scheme
        accs_OR = []
        for m in m_values:
            trial_accs = []
            for t in range(n_trials):
                encs = lab.encode_OR(m=m, K_hashes=K, seed=42+t)
                trial_accs.append(lab.evaluate_OR(encs))
            accs_OR.append(np.mean(trial_accs))
        
        chi_c_OR = fit_critical_point(m_values, accs_OR, Delta)
        if chi_c_OR:
            results_OR['K'].append(K)
            results_OR['chi_c'].append(chi_c_OR)
            print(f"  OR:  χ_c = {chi_c_OR:.2f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: χ_c vs K
    ax1.plot(results_AND['K'], results_AND['chi_c'], 'o-', 
             label='AND (concatenated)', color='blue', linewidth=2)
    ax1.plot(results_OR['K'], results_OR['chi_c'], 's-', 
             label='OR (independent)', color='red', linewidth=2)
    ax1.set_xlabel('Number of Hash Functions K')
    ax1.set_ylabel('Critical Point χ_c')
    ax1.set_title('A. OR vs AND: Critical Point Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: m_c vs K (derived from chi_c)
    # chi_c = m_c * Delta^2  =>  m_c = chi_c / Delta^2
    m_c_AND = np.array(results_AND['chi_c']) / (Delta ** 2)
    m_c_OR = np.array(results_OR['chi_c']) / (Delta ** 2)
    
    ax2.plot(results_AND['K'], m_c_AND, 'o-', 
             label='AND: m_c grows with K', color='blue', linewidth=2)
    ax2.plot(results_OR['K'], m_c_OR, 's-', 
             label='OR: m_c ∝ 1/K?', color='red', linewidth=2)
    ax2.set_xlabel('Number of Hash Functions K')
    ax2.set_ylabel('Critical Code Length m_c')
    ax2.set_title('B. Code Length Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_semantics_exp1_OR_vs_AND.png', dpi=300)
    print("\n✓ Saved: k_semantics_exp1_OR_vs_AND.png")
    
    return {'AND': results_AND, 'OR': results_OR, 'Delta': Delta}

# ============================================================================
# EXPERIMENT 2: Per-Hash Mutual Information (Additivity Test)
# ============================================================================
def experiment_2_MI_additivity():
    print("\n" + "="*80)
    print(" EXPERIMENT 2: Per-Hash Mutual Information (Additivity Test)")
    print("="*80)
    
    K_max = 8
    m = 256  # Fixed code length
    n_samples = 500
    
    lab = REWALabEnhanced(N=2000, W_universe=300000, L=75, seed=42)
    lab.generate_clustered_data(k_clusters=20, rho=0.4)
    
    # Split data: train (for encoding) and test (for MI estimation)
    n_test = 500
    test_indices = np.random.choice(lab.N, n_test, replace=False)
    
    results = {'K': [], 'MI_per_hash': [], 'MI_total': []}
    
    for K in range(1, K_max+1):
        print(f"\nTesting K={K}...")
        
        # Encode with K hashes (AND scheme)
        enc = lab.encode_AND(m=m, K_hashes=K, seed=42)
        enc_test = enc[test_indices]
        labels_test = lab.cluster_labels[test_indices]
        
        # Compute MI: I(encoding; cluster_label)
        # Use binning: hash the encoding to discrete bins
        n_bins = min(100, 2**10)  # Limit bins
        
        # Hash each encoding to a bin
        enc_hashes = []
        for i in range(len(enc_test)):
            # Simple hash: sum of bit positions
            enc_hash = hash(tuple(np.where(enc_test[i] > 0)[0])) % n_bins
            enc_hashes.append(enc_hash)
        
        # Compute MI
        from sklearn.metrics import mutual_info_score
        mi_total = mutual_info_score(labels_test, enc_hashes)
        
        # Estimate MI per hash (approximate)
        # For K hashes, if independent: MI_total ≈ K * MI_single
        mi_per_hash = mi_total / K
        
        results['K'].append(K)
        results['MI_per_hash'].append(mi_per_hash)
        results['MI_total'].append(mi_total)
        
        print(f"  MI_total = {mi_total:.3f}, MI_per_hash = {mi_per_hash:.3f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['K'], results['MI_total'], 'o-', color='purple')
    ax1.set_xlabel('Number of Hash Functions K')
    ax1.set_ylabel('Total Mutual Information I(B; Y)')
    ax1.set_title('A. Total MI vs K')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results['K'], results['MI_per_hash'], 's-', color='orange')
    ax2.set_xlabel('Number of Hash Functions K')
    ax2.set_ylabel('MI per Hash I(B; Y) / K')
    ax2.set_title('B. MI per Hash (should be constant if additive)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_semantics_exp2_MI_additivity.png', dpi=300)
    print("\n✓ Saved: k_semantics_exp2_MI_additivity.png")
    
    return results

# ============================================================================
# EXPERIMENT 3: Finite-Size Correction C_M(K)
# ============================================================================
def experiment_3_C_M_analysis():
    print("\n" + "="*80)
    print(" EXPERIMENT 3: Finite-Size Correction C_M(K)")
    print("="*80)
    
    K_values = [2, 3, 4, 6, 8]
    L_values = [50, 75, 100]
    rho = 0.4
    N = 2000
    
    results = {'K': [], 'L': [], 'C_M': []}
    
    for L in L_values:
        print(f"\nTesting L={L}...")
        for K in K_values:
            print(f"  K={K}...")
            
            lab = REWALabEnhanced(N=N, W_universe=300000, L=L, seed=42)
            lab.generate_clustered_data(k_clusters=20, rho=rho)
            Delta = lab.measure_gap(n_samples=500)
            
            m_values = np.geomspace(16, 4096, num=15, dtype=int)
            accs = []
            for m in m_values:
                trial_accs = []
                for t in range(3):
                    enc = lab.encode_AND(m=m, K_hashes=K, seed=42+t)
                    trial_accs.append(lab.evaluate_AND(enc))
                accs.append(np.mean(trial_accs))
            
            chi_c = fit_critical_point(m_values, accs, Delta)
            if chi_c:
                # C_M = chi_c / log(N)
                # But theory says chi_c = C_M * K * L * rho (if additive)
                # Or chi_c = C_M * log(N) (standard)
                # Let's compute both
                C_M_standard = chi_c / np.log(N)
                C_M_KL = chi_c / (K * L * rho)
                
                results['K'].append(K)
                results['L'].append(L)
                results['C_M'].append(C_M_standard)
                
                print(f"    χ_c = {chi_c:.2f}, C_M = {C_M_standard:.2f}")
    
    # Plot C_M vs K for different L
    plt.figure(figsize=(10, 6))
    
    for L in L_values:
        K_L = [results['K'][i] for i in range(len(results['K'])) if results['L'][i] == L]
        C_M_L = [results['C_M'][i] for i in range(len(results['C_M'])) if results['L'][i] == L]
        plt.plot(K_L, C_M_L, 'o-', label=f'L={L}', linewidth=2)
    
    plt.xlabel('Number of Hash Functions K')
    plt.ylabel('C_M = χ_c / log(N)')
    plt.title('Finite-Size Correction: Does C_M converge with K?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('k_semantics_exp3_C_M_analysis.png', dpi=300)
    print("\n✓ Saved: k_semantics_exp3_C_M_analysis.png")
    
    return results

# ============================================================================
# EXPERIMENT 4: Correlation Stress Test
# ============================================================================
def experiment_4_correlation_stress():
    print("\n" + "="*80)
    print(" EXPERIMENT 4: Correlation Stress Test")
    print("="*80)
    
    correlation_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    K = 4
    m_values = np.geomspace(16, 4096, num=15, dtype=int)
    
    results = {'correlation': [], 'chi_c': []}
    
    for corr in correlation_levels:
        print(f"\nTesting correlation={corr}...")
        
        # Generate data with correlated witnesses
        lab = REWALabEnhanced(N=2000, W_universe=300000, L=75, seed=42)
        lab.generate_clustered_data(k_clusters=20, rho=0.4)
        
        # Introduce correlation: replace some witnesses with copies
        if corr > 0:
            for i in range(lab.N):
                witnesses = list(lab.items[i])
                n_corr = int(corr * len(witnesses))
                if n_corr > 0:
                    # Replace n_corr witnesses with duplicates of first witness
                    for j in range(n_corr):
                        witnesses[-(j+1)] = witnesses[0]
                lab.items[i] = set(witnesses)
        
        Delta = lab.measure_gap(n_samples=500)
        
        accs = []
        for m in m_values:
            trial_accs = []
            for t in range(3):
                enc = lab.encode_AND(m=m, K_hashes=K, seed=42+t)
                trial_accs.append(lab.evaluate_AND(enc))
            accs.append(np.mean(trial_accs))
        
        chi_c = fit_critical_point(m_values, accs, Delta)
        if chi_c:
            results['correlation'].append(corr)
            results['chi_c'].append(chi_c)
            print(f"  χ_c = {chi_c:.2f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(results['correlation'], results['chi_c'], 'o-', color='red', linewidth=2)
    plt.xlabel('Witness Correlation Level')
    plt.ylabel('Critical Point χ_c')
    plt.title('Correlation Stress Test: How does correlation affect χ_c?')
    plt.grid(True, alpha=0.3)
    plt.savefig('k_semantics_exp4_correlation.png', dpi=300)
    print("\n✓ Saved: k_semantics_exp4_correlation.png")
    
    return results

def run_suite():
    all_results = {}
    
    print("\n" + "="*80)
    print(" K SEMANTICS INVESTIGATION SUITE")
    print("="*80)
    
    all_results['exp1'] = experiment_1_OR_vs_AND()
    all_results['exp2'] = experiment_2_MI_additivity()
    all_results['exp3'] = experiment_3_C_M_analysis()
    all_results['exp4'] = experiment_4_correlation_stress()
    
    # Save results
    def convert(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    
    with open('k_semantics_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print("\n" + "="*80)
    print(" K SEMANTICS INVESTIGATION COMPLETE")
    print("="*80)
    print("Generated 4 figures and k_semantics_results.json")

if __name__ == "__main__":
    run_suite()
