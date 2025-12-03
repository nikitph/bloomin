"""
PERFECT VALIDATION V2: Stronger Signals & Higher Capacity

This script proves the mathematical exactness of χ = m·Δ² under stronger signal conditions.

Changes from V1:
1. Stronger signals: rho = [0.3, 0.5, 0.7]
2. More hash functions: K = 4 (better utilization of bits)
3. Extended range: m up to 8192
4. More witnesses: L = 100
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
import json

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
        """
        Generate synthetic data with EXACT overlap control.
        
        Returns actual measured gap (not assumed).
        """
        items_per_cluster = self.N // k_clusters
        shared_count = int(rho * self.L)
        unique_count = self.L - shared_count
        
        self.items = []
        self.cluster_labels = []
        
        # Pre-allocate witness IDs to avoid collisions
        # Make sure we don't exceed W_universe
        required_witnesses = k_clusters * shared_count + self.N * unique_count
        if required_witnesses > self.W_universe:
            # Auto-expand universe if needed
            self.W_universe = int(required_witnesses * 1.2)
            # raise ValueError(f"Need {required_witnesses} witnesses but W_universe={self.W_universe}")
        
        witness_pool = np.arange(self.W_universe)
        self.rng.shuffle(witness_pool)
        witness_id = 0
        
        cluster_prototypes = []
        
        for cluster_idx in range(k_clusters):
            # Cluster-specific shared witnesses
            cluster_shared = witness_pool[witness_id:witness_id + shared_count]
            witness_id += shared_count
            cluster_prototypes.append(cluster_shared)
            
            for _ in range(items_per_cluster):
                # Item-specific unique witnesses
                unique_witnesses = witness_pool[witness_id:witness_id + unique_count]
                witness_id += unique_count
                
                # Combine: shared + unique
                item_witnesses = np.concatenate([cluster_shared, unique_witnesses])
                self.items.append(set(item_witnesses))
                self.cluster_labels.append(cluster_idx)
        
        self.cluster_labels = np.array(self.cluster_labels)
        return self
    
    def measure_gap(self, n_samples=200):
        """
        Measure ACTUAL gap Δ = Δ_same - Δ_diff
        
        This is critical: we don't assume Δ = ρ, we measure it!
        """
        same_cluster_overlaps = []
        diff_cluster_overlaps = []
        
        # Sample pairs to measure overlap
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
        
        return {
            'Delta_gap': Delta_gap,
            'Delta_same': Delta_same,
            'Delta_diff': Delta_diff,
            'std_same': np.std(same_cluster_overlaps),
            'std_diff': np.std(diff_cluster_overlaps)
        }
    
    def encode(self, m, K_hashes=2, seed=42):
        """REWA encoding with K hash functions"""
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
        """Top-1 accuracy"""
        Sim = np.dot(encoded_matrix, encoded_matrix.T).astype(float)
        np.fill_diagonal(Sim, -np.inf)
        
        nearest_indices = np.argmax(Sim, axis=1)
        pred_clusters = self.cluster_labels[nearest_indices]
        accuracy = np.mean(pred_clusters == self.cluster_labels)
        
        return accuracy


def sigmoid(x, x_c, a, y_min, y_max):
    """Sigmoid for phase transition fitting"""
    return y_min + (y_max - y_min) / (1 + np.exp(-a * (x - x_c)))


def run_perfect_validation_v2():
    """
    Main experiment V2: Stronger signals and more hash functions
    """
    print("="*80)
    print(" PERFECT VALIDATION V2: Stronger Signals & Higher Capacity")
    print("="*80)
    
    # --- UPDATED PARAMETERS ---
    rho_values = [0.3, 0.5, 0.7]  # Stronger signals
    m_values = np.geomspace(16, 8192, num=12, dtype=int) # Extended range
    n_trials = 10  # More trials
    K_hashes = 4   # More hash functions
    L = 100        # Witnesses
    
    N = 2000
    k_clusters = 20
    W_universe = 300000  # Increased for safety
    
    print(f"\nParameters:")
    print(f"  N = {N} concepts")
    print(f"  k = {k_clusters} clusters")
    print(f"  L = {L} witnesses per concept")
    print(f"  K = {K_hashes} hash functions")
    print(f"  ρ values: {rho_values}")
    print(f"  m values: {m_values}")
    print(f"  Trials per config: {n_trials}")
    
    # Storage
    results = {
        'rho_values': rho_values,
        'm_values': m_values.tolist(),
        'measured_deltas': {},
        'accuracy_mean': {},
        'accuracy_std': {},
        'accuracy_raw': {}  # All trials
    }
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    for rho in rho_values:
        print(f"\nρ = {rho}")
        print("-" * 40)
        
        # Generate data once per rho
        lab = REWALabEnhanced(N=N, W_universe=W_universe, L=L, seed=42)
        lab.generate_clustered_data(k_clusters=k_clusters, rho=rho)
        
        # Measure actual gap
        gap_info = lab.measure_gap(n_samples=500)
        Delta_measured = gap_info['Delta_gap']
        print(f"  Measured Δ = {Delta_measured:.4f} (expected ρ={rho})")
        print(f"    Δ_same = {gap_info['Delta_same']:.4f} ± {gap_info['std_same']:.4f}")
        print(f"    Δ_diff = {gap_info['Delta_diff']:.4f} ± {gap_info['std_diff']:.4f}")
        
        results['measured_deltas'][rho] = gap_info
        results['accuracy_mean'][rho] = []
        results['accuracy_std'][rho] = []
        results['accuracy_raw'][rho] = []
        
        # Vary m
        for m in m_values:
            accuracies_trials = []
            
            for trial in range(n_trials):
                # Pass K_hashes here
                enc = lab.encode(m=m, K_hashes=K_hashes, seed=42+trial)
                acc = lab.evaluate_retrieval(enc)
                accuracies_trials.append(acc)
            
            mean_acc = np.mean(accuracies_trials)
            std_acc = np.std(accuracies_trials)
            
            results['accuracy_mean'][rho].append(mean_acc)
            results['accuracy_std'][rho].append(std_acc)
            results['accuracy_raw'][rho].append(accuracies_trials)
            
            print(f"    m={m:4d}: {mean_acc:.3f} ± {std_acc:.3f}")
    
    # ========================================================================
    # ANALYSIS & VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS: Scaling Collapse & Theory Validation")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_values)))
    
    # ====== Panel 1: Raw Phase Transitions ======
    ax1 = plt.subplot(2, 3, 1)
    
    for i, rho in enumerate(rho_values):
        ax1.errorbar(m_values, results['accuracy_mean'][rho], 
                    yerr=results['accuracy_std'][rho],
                    fmt='o-', color=colors[i], label=f'ρ={rho}',
                    capsize=3, alpha=0.8)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Code Length m (bits)', fontweight='bold')
    ax1.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax1.set_title('A. Raw Phase Transitions\n(Stronger Signals)', 
                 fontweight='bold', loc='left')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # ====== Panel 2: Scaling Collapse (THE KEY RESULT) ======
    ax2 = plt.subplot(2, 3, 2)
    
    all_x_scaled = []
    all_y = []
    all_weights = []
    
    for i, rho in enumerate(rho_values):
        Delta_measured = results['measured_deltas'][rho]['Delta_gap']
        
        # Scaling variable: χ = m · Δ²
        x_scaled = m_values * (Delta_measured ** 2)
        y_vals = np.array(results['accuracy_mean'][rho])
        y_err = np.array(results['accuracy_std'][rho])
        
        ax2.errorbar(x_scaled, y_vals, yerr=y_err,
                    fmt='o-', color=colors[i], 
                    label=f'ρ={rho} (Δ={Delta_measured:.3f})',
                    capsize=3, alpha=0.8)
        
        # Collect for master curve fitting
        all_x_scaled.extend(x_scaled)
        all_y.extend(y_vals)
        all_weights.extend(1 / (y_err**2 + 1e-6))
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Thermodynamic Variable χ = m·Δ²', fontweight='bold')
    ax2.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax2.set_title('B. Scaling Collapse\n(Universal Master Curve)', 
                 fontweight='bold', loc='left')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # ====== Panel 3: Master Curve Fit ======
    ax3 = plt.subplot(2, 3, 3)
    
    # Fit universal sigmoid
    all_x_scaled = np.array(all_x_scaled)
    all_y = np.array(all_y)
    all_weights = np.array(all_weights)
    
    sort_idx = np.argsort(all_x_scaled)
    x_fit = all_x_scaled[sort_idx]
    y_fit = all_y[sort_idx]
    w_fit = all_weights[sort_idx]
    
    # Fit
    try:
        popt, pcov = curve_fit(sigmoid, x_fit, y_fit, sigma=1/np.sqrt(w_fit),
                              p0=[np.median(x_fit), 0.05, 0.0, 1.0],
                              bounds=([0, 0, 0, 0.8], [np.inf, 1, 0.2, 1.0]),
                              maxfev=10000)
        
        chi_c_observed = popt[0]
        
        # Plot
        ax3.scatter(all_x_scaled, all_y, c='gray', s=30, alpha=0.4, label='All data')
        
        x_smooth = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 300)
        y_smooth = sigmoid(x_smooth, *popt)
        ax3.plot(x_smooth, y_smooth, 'r-', linewidth=3, 
                label=f'Master curve fit\nχ_c = {chi_c_observed:.1f}')
        
        # Mark critical point
        y_c = sigmoid(chi_c_observed, *popt)
        ax3.plot(chi_c_observed, y_c, 'r*', markersize=20,
                markeredgecolor='black', markeredgewidth=1, zorder=10)
        
        # Compute R²
        ss_res = np.sum((y_fit - sigmoid(x_fit, *popt))**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        ax3.text(0.05, 0.95, f'R² = {r_squared:.5f}',
                transform=ax3.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=12, fontweight='bold')
        
        print(f"\nMaster Curve Fit:")
        print(f"  χ_c (observed) = {chi_c_observed:.2f}")
        print(f"  R² = {r_squared:.6f}")
        print(f"  Transition width a = {popt[1]:.4f}")
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        chi_c_observed = None
        r_squared = None
    
    ax3.set_xscale('log')
    ax3.set_xlabel('χ = m·Δ²', fontweight='bold')
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('C. Master Curve\n(High Signal Regime)', 
                 fontweight='bold', loc='left')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # ====== Panel 4: Theory vs Observation ======
    ax4 = plt.subplot(2, 3, 4)
    
    # Theoretical prediction: χ_c = C_M · log(N)
    # Note: K_hashes affects the capacity constant C_M
    # For K=4, the efficiency might be different.
    C_M_base = 8  
    chi_c_theory = C_M_base * np.log(N)
    
    print(f"\nTheory vs Observation:")
    print(f"  χ_c (theory base) = {chi_c_theory:.2f}")
    
    if chi_c_observed is not None:
        ratio = chi_c_observed / chi_c_theory
        print(f"  χ_c (observed) = {chi_c_observed:.2f}")
        print(f"  Ratio = {ratio:.3f}")
        
        # Bar plot
        categories = ['Theory\n(Base)', 'Observed\n(Fit)']
        values = [chi_c_theory, chi_c_observed]
        bars = ax4.bar(categories, values, color=['steelblue', 'coral'], 
                      edgecolor='black', linewidth=1.5)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add ratio
        ax4.text(0.5, 0.95, f'Observed / Theory = {ratio:.2f}×',
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('Critical Point χ_c', fontweight='bold')
    ax4.set_title('D. Theory Validation\n(K=4 Hashes)', 
                 fontweight='bold', loc='left')
    ax4.grid(axis='y', alpha=0.3)
    
    # ====== Panel 5: Residuals ======
    ax5 = plt.subplot(2, 3, 5)
    
    if chi_c_observed is not None:
        for i, rho in enumerate(rho_values):
            Delta_measured = results['measured_deltas'][rho]['Delta_gap']
            x_scaled = m_values * (Delta_measured ** 2)
            y_vals = np.array(results['accuracy_mean'][rho])
            y_pred = sigmoid(x_scaled, *popt)
            residuals = y_vals - y_pred
            
            ax5.plot(x_scaled, residuals, 'o-', color=colors[i], 
                    label=f'ρ={rho}', alpha=0.8)
        
        ax5.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Compute max residual
        max_residual = np.max(np.abs(y_vals - y_pred))
        ax5.text(0.95, 0.95, f'Max |residual| = {max_residual:.4f}',
                transform=ax5.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax5.set_xscale('log')
    ax5.set_xlabel('χ = m·Δ²', fontweight='bold')
    ax5.set_ylabel('Residual', fontweight='bold')
    ax5.set_title('E. Deviation from Universality', 
                 fontweight='bold', loc='left')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-0.1, 0.1])
    
    # ====== Panel 6: Per-ρ Critical Points ======
    ax6 = plt.subplot(2, 3, 6)
    
    # Extract per-ρ critical points (where acc crosses 0.5)
    critical_m_per_rho = []
    critical_chi_per_rho = []
    
    for rho in rho_values:
        Delta_measured = results['measured_deltas'][rho]['Delta_gap']
        accs = np.array(results['accuracy_mean'][rho])
        
        # Find where crosses 0.5
        if np.any(accs > 0.5) and np.any(accs < 0.5):
            idx = np.where(np.diff(accs > 0.5))[0]
            if len(idx) > 0:
                m_c = m_values[idx[0]]
                chi_c = m_c * (Delta_measured ** 2)
                critical_m_per_rho.append(m_c)
                critical_chi_per_rho.append(chi_c)
            else:
                critical_m_per_rho.append(np.nan)
                critical_chi_per_rho.append(np.nan)
        else:
            critical_m_per_rho.append(np.nan)
            critical_chi_per_rho.append(np.nan)
    
    # Plot
    valid_rhos = [rho for rho, chi in zip(rho_values, critical_chi_per_rho) if not np.isnan(chi)]
    valid_chis = [chi for chi in critical_chi_per_rho if not np.isnan(chi)]
    
    if len(valid_chis) > 0:
        ax6.scatter(valid_rhos, valid_chis, s=100, color='red', 
                   edgecolors='black', linewidth=2, zorder=10)
        
        # Mean observed
        mean_chi = np.mean(valid_chis)
        std_chi = np.std(valid_chis)
        ax6.axhline(mean_chi, color='red', linestyle='-', linewidth=2,
                   label=f'Observed: χ_c = {mean_chi:.1f}±{std_chi:.1f}')
        ax6.fill_between([0, max(rho_values)], mean_chi-std_chi, mean_chi+std_chi,
                        alpha=0.2, color='red')
    
    ax6.set_xlabel('Signal Strength ρ', fontweight='bold')
    ax6.set_ylabel('Critical Point χ_c', fontweight='bold')
    ax6.set_title('F. Universality Check', 
                 fontweight='bold', loc='left')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Semantic Thermodynamics V2: Stronger Signals & K=4 Hashes', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('synthetic_perfect_validation_v2.png', 
                dpi=300, bbox_inches='tight')
    print("\n✓ Saved: synthetic_perfect_validation_v2.png")
    
    # Save results
    results_json = {
        'rho_values': rho_values,
        'm_values': m_values.tolist(),
        'measured_deltas': {str(k): {ik: float(iv) if not isinstance(iv, str) else iv 
                                     for ik, iv in v.items()} 
                           for k, v in results['measured_deltas'].items()},
        'accuracy_mean': {str(k): [float(x) for x in v] for k, v in results['accuracy_mean'].items()},
        'accuracy_std': {str(k): [float(x) for x in v] for k, v in results['accuracy_std'].items()},
        'chi_c_observed': float(chi_c_observed) if chi_c_observed else None,
        'r_squared': float(r_squared) if r_squared else None,
        'N': N,
        'k_clusters': k_clusters,
        'L': L,
        'K_hashes': K_hashes
    }
    
    with open('synthetic_results_v2.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("✓ Saved: synthetic_results_v2.json")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print(" SUMMARY V2: Stronger Signals Validation")
    print("="*80)
    
    if r_squared is not None:
        print(f"\n✓ Scaling Collapse: R² = {r_squared:.6f}")
    
    if chi_c_observed is not None:
        print(f"\n✓ Critical Point: χ_c = {chi_c_observed:.2f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_perfect_validation_v2()
