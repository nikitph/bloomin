import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time

# --- Configuration & Style ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

class REWALab:
    """
    A laboratory for simulating Semantic Physics using the REWA framework.
    Generates synthetic witness data with controlled overlap and performs hashing.
    """
    
    def __init__(self, N=2000, W_universe=10000, L=100, seed=42):
        self.N = N                # Number of items
        self.W_universe = W_universe # Size of witness universe
        self.L = L                # Witnesses per item
        self.rng = np.random.default_rng(seed)
        
    def generate_clustered_data(self, k_clusters=20, rho=0.1):
        """
        Generates N items grouped into k clusters.
        rho: Overlap fraction (Signal Strength). 
             Items in cluster share rho*L witnesses.
        """
        self.items = []
        self.cluster_labels = []
        
        items_per_cluster = self.N // k_clusters
        shared_count = int(rho * self.L)
        
        # 1. Create Cluster Prototypes (Shared Witnesses)
        prototypes = []
        all_indices = np.arange(self.W_universe)
        
        for _ in range(k_clusters):
            proto = self.rng.choice(all_indices, size=shared_count, replace=False)
            prototypes.append(proto)
            
        # 2. Generate Items
        for c_idx in range(k_clusters):
            proto = prototypes[c_idx]
            # Remaining witnesses must not overlap with prototype to ensure rho is exact
            # For simplicity in synthetic gen, we pick from global pool excluding prototype
            # (In high dim W_universe >> L, random collisions are negligible)
            
            for _ in range(items_per_cluster):
                unique_needed = self.L - shared_count
                # Simple approximation: random pick from universe
                unique_part = self.rng.choice(all_indices, size=unique_needed, replace=False)
                
                # Combine
                item_witnesses = np.concatenate([proto, unique_part])
                self.items.append(item_witnesses)
                self.cluster_labels.append(c_idx)
                
        self.items = np.array(self.items, dtype=object)
        self.cluster_labels = np.array(self.cluster_labels)
        return self

    def encode(self, m, seed=None):
        """
        REWA Encoding: Projects witnesses into binary vector of size m.
        Uses a simple randomized hash projection (Bloom-like).
        """
        if seed is None: seed = 42
        
        # We simulate k independent hash functions by generating a large permutation table
        # For REWA, simple random projection suffices for demonstration
        
        # Pre-generate a mapping: witness_id -> [h1, h2...]
        # Here we use 2 hash functions (K=2) for robustness
        K_hashes = 2
        hash_map = self.rng.integers(0, m, size=(self.W_universe, K_hashes))
        
        encoded_matrix = np.zeros((self.N, m), dtype=np.int8)
        
        for i, witnesses in enumerate(self.items):
            # Vectorized hashing
            # Ensure witnesses are integers
            w_indices = np.array(witnesses, dtype=int)
            indices = hash_map[w_indices].flatten()
            encoded_matrix[i, indices] = 1
            
        return encoded_matrix

    def evaluate_retrieval(self, encoded_matrix):
        """
        Computes Top-1 Retrieval Accuracy.
        For every item, find nearest neighbor (max dot product).
        Success if NN is in same cluster.
        """
        # Compute Similarity Matrix (Dot Product for Binary REWA)
        # S[i, j] = bit overlap
        # Using float to prevent overflow during matmul, though values are small ints
        Sim = np.dot(encoded_matrix, encoded_matrix.T).astype(float)
        
        # Mask self-similarity
        np.fill_diagonal(Sim, -np.inf)
        
        # Find Nearest Neighbors
        nearest_indices = np.argmax(Sim, axis=1)
        
        # Check against ground truth
        pred_clusters = self.cluster_labels[nearest_indices]
        accuracy = np.mean(pred_clusters == self.cluster_labels)
        
        return accuracy

# ==============================================================================
# EXPERIMENT A: The Phase Transition (Scaling Collapse)
# ==============================================================================
print("--- Running Experiment A: Phase Transitions ---")

rhos = [0.05, 0.10, 0.20] # Different "Signal Strengths" (Gaps)
m_values = np.geomspace(16, 2048, num=15, dtype=int)
results_A = {rho: [] for rho in rhos}

lab = REWALab(N=1000, W_universe=5000, L=100)

for rho in rhos:
    print(f"  Generating universe with overlap rho={rho}...")
    lab.generate_clustered_data(k_clusters=20, rho=rho)
    
    for m in m_values:
        enc = lab.encode(m=m)
        acc = lab.evaluate_retrieval(enc)
        results_A[rho].append(acc)

# --- Plotting Experiment A ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Raw m
for rho in rhos:
    ax1.plot(m_values, results_A[rho], 'o-', label=f'Gap $\\rho={rho}$')

ax1.set_xscale('log')
ax1.set_xlabel('Number of Bits (m)')
ax1.set_ylabel('Top-1 Retrieval Accuracy')
ax1.set_title('Raw Phase Transitions\n(Different Critical Points)')
ax1.legend()
ax1.grid(True, which="both", ls="-")

# Subplot 2: Scaling Collapse (The Physics Proof)
# Theory: Critical m scales with 1/rho^2.
# Therefore, plotting against m * rho^2 should align curves.
for rho in rhos:
    # x-axis transformation: Semantic Capacity
    scaled_x = m_values * (rho**2) 
    ax2.plot(scaled_x, results_A[rho], 'o-', label=f'Gap $\\rho={rho}$')

ax2.set_xscale('log')
ax2.set_xlabel('Thermodynamic Scaling Variable ($m \\cdot \\rho^2$)')
ax2.set_ylabel('Top-1 Retrieval Accuracy')
ax2.set_title('Scaling Collapse\n(Universal Equation of State)')
ax2.legend()
ax2.grid(True, which="both", ls="-")

plt.tight_layout()
plt.savefig('physics_exp_a_scaling_collapse.png')
print("Saved physics_exp_a_scaling_collapse.png")

# ==============================================================================
# EXPERIMENT B: Heat Capacity (Robustness to Noise)
# ==============================================================================
print("\n--- Running Experiment B: Semantic Heat Capacity ---")

m_fixed = 512
noise_levels = np.linspace(0, 0.3, 10) # 0% to 30% bit flips
rhos_B = [0.05, 0.20] # Weak structure vs Strong structure
results_B = {rho: [] for rho in rhos_B}

for rho in rhos_B:
    lab.generate_clustered_data(rho=rho)
    base_enc = lab.encode(m=m_fixed)
    
    for eta in noise_levels:
        # Inject Entropy (Flip bits with probability eta)
        noise_mask = lab.rng.random(base_enc.shape) < eta
        noisy_enc = np.bitwise_xor(base_enc, noise_mask.astype(np.int8))
        
        acc = lab.evaluate_retrieval(noisy_enc)
        results_B[rho].append(acc)

# --- Plotting Experiment B ---
plt.figure(figsize=(8, 6))
for rho in rhos_B:
    plt.plot(noise_levels, results_B[rho], 'o-', linewidth=2, label=f'Gap $\\Delta={rho}$')

plt.xlabel('System Entropy (Noise Level $\\eta$)')
plt.ylabel('Retained Accuracy')
plt.title('Experiment B: Specific Heat of Semantic Index')
plt.legend()
plt.savefig('physics_exp_b_heat_capacity.png')
print("Saved physics_exp_b_heat_capacity.png")

# ==============================================================================
# EXPERIMENT C: Conservation of Information (Noether's Theorem)
# ==============================================================================
print("\n--- Running Experiment C: Conservation Laws ---")

lab.generate_clustered_data(rho=0.1)

# 1. Calculate Total Pairwise Intersection (Proxy for System Energy/Info)
# We do this on raw witnesses (Ground Truth)
def calc_total_overlap(items):
    total_overlap = 0
    # Sample a subset to keep it fast
    subset = items[:100] 
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            intersect = np.intersect1d(subset[i], subset[j]).size
            total_overlap += intersect
    return total_overlap

E_initial = calc_total_overlap(lab.items)

# 2. Apply Reversible Transformation T (Permutation of Universe)
# This simulates a "Change of Basis" or "Rotated Embedding"
perm = lab.rng.permutation(lab.W_universe)
permuted_items = [perm[np.array(w_list, dtype=int)] for w_list in lab.items]

E_final = calc_total_overlap(np.array(permuted_items, dtype=object))

print(f"Initial Semantic Energy: {E_initial}")
print(f"Transformed Energy:      {E_final}")
print(f"Conservation Error:      {abs(E_initial - E_final)}")
assert E_initial == E_final, "Conservation Law Violated!"
print(">> Law I Verified: Information is conserved under reversible transformation.")


# ==============================================================================
# EXPERIMENT D: DPI (Irreversibility)
# ==============================================================================
print("\n--- Running Experiment D: The Second Law (DPI) ---")

# Compare "Lossless" (Set Intersection) vs Lossy Hashing
lab.generate_clustered_data(rho=0.15)

# 1. Ground Truth (Lossless)
# Accuracy using exact Jaccard similarity
# (Simulated by using huge m for REWA, or just logic)
# Here we just treat m=10,000 as near-lossless
acc_lossless = lab.evaluate_retrieval(lab.encode(m=10000))

# 2. Lossy Compressions
m_steps = [1024, 256, 64, 16]
acc_lossy = []
for m in m_steps:
    acc_lossy.append(lab.evaluate_retrieval(lab.encode(m=m)))

# Plotting
labels = ['Lossless (Full Sets)'] + [f'Hash m={m}' for m in m_steps]
values = [acc_lossless] + acc_lossy

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color='skyblue')
plt.ylabel('Retrieval Accuracy (Proxy for MI)')
plt.title('Experiment D: DPI (Information Loss)')
plt.grid(axis='y')

# Add text labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

plt.savefig('physics_exp_d_dpi.png')
print("Saved physics_exp_d_dpi.png")
