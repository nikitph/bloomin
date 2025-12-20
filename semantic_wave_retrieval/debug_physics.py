import torch
import numpy as np
from semantic_wave_retrieval.clustering_engine import ClusteringEngine
from semantic_wave_retrieval.decision_engine import DecisionEngine
from semantic_wave_retrieval.utils import generate_synthetic_data

def debug_clustering():
    print("\n--- Debug Clustering ---")
    # Generate blobs to check alignment
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    data_np, true_labels = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=1.0, random_state=42)
    data = torch.tensor(data_np, dtype=torch.float32)
    
    engine = ClusteringEngine(data, k_neighbors=10)
    
    # Init 
    p = torch.ones(200) / 200
    noise = torch.randn(200) * 0.1
    epsilon = 0.5
    u = torch.tanh((p - p.mean()) / epsilon) + noise
    print(f"Init U: mean={u.mean():.4f}, std={u.std():.4f}")
    
    dt = 0.05
    for t in range(200):
        # delta_u = -L u
        delta_u = -engine.sparse_dense_mul(engine.L, u) if hasattr(engine, 'sparse_dense_mul') else -torch.sparse.mm(engine.L, u.unsqueeze(1)).squeeze()
        
        mu = (u**3 - u) - (epsilon**2) * delta_u
        delta_mu = -torch.sparse.mm(engine.L, mu.unsqueeze(1)).squeeze() # -L mu
        
        u = u + dt * delta_mu
        
        if t % 20 == 0:
            pred_labels = (u.numpy() > 0).astype(int)
            ari = adjusted_rand_score(true_labels, pred_labels)
            print(f"T={t}: U mean={u.mean():.4f}, std={u.std():.4f}, ARI={ari:.4f}")
            
    print(f"Final ARI={adjusted_rand_score(true_labels, (u.numpy()>0).astype(int)):.4f}")

def debug_decision():
    print("\n--- Debug Decision ---")
    # 1D line graph for easy tunneling check
    N = 50
    data = torch.linspace(-5, 5, N).view(-1, 1)
    engine = DecisionEngine(data, k_neighbors=2) # Line connectivity roughly
    
    # Double well potential: x^4 - x^2 (Minima at +/- 1/sqrt(2)?)
    # actually let's do simple: V = 1 if x != target else 0.
    V = torch.ones(N)
    V[25] = 0.0 # Target at center
    V[10] = 0.5 # Local min
    
    psi = torch.ones(N)
    psi /= psi.norm()
    
    print("Running Schrodinger...")
    psi = engine.schrodinger_step(psi, V, T_schrod=50, dt=0.5)
    print(f"Psi Max Loc: {psi.argmax().item()}")
    print(f"Psi at Target(25): {psi[25]:.4f}, at Local(10): {psi[10]:.4f}")

if __name__ == "__main__":
    debug_clustering()
    debug_decision()
