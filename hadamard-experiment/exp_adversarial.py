import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

def run_adversarial(N=2000, d=1024, m=128):
    print(f"--- Experiment 3: Adversarial Structure ---")
    print(f"{'Delta':<10} | {'RP Confusion':<12} {'WP Confusion':<12}")
    print("-" * 45)
    
    deltas = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    
    for delta in deltas:
        np.random.seed(42)
        X = np.random.randn(N, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        
        # Plant 2 Clusters separated by delta
        # Center 1
        c1 = np.random.randn(d)
        c1 /= np.linalg.norm(c1)
        
        # Center 2: Move delta away from c1
        # Approximate by adding delta * random_ortho
        # For precise control, we just pick another random point and interpolate? 
        # Simpler: c2 = c1 + delta * noise, then normalize.
        # If delta is small, they are close.
        noise_dir = np.random.randn(d)
        noise_dir -= noise_dir.dot(c1) * c1 # Orthogonalize
        noise_dir /= np.linalg.norm(noise_dir)
        
        # We want ||c1 - c2|| = delta approximately
        # c2 = c1 + delta * noise_dir (roughly)
        c2 = c1 + delta * noise_dir
        c2 /= np.linalg.norm(c2)
        
        # Plant 10 points at c1 (Cluster A)
        # Plant 10 points at c2 (Cluster B)
        # Indices 0-9: A, 10-19: B
        sigma = 0.01
        for i in range(10):
            p = c1 + np.random.randn(d) * sigma
            p /= np.linalg.norm(p)
            X[i] = p
            
        for i in range(10):
            p = c2 + np.random.randn(d) * sigma
            p /= np.linalg.norm(p)
            X[10+i] = p
            
        # Query near c1
        Q = (c1 + np.random.randn(d) * sigma).reshape(1, -1)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        
        # Target: 0-9. Adversarial: 10-19.
        target_set = set(range(10))
        adversarial_set = set(range(10, 20))
        
        # RP
        G = np.random.randn(d, m)
        X_rp = X @ G
        Q_rp = Q @ G
        dists_rp = euclidean_distances(Q_rp, X_rp)[0]
        rp_top = np.argsort(dists_rp)[:10]
        rp_confused = len(adversarial_set.intersection(rp_top)) / 10
        
        # WP
        D = np.random.choice([-1, 1], size=d)
        H = hadamard(d)
        X_wp = ((X * D) @ H)[:, :m]
        Q_wp = ((Q * D) @ H)[:, :m]
        dists_wp = euclidean_distances(Q_wp, X_wp)[0]
        wp_top = np.argsort(dists_wp)[:10]
        wp_confused = len(adversarial_set.intersection(wp_top)) / 10
        
        print(f"{delta:<10} | {rp_confused:.2f}         {wp_confused:.2f}")

if __name__ == "__main__":
    run_adversarial()
