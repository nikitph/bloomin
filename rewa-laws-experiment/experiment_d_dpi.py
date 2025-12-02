import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_synthetic, rewa_hash_encode

def estimate_mi_encoded(B, query_indices, target_indices):
    """
    Estimates MI between encoded query and target.
    This is tricky without a full distribution model.
    Proxy: We can measure the 'distinguishability' or simply the retrieval accuracy 
    as a proxy for I(E(Wq); E(Wt)).
    
    The prompt suggests: "Measure ranking performance (Top-k) and show monotonic relation"
    So we will use Top-1 Accuracy as the operational proxy for MI.
    """
    hits = 0
    total = 0
    
    # We need to know the ground truth labels to check accuracy
    # We'll pass them in or restructure.
    # Let's just return the accuracy computed inside the main loop.
    return 0

def run_experiment_d():
    print("Running Experiment D: DPI / Compression...")
    
    N = 2000
    k = 10
    W = 10000
    L = 100
    rho = 0.05
    
    items, labels = make_synthetic(N, k, W, L, rho, seed=42)
    
    configs = [
        ("Lossless (Full Sets)", "lossless"),
        ("Hash m=1024", 1024),
        ("Hash m=256", 256),
        ("Hash m=64", 64),
        ("Hash m=16", 16)
    ]
    
    accuracies = []
    
    num_queries = 200
    query_indices = np.random.choice(N, num_queries, replace=False)
    
    for name, param in configs:
        print(f"  Testing {name}...")
        
        hits = 0
        
        if param == "lossless":
            # Direct set overlap
            for q_idx in query_indices:
                q_set = set(items[q_idx])
                best_score = -1
                best_idx = -1
                
                # Scan all (slow but exact)
                for i in range(N):
                    if i == q_idx: continue
                    score = len(q_set & set(items[i]))
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if labels[best_idx] == labels[q_idx]:
                    hits += 1
                    
        else:
            # Hash Encoding
            m = param
            B = rewa_hash_encode(items, m, seed=42)
            
            for q_idx in query_indices:
                q_vec = B[q_idx]
                scores = np.dot(B, q_vec).astype(float)
                scores[q_idx] = -np.inf
                best_idx = np.argmax(scores)
                
                if labels[best_idx] == labels[q_idx]:
                    hits += 1
        
        acc = hits / num_queries
        accuracies.append(acc)
        print(f"    Acc: {acc:.3f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    x_labels = [c[0] for c in configs]
    plt.bar(x_labels, accuracies, color='skyblue')
    plt.ylabel('Retrieval Accuracy (Proxy for MI)')
    plt.title('Experiment D: DPI (Information Loss)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y')
    
    output_path = 'experiment_d_dpi.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment_d()
