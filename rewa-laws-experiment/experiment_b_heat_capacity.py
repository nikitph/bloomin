import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_synthetic, rewa_hash_encode

def run_experiment_b():
    print("Running Experiment B: Heat Capacity...")
    
    # Parameters
    N = 2000
    k = 10
    W = 10000
    L = 100
    rhos = [0.02, 0.1] # Low vs High Delta
    m = 512 # Fixed m on the plateau
    etas = np.arange(0, 0.55, 0.05) # Noise levels
    trials = 5
    
    results = {} # rho -> {eta -> accuracy}
    
    for rho in rhos:
        print(f"  Testing rho={rho}...")
        accuracies = []
        
        for eta in etas:
            trial_accs = []
            for t in range(trials):
                items, labels = make_synthetic(N, k, W, L, rho, seed=t)
                B = rewa_hash_encode(items, m, seed=t)
                
                # Add Noise: Flip fraction eta of bits
                # We flip bits in the database B_noisy
                noise_mask = np.random.random(B.shape) < eta
                B_noisy = B.copy()
                # Flip 0->1 and 1->0 where mask is True
                B_noisy[noise_mask] = 1 - B_noisy[noise_mask]
                
                # Query with clean query (or noisy? Prompt says "Add controlled noise to the index")
                # So Index is noisy, Query is clean (or vice versa, usually symmetric for Hamming)
                # Let's assume Query is also from the noisy distribution or clean?
                # "Randomly flip a fraction eta of bits in the encoded database entries"
                # Let's keep queries clean to see retrieval from noisy memory.
                
                num_queries = 200
                query_indices = np.random.choice(N, num_queries, replace=False)
                
                hits = 0
                for q_idx in query_indices:
                    # Use the stored noisy version for retrieval
                    # But what is the query? 
                    # If we are retrieving FROM the noisy index, the query is likely a fresh encoding of the concept.
                    # Let's use the CLEAN encoding for the query, and search in NOISY index.
                    q_vec = B[q_idx] # Clean query
                    
                    # Similarity: Dot product (or Hamming distance)
                    # For bit flips, Hamming distance is better, but Dot product is standard for "activation".
                    # Let's use Dot product to be consistent with Exp A.
                    scores = np.dot(B_noisy, q_vec).astype(float)
                    
                    scores[q_idx] = -np.inf # Mask self (if self is in index)
                    
                    top1_idx = np.argmax(scores)
                    if labels[top1_idx] == labels[q_idx]:
                        hits += 1
                        
                trial_accs.append(hits / num_queries)
                
            avg_acc = np.mean(trial_accs)
            accuracies.append(avg_acc)
            print(f"    eta={eta:.2f}, Acc={avg_acc:.3f}")
            
        results[rho] = accuracies

    # Plotting
    plt.figure(figsize=(10, 6))
    for rho, accs in results.items():
        # Calculate slope at eta=0 (approx)
        slope = (accs[1] - accs[0]) / (etas[1] - etas[0])
        plt.plot(etas, accs, marker='o', label=f'rho={rho} (Slope ~ {slope:.2f})')
    
    plt.xlabel('Noise Fraction (eta)')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Experiment B: Heat Capacity (Robustness to Noise)')
    plt.grid(True)
    plt.legend()
    
    output_path = 'experiment_b_heat_capacity.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment_b()
