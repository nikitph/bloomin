import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_synthetic, rewa_hash_encode
import os

def run_experiment_a():
    print("Running Experiment A: Phase Transition...")
    
    # Parameters
    N = 2000 # Reduced from 10k for speed in this demo, scale up for full paper
    k = 10
    W = 10000
    L = 100
    rhos = [0.02, 0.05, 0.1]
    ms = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    trials = 5
    
    results = {} # rho -> {m -> accuracy}
    
    for rho in rhos:
        print(f"  Testing rho={rho} (Delta={rho*L})...")
        accuracies = []
        
        for m in ms:
            trial_accs = []
            for t in range(trials):
                # Generate data
                items, labels = make_synthetic(N, k, W, L, rho, seed=t)
                
                # Encode
                B = rewa_hash_encode(items, m, seed=t)
                
                # Test Retrieval (Top-1)
                # We'll use a subset of queries to save time
                num_queries = 200
                query_indices = np.random.choice(N, num_queries, replace=False)
                
                hits = 0
                for q_idx in query_indices:
                    q_vec = B[q_idx]
                    
                    # Compute Hamming similarity (dot product for binary)
                    # scores = B @ q_vec
                    scores = np.dot(B, q_vec).astype(float)
                    
                    # Mask self
                    scores[q_idx] = -np.inf
                    
                    # Top-1
                    top1_idx = np.argmax(scores)
                    
                    # Check if in same cluster
                    if labels[top1_idx] == labels[q_idx]:
                        hits += 1
                
                acc = hits / num_queries
                trial_accs.append(acc)
            
            avg_acc = np.mean(trial_accs)
            accuracies.append(avg_acc)
            print(f"    m={m}, Acc={avg_acc:.3f}")
            
        results[rho] = accuracies

    # Plotting
    plt.figure(figsize=(10, 6))
    for rho, accs in results.items():
        plt.plot(ms, accs, marker='o', label=f'rho={rho} (Delta={int(rho*L)})')
    
    plt.xscale('log', base=2)
    plt.xlabel('Number of Hash Bits (m)')
    plt.ylabel('Top-1 Retrieval Accuracy')
    plt.title('Experiment A: Phase Transition in Retrieval Accuracy')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    output_path = 'experiment_a_phase_transition.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment_a()
