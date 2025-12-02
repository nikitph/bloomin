import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_synthetic, rewa_hash_encode

def run_experiment_a2():
    print("Running Experiment A-2: Scaling Collapse...")
    
    # Parameters
    N = 2000 
    k = 10
    W = 10000
    L = 100
    rhos = [0.02, 0.05, 0.1]
    # We need a dense sweep of m to see the curve shape clearly
    ms = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    trials = 5
    
    results = {} # rho -> {m -> accuracy}
    
    for rho in rhos:
        print(f"  Testing rho={rho}...")
        accuracies = []
        
        for m in ms:
            trial_accs = []
            for t in range(trials):
                items, labels = make_synthetic(N, k, W, L, rho, seed=t)
                B = rewa_hash_encode(items, m, seed=t)
                
                num_queries = 200
                query_indices = np.random.choice(N, num_queries, replace=False)
                
                hits = 0
                for q_idx in query_indices:
                    q_vec = B[q_idx]
                    scores = np.dot(B, q_vec).astype(float)
                    scores[q_idx] = -np.inf
                    top1_idx = np.argmax(scores)
                    if labels[top1_idx] == labels[q_idx]:
                        hits += 1
                
                trial_accs.append(hits / num_queries)
            
            avg_acc = np.mean(trial_accs)
            accuracies.append(avg_acc)
            # print(f"    m={m}, Acc={avg_acc:.3f}")
            
        results[rho] = accuracies

    # Plotting
    plt.figure(figsize=(10, 6))
    
    for rho, accs in results.items():
        Delta = rho * L
        # Normalized x-axis: x = m * Delta^2 / log(N)
        # Note: log base? Theory usually natural log or base 2. 
        # Let's use natural log (np.log) as standard in physics/info theory scaling, 
        # or base 2 if bits. The prompt says "log N". Let's try natural log first.
        # Actually, capacity C ~ Delta^2. m_c ~ 1/C * log N.
        # So m * C / log N ~ constant.
        # x = m * Delta^2 / log(N)
        
        x_values = [m * (Delta**2) / np.log(N) for m in ms]
        
        plt.plot(x_values, accs, marker='o', linestyle='-', alpha=0.7, label=f'rho={rho}')
    
    plt.xlabel(r'Normalized Parameter $x = m \Delta^2 / \log N$')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Experiment A-2: Scaling Collapse (Universality)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_path = 'experiment_a2_scaling_collapse.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment_a2()
