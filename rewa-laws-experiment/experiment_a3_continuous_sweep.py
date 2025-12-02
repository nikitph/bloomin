import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_synthetic, rewa_hash_encode
from scipy.interpolate import interp1d

def run_experiment_a3():
    print("Running Experiment A-3: Continuous Rho Sweep...")
    
    # Parameters
    N = 2000
    k = 10
    W = 10000
    L = 100
    
    # Sweep rho from 0.01 to 0.10
    rhos = np.linspace(0.01, 0.10, 10)
    
    # We need to find m_c for each rho.
    # Since m_c varies wildly (huge for small rho, small for large rho),
    # we need a dynamic search range or a very wide log sweep.
    # m_c ~ 1/rho^2. 
    # rho=0.01 -> Delta=1. m_c ~ large.
    # rho=0.1 -> Delta=10. m_c ~ small.
    
    # Let's define a wide range of m
    ms = np.logspace(1, 13, 20, base=2).astype(int) # 2 to 8192
    ms = sorted(list(set(ms))) # unique
    
    trials = 3 # Reduced trials for speed
    target_acc = 0.3
    
    mc_values = []
    inv_delta_sq_values = []
    
    for rho in rhos:
        Delta = rho * L
        print(f"  Testing rho={rho:.2f} (Delta={Delta:.1f})...")
        
        accuracies = []
        for m in ms:
            trial_accs = []
            for t in range(trials):
                items, labels = make_synthetic(N, k, W, L, rho, seed=t)
                B = rewa_hash_encode(items, m, seed=t)
                
                num_queries = 100
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
        
        # Interpolate to find m where acc = 0.3
        # We expect a monotonic increase.
        try:
            f = interp1d(accuracies, ms, kind='linear', fill_value="extrapolate")
            m_c = f(target_acc)
        except:
            m_c = np.nan
            
        print(f"    Found m_c ~ {m_c:.1f}")
        
        if not np.isnan(m_c) and m_c > 0:
            mc_values.append(m_c)
            inv_delta_sq_values.append(1.0 / (Delta**2))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(inv_delta_sq_values, mc_values, 'bo-', label='Measured m_c')
    
    # Fit a line
    if len(mc_values) > 1:
        z = np.polyfit(inv_delta_sq_values, mc_values, 1)
        p = np.poly1d(z)
        plt.plot(inv_delta_sq_values, p(inv_delta_sq_values), 'r--', label=f'Fit: y={z[0]:.1f}x + {z[1]:.1f}')
        
    plt.xlabel(r'$1 / \Delta^2$')
    plt.ylabel(r'Critical Bits $m_c$ (Acc=0.3)')
    plt.title(r'Experiment A-3: Scaling Law $m_c \propto 1/\Delta^2$')
    plt.grid(True)
    plt.legend()
    
    output_path = 'experiment_a3_continuous_sweep.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment_a3()
