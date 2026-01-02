import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hsgbdh.minimal import MinimalHSGBDH # Single State
from hsgbdh.stable import stable_transitive_closure

class CapacityBenchmark:
    def __init__(self, n=64, d=64):
        self.n = n
        self.d = d
        self.model = MinimalHSGBDH(n, d, k_proposals=5, max_hops=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, length=3, epochs=150):
        print(f"Training Single-State Model (n={self.n}, d={self.d}) on L={length}...")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            chain_nodes = torch.randperm(self.n)[:length]
            node_embeddings = torch.eye(self.n, self.d)
            x_seq = torch.zeros(1, length, self.d)
            for t in range(length):
                x_seq[0, t] = node_embeddings[chain_nodes[t]]
            
            outputs = self.model(x_seq, targets=x_seq)
            
            loss = 0
            for t in range(length - 1):
                target_emb = x_seq[0, t+1]
                pred_emb = outputs[0, t]
                loss += (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=0))
            
            if length > 1:
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        test_lengths = [3, 4, 5, 7, 9, 12, 16, 20]
        results = []
        node_embeddings = torch.eye(self.n, self.d)
        
        for L in test_lengths:
            if L > self.n: continue
            
            self.model.eval()
            with torch.no_grad():
                seq_indices = list(range(L))
                x_seq = torch.stack([node_embeddings[i] for i in seq_indices]).unsqueeze(0)
                self.model(x_seq, targets=x_seq)
                
                G_raw = self.model.last_G[0]
                # Standard Stable Closure
                G_star = stable_transitive_closure(G_raw, max_hops=L+2)
                
                # Score
                start_vec = node_embeddings[0].unsqueeze(0)
                end_vec = node_embeddings[L-1].unsqueeze(0)
                # Re-map 
                start_idx = F.relu(F.layer_norm(start_vec @ self.model.E, (self.n,))).argmax().item()
                end_idx = F.relu(F.layer_norm(end_vec @ self.model.E, (self.n,))).argmax().item()
                
                score = G_star[start_idx, end_idx].item()
                results.append((L, score))
                
        return results

def run_capacity_experiment():
    # Baseline Capacity n=64
    bench_64 = CapacityBenchmark(n=64, d=64)
    bench_64.train(length=3, epochs=150)
    res_64 = bench_64.evaluate()
    
    # Double Capacity n=128 (roughly matching parameters of Dual State)
    bench_128 = CapacityBenchmark(n=128, d=64) # Keep embedding dim same, increase graph size
    bench_128.train(length=3, epochs=150)
    res_128 = bench_128.evaluate()
    
    print("\n--- Capacity Ablation Results ---")
    print("Single State n=64:", res_64)
    print("Single State n=128:", res_128)
    
    # Plot
    l64, s64 = zip(*res_64)
    l128, s128 = zip(*res_128)
    
    plt.figure(figsize=(10,6))
    plt.plot(l64, s64, marker='o', label='Single State n=64')
    plt.plot(l128, s128, marker='^', label='Single State n=128 (Double Capacity)')
    plt.title('Does Capacity Solve Signal Loss?')
    plt.xlabel('Length')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.savefig('capacity_ablation.png')
    print("Plot saved to capacity_ablation.png")

if __name__ == "__main__":
    run_capacity_experiment()
