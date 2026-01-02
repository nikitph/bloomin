import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hsgbdh.minimal import MinimalHSGBDH # Single State
from hsgbdh.stable import multi_timescale_closure, stable_transitive_closure

class TimescaleBenchmark:
    def __init__(self, n=64, d=64):
        self.n = n
        self.d = d
        self.model = MinimalHSGBDH(n, d, k_proposals=5, max_hops=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, length=3, epochs=150):
        print(f"Training Single-State Model on L={length}...")
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
        results_std = []
        results_multi = []
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
                G_std = stable_transitive_closure(G_raw, max_hops=L+2, temperature=1.0, decay=0.9)
                
                # Multi-Timescale Closure
                G_multi = multi_timescale_closure(G_raw, max_hops=L+2, temperature=1.0)
                
                # Score
                start_vec = node_embeddings[0].unsqueeze(0)
                end_vec = node_embeddings[L-1].unsqueeze(0)
                start_idx = F.relu(F.layer_norm(start_vec @ self.model.E, (self.n,))).argmax().item()
                end_idx = F.relu(F.layer_norm(end_vec @ self.model.E, (self.n,))).argmax().item()
                
                score_std = G_std[start_idx, end_idx].item()
                score_multi = G_multi[start_idx, end_idx].item()
                
                results_std.append((L, score_std))
                results_multi.append((L, score_multi))
                
        return results_std, results_multi

def run_timescale_experiment():
    bench = TimescaleBenchmark()
    bench.train(length=3, epochs=150)
    res_std, res_multi = bench.evaluate()
    
    print("\n--- Experiment 4 Results ---")
    print("Standard (Fixed 0.9):", res_std)
    print("Multi-Timescale:", res_multi)
    
    # Plot
    l_std, s_std = zip(*res_std)
    l_mul, s_mul = zip(*res_multi)
    
    plt.figure(figsize=(10,6))
    plt.plot(l_std, s_std, marker='x', linestyle='--', label='Standard Stable (0.9)')
    plt.plot(l_mul, s_mul, marker='o', linewidth=2, label='Multi-Timescale')
    plt.title('Impact of Multi-Timescale Decay')
    plt.xlabel('Length')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.savefig('timescale_experiment.png')
    print("Plot saved to timescale_experiment.png")

if __name__ == "__main__":
    run_timescale_experiment()
