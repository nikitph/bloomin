import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hsgbdh.dual_model import DualStateHSGBDH
from hsgbdh.stable import stable_transitive_closure

class DualBenchmark:
    def __init__(self, n=64, d=64):
        self.n = n
        self.d = d
        self.model = DualStateHSGBDH(n, d)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, length=3, epochs=150):
        print(f"Training Dual Model on L={length}...")
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
                
                # CRITICAL: We evaluate G_transition ONLY.
                G_trans = self.model.last_G_transition[0]
                
                # Compute Closure on pure transition graph
                G_star = stable_transitive_closure(G_trans, max_hops=L+2, temperature=1.0, decay=0.9)
                
                # Score
                start_vec = node_embeddings[0].unsqueeze(0)
                end_vec = node_embeddings[L-1].unsqueeze(0)
                start_idx = F.relu(F.layer_norm(start_vec @ self.model.E, (self.n,))).argmax().item()
                end_idx = F.relu(F.layer_norm(end_vec @ self.model.E, (self.n,))).argmax().item()
                
                score = G_star[start_idx, end_idx].item()
                results.append((L, score))
        return results

def run_dual_experiment():
    bench = DualBenchmark()
    bench.train(length=3, epochs=1000) # Extended training for deeper generalization
    results = bench.evaluate()
    print("\n--- Experiment 3c: Dual State Results ---")
    print(results)
    
    # Plot
    lengths, scores = zip(*results)
    plt.figure(figsize=(10,6))
    plt.plot(lengths, scores, marker='o', color='green', label='3c: Dual State (G_transition)')
    plt.title('Experiment 3c: Explicit Transition Logic')
    plt.xlabel('Length')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.savefig('dual_experiment.png')
    print("Plot saved to dual_experiment.png")

if __name__ == "__main__":
    run_dual_experiment()
