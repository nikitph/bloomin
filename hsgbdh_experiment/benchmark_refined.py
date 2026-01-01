import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from hsgbdh.minimal import MinimalHSGBDH
from hsgbdh.stable import stable_transitive_closure, damped_closure

class RefinedBenchmark:
    def __init__(self, n=64, d=64):
        self.n = n
        self.d = d
        self.model = MinimalHSGBDH(n, d, k_proposals=5, max_hops=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
    def train_mixed(self, lengths=[2, 3, 4, 5], epochs=150):
        print(f"Training on mixed lengths: {lengths}...")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Random length
            L = random.choice(lengths)
            chain_nodes = torch.randperm(self.n)[:L]
            
            # Embeddings (Mock)
            node_embeddings = torch.eye(self.n, self.d)
            
            x_seq = torch.zeros(1, L, self.d)
            for t in range(L):
                node_idx = chain_nodes[t]
                x_seq[0, t] = node_embeddings[node_idx]
            
            outputs = self.model(x_seq, targets=x_seq)
            
            loss = 0
            for t in range(L - 1):
                target_emb = x_seq[0, t+1]
                pred_emb = outputs[0, t]
                loss += (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=0))
            
            if L > 1:
                loss.backward()
                self.optimizer.step()

    def evaluate(self, use_damped=False, penalty=0.5):
        test_lengths = [3, 4, 5, 7, 9, 12, 16, 20]
        results = {'length': [], 'score': []}
        
        node_embeddings = torch.eye(self.n, self.d)
        
        for L in test_lengths:
            if L > self.n: continue
            
            self.model.eval()
            with torch.no_grad():
                seq_indices = list(range(L))
                x_seq = torch.stack([node_embeddings[i] for i in seq_indices]).unsqueeze(0)
                self.model(x_seq, targets=x_seq)
                G_raw = self.model.last_G[0]
            
            if use_damped:
                # Experiment 3a
                G_star = damped_closure(G_raw, max_hops=L+2, self_loop_penalty=penalty)
            else:
                # Standard Stable
                G_star = stable_transitive_closure(G_raw, max_hops=L+2)
                
            # Score
            start_neuron = 0 # Since we use Identity embedding for mock
            end_neuron = L-1
            # Wait, MinimialHSGBDH projects embedding to neurons via E.
            # But in `MinimalHSGBDH` E is random.
            # We need to map start/end to neurons correctly.
            
            # Re-calculate neuron indices
            start_vec = node_embeddings[0].unsqueeze(0)
            end_vec = node_embeddings[L-1].unsqueeze(0)
            start_neuron_idx = F.relu(F.layer_norm(start_vec @ self.model.E, (self.n,))).argmax().item()
            end_neuron_idx = F.relu(F.layer_norm(end_vec @ self.model.E, (self.n,))).argmax().item()
            
            score = G_star[start_neuron_idx, end_neuron_idx].item()
            results['length'].append(L)
            results['score'].append(score)
            
        return results

def run_experiments():
    # Experiment 3a: Train L=3, Eval with Damped
    print("\n--- Experiment 3a: Self-Loop Suppression ---")
    bench_a = RefinedBenchmark()
    # Train on L=3 only (replicate resonance condition)
    bench_a.train_mixed(lengths=[3], epochs=150) 
    
    results_a = bench_a.evaluate(use_damped=True, penalty=0.01) # Heavy penalty
    print("Results 3a (Damped):", list(zip(results_a['length'], results_a['score'])))
    
    # Experiment 3b: Train Mixed, Eval Standard
    print("\n--- Experiment 3b: Mixed Length Training ---")
    bench_b = RefinedBenchmark()
    bench_b.train_mixed(lengths=[2, 3, 4, 5], epochs=150)
    
    results_b = bench_b.evaluate(use_damped=False)
    print("Results 3b (Mixed Train, Std Eval):", list(zip(results_b['length'], results_b['score'])))
    
    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(results_a['length'], results_a['score'], label='3a: Damped (Train L=3)')
    plt.plot(results_b['length'], results_b['score'], label='3b: Mixed Train')
    plt.xlabel('Length')
    plt.ylabel('Confidence')
    plt.title('Refined Reasoning Experiments')
    plt.legend()
    plt.grid(True)
    plt.savefig('refined_experiments.png')
    print("Plot saved to refined_experiments.png")

if __name__ == "__main__":
    run_experiments()
