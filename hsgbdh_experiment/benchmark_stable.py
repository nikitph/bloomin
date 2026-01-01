import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from hsgbdh.minimal import MinimalHSGBDH
from hsgbdh.stable import stable_transitive_closure

class StableChainReasoningBenchmark:
    def __init__(self, n=64, d=64):
        self.n = n
        self.d = d
        self.train_length = 3
        # Expanded test lengths as requested
        self.test_lengths = [3, 4, 5, 7, 9, 12, 16, 20, 25]
        
        self.model = MinimalHSGBDH(n, d, k_proposals=5, max_hops=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def generate_chain_task(self, length):
        edges = [(i, i+1) for i in range(length-1)]
        return edges, (0, length-1)

    def train_on_length(self, length, epochs=50):
        print(f"Training on chains of length {length}...")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            chain_nodes = torch.randperm(self.n)[:length]
            
            x_seq = torch.zeros(1, length, self.d)
            # Embedding logic (mock)
            node_embeddings = torch.eye(self.n, self.d)
            
            for t in range(length):
                node_idx = chain_nodes[t]
                x_seq[0, t] = node_embeddings[node_idx]
            
            outputs = self.model(x_seq, targets=x_seq)
            
            loss = 0
            for t in range(length - 1):
                target_emb = x_seq[0, t+1]
                pred_emb = outputs[0, t]
                loss += (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=0))
            
            if length > 1:
                loss.backward()
                self.optimizer.step()

    def evaluate_with_stable_closure(self):
        results = {
            'length': [],
            'reachability': [],
            'confidence': [],
            'inference_type': []
        }
        
        # Consistent Embeddings
        torch.manual_seed(42)
        node_embeddings = torch.eye(self.n, self.d) # Use Orthogonal for clean test
        
        for L in self.test_lengths:
            print(f"Evaluating Length {L}...")
            # 1. Generate Task
            if L > self.n: continue
            
            # 2. Build Graph (Run Forward)
            self.model.eval()
            with torch.no_grad():
                seq_indices = list(range(L))
                x_seq = torch.stack([node_embeddings[i] for i in seq_indices]).unsqueeze(0)
                
                # Teacher forcing build
                self.model(x_seq, targets=x_seq)
                
                # Extract learned graph
                G_raw = self.model.last_G[0] # (n, n)
                print(f"L={L}, G_raw max: {G_raw.max().item()}")
            
            # 3. Compute Stable Closure
            # Use max_hops = L (dynamic) to allow full reachability
            # Temperature=1.0, Decay=0.9
            G_star = stable_transitive_closure(
                G_raw, 
                max_hops=L+2, 
                temperature=1.0, 
                decay=0.9
            )
            
            # 4. Measure Reachability Score
            # Map start/end to neurons. E is (d, n).
            # Start: 0. End: L-1.
            start_vec = node_embeddings[0].unsqueeze(0)
            end_vec = node_embeddings[L-1].unsqueeze(0)
            
            start_neuron = F.relu(F.layer_norm(start_vec @ self.model.E, (self.n,))).argmax().item()
            end_neuron = F.relu(F.layer_norm(end_vec @ self.model.E, (self.n,))).argmax().item()
            
            score = G_star[start_neuron, end_neuron].item()
            
            results['length'].append(L)
            results['reachability'].append(score)
            results['confidence'].append(score) # Already normalized 0-1
        
        return results

def run_stable_experiment():
    benchmark = StableChainReasoningBenchmark(n=64, d=64)
    
    # Train
    benchmark.train_on_length(3, epochs=150)
    
    # Test
    results = benchmark.evaluate_with_stable_closure()
    
    print("\n--- Stable Generalization Results ---")
    for l, conf in zip(results['length'], results['confidence']):
        print(f"Length {l}: Confidence={conf:.4f}")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['length'], results['confidence'], marker='o', linewidth=2, label='Stable HSGBDH')
    plt.axhline(y=0.0, color='gray', linestyle='--')
    plt.ylim(0, 1.1)
    
    plt.title("Stable Length Generalization (Normalized)")
    plt.xlabel("Chain Length")
    plt.ylabel("Confidence Score [0,1]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('stable_generalization.png')
    print("Plot saved to stable_generalization.png")

if __name__ == "__main__":
    run_stable_experiment()
