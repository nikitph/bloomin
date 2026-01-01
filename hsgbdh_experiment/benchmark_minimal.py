import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from hsgbdh.minimal import MinimalHSGBDH
from hsgbdh.graph import BlockSparseGraph

class ChainReasoningBenchmark:
    """
    The ONLY thing that matters: length extrapolation
    """
    def __init__(self, n=64, d=32):
        self.n = n
        self.d = d
        # Train on short chains
        self.train_length = 3
        # Test on long chains
        self.test_lengths = [3, 4, 5, 7, 9, 12, 16] 
        
        self.model = MinimalHSGBDH(n, d, k_proposals=5, max_hops=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def generate_chain_task(self, length):
        """
        Create synthetic reasoning chain:
        0 -> 1 -> 2 -> ... -> length-1
        Returns:
            sequence of nodes (input)
            edges (ground truth for checking)
            start_node, end_node
        """
        # Nodes are indices 0..length-1
        # Edges: (i, i+1)
        edges = [(i, i+1) for i in range(length-1)]
        return edges, (0, length-1)

    def train_on_length(self, length, epochs=50):
        print(f"Training on chains of length {length}...")
        
        # We need to map nodes to embeddings? 
        # MinimalHSGBDH learns embeddings via E, Dx, Dy.
        # We'll fix input embeddings to be one-hots or orthogonal for simplicity,
        # or let them be learned.
        # If learned, we need consistent IDs.
        # Let's say we have global vocabulary of size N=64.
        
        # Creating a specific chain instance
        # Ideally we train on MANY chains of length `length` with different node IDs.
        
        # For this PoC, we train on random chains of length 3 within our 64 nodes.
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Random chain of length 3
            # Pick 'length' unique nodes from n
            chain_nodes = torch.randperm(self.n)[:length]
            
            # Input sequence: node 0, node 1, node 2...
            # We want the model to predict next node? 
            # Or just consume sequence and build graph.
            # "x_seq" inputs.
            # If we feed [A, B, C], we expect graph A->B, B->C.
            
            # Let's feed one-hot vectors for the chain nodes
            x_seq = torch.zeros(1, length, self.d)
            # We need fixed embeddings for nodes to make sense?
            # Or let E handle it.
            # BUT E is random initially.
            # We need a stable input representation for "Node K".
            # Let's use a fixed embedding look-up table for the simulation.
            
            node_embeddings = torch.eye(self.n, self.d) # If d >= n? 
            if self.d < self.n:
                 torch.manual_seed(42)
                 node_embeddings = torch.randn(self.n, self.d)
            
            for t in range(length):
                node_idx = chain_nodes[t]
                x_seq[0, t] = node_embeddings[node_idx]
            
            # Forward with Teacher Forcing (targets = x_seq)
            # Predicting next token from current.
            # forward expects x_seq.
            # We want to predict x_seq[t+1] from x_seq[t].
            # MinimalHSGBDH.forward loops t=0..len.
            # If we pass targets=x_seq. The logic inside:
            # t uses targets[t+1].
            
            outputs = self.model(x_seq, targets=x_seq) # (1, len, d)
            
            loss = 0
            # Predict t+1 from t
            for t in range(length - 1):
                target_emb = x_seq[0, t+1]
                pred_emb = outputs[0, t]
                loss += (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=0))
            
            # Add G regularization?
            # G sparsity?
            # Accessing G logic is internal? 
            # Differentiable G is in computation graph.
            # But we aren't returning it.
            # It's fine for minimal proof.
            
            if length > 1:
                loss.backward()
                self.optimizer.step()
                
    def evaluate_reachability(self, length):
        # Generate task
        edges, (start, end) = self.generate_chain_task(length)
        if length > self.n: return 0.0
        
        # Consistent Embeddings
        torch.manual_seed(42)
        node_embeddings = torch.randn(self.n, self.d)
        
        # Run test
        strength = self.model.test_reachability(0, length-1, length, node_embeddings)
        
        return strength

def run_experiment():
    benchmark = ChainReasoningBenchmark(n=64, d=64) # d=n for easy one-hot
    
    # Train
    benchmark.train_on_length(3, epochs=100)
    
    # Test
    results = {}
    print("\n--- Testing Length Generalization ---")
    
    # Need to calibrate what "Success" score is.
    # We can normalize by score of length 3 (seen).
    base_score = benchmark.evaluate_reachability(3)
    print(f"Base Score (Len 3): {base_score:.4f}")
    
    for l in benchmark.test_lengths:
        score = benchmark.evaluate_reachability(l)
        normalized = score / (base_score + 1e-8)
        results[l] = normalized
        print(f"Length {l}: Raw={score:.4f}, Norm={normalized:.2f}")

    # Plot
    lengths = list(results.keys())
    scores = list(results.values())
    
    plt.figure()
    plt.plot(lengths, scores, marker='o')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Failure Threshold')
    plt.title("HSGBDH Length Generalization")
    plt.xlabel("Chain Length")
    plt.ylabel("Normalized Reachability Score")
    plt.ylim(0, 1.2)
    plt.grid(True)
    plt.savefig('length_generalization.png')
    print("Plot saved to length_generalization.png")

if __name__ == "__main__":
    run_experiment()
