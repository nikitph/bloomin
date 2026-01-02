import torch
import torch.nn as nn
import torch.nn.functional as F
from .semiring import AdaptiveSemiring, ExponentialSchedule

class StructuredLogicHead(nn.Module):
    def __init__(self, d_model, num_logic_types=3):
        super().__init__()
        self.d_model = d_model
        # W_logic matrices for different relation types
        self.W_logic = nn.ParameterList([
            nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
            for _ in range(num_logic_types)
        ])
    
    def forward(self, x, y):
        # x, y: (batch, d)
        consistencies = []
        for W in self.W_logic:
            pred_y = x @ W
            sim = F.cosine_similarity(pred_y, y, dim=-1)
            consistencies.append(sim)
        return torch.stack(consistencies, dim=-1)

class MinimalHSGBDH(nn.Module):
    """
    Single-level proof of concept using Dense Differentiable Graph.
    """
    def __init__(self, n, d, k_proposals=16, max_hops=5):
        super().__init__()
        self.n = n
        self.d = d
        self.max_hops = max_hops
        
        # Core BDH components
        self.E = nn.Parameter(torch.randn(d, n) * 0.1) # Increased init variance
        self.Dx = nn.Parameter(torch.randn(n, d) * 0.1)
        self.Dy = nn.Parameter(torch.randn(n, d) * 0.1)
        
        # Logic verification
        self.logic = StructuredLogicHead(d, num_logic_types=1) # Simplified to 1 type for now
        
        # Semiring
        self.semiring = AdaptiveSemiring(
            ExponentialSchedule(1.0, 0.01, steps=1000)
        )
        
    def forward(self, x_seq, targets=None, query_nodes=None):
        batch_size, seq_len, d = x_seq.shape
        outputs = []
        
        # Initialize dense graph state (Batch, N, N)
        G = torch.eye(self.n, device=x_seq.device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.01
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # (batch, d)
            
            # 1. BDH encoding
            # Use tanh or relu? ReLU as per prompt. Gated.
            v_t = F.layer_norm(x_t @ self.E, normalized_shape=(self.n,))
            x_neurons = F.relu(v_t)  # (batch, n)
            
            # 2. Reasoning (Propagate x_neurons through G)
            # Hop 1
            prop1 = torch.bmm(x_neurons.unsqueeze(1), G).squeeze(1)
            # Hop 2
            prop2 = torch.bmm(prop1.unsqueeze(1), G).squeeze(1)
            # Hop 3
            prop3 = torch.bmm(prop2.unsqueeze(1), G).squeeze(1)
            
            # Combined evidence
            reasoning_out = torch.max(prop1, torch.max(prop2, prop3))
            
            # 3. Output
            y_t = F.relu(reasoning_out @ self.Dy)
            outputs.append(y_t)
            
            # 4. Update Graph (Differentiable)
            # Use target if available (Teacher Forcing), else use y_t
            if targets is not None and t < seq_len - 1:
                # Use next token from targets as positive signal
                target_next = targets[:, t+1, :] # (B, d)
                # Map target to neurons
                tv_t = F.layer_norm(target_next @ self.E, normalized_shape=(self.n,))
                target_neurons = F.relu(tv_t)
                update_signal = target_neurons
                
                # Update G
                update_matrix = torch.bmm(x_neurons.unsqueeze(2), update_signal.unsqueeze(1)) # (B, n, n)
                G = torch.max(G, update_matrix * 0.5)
            elif targets is None:
                # Use prediction y_t as signal
                # Project y_t (B, d) back to neuron space (B, n)
                tv_t = F.layer_norm(y_t @ self.E, normalized_shape=(self.n,))
                pred_neurons = F.relu(tv_t)
                update_signal = pred_neurons
                
                # Update G
                update_matrix = torch.bmm(x_neurons.unsqueeze(2), update_signal.unsqueeze(1)) # (B, n, n)
                G = torch.max(G, update_matrix * 0.5)
        
        # Store G for inspection if needed
        self.last_G = G
        return torch.stack(outputs, dim=1)

    def test_reachability(self, start_idx, end_idx, length, node_embeddings):
        """
        Manually build graph with teacher forcing sequence, then query.
        """
        self.eval()
        with torch.no_grad():
            # Create sequence 0..length
            seq_indices = list(range(length))
            x_seq = torch.stack([node_embeddings[i] for i in seq_indices]).unsqueeze(0) # (1, L, d)
            
            # Run with teacher forcing (self as target)
            self.forward(x_seq, targets=x_seq)
            
            # Check G reachability
            G = self.last_G[0] # (n, n)
            
            # Closure
            G_star = G.clone()
            for _ in range(self.max_hops):
                G_star = torch.max(G_star, G_star @ G)
                
            # Map start/end to neurons
            start_vec = node_embeddings[start_idx].unsqueeze(0)
            end_vec = node_embeddings[end_idx].unsqueeze(0)
            
            start_neuron = F.relu(F.layer_norm(start_vec @ self.E, (self.n,))).argmax().item()
            end_neuron = F.relu(F.layer_norm(end_vec @ self.E, (self.n,))).argmax().item()
            
            strength = G_star[start_neuron, end_neuron].item()
            return strength
