import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import BlockSparseGraph
from .stable import stable_transitive_closure

class DualStateHSGBDH(nn.Module):
    """
    Separate state for 'what persists' vs 'what transitions'.
    Experiment 3c.
    """
    def __init__(self, n, d, k_proposals=5):
        super().__init__()
        self.n = n
        self.d = d
        
        # Core BDH
        self.E = nn.Parameter(torch.randn(d, n) * 0.1)
        self.Dx = nn.Parameter(torch.randn(n, d) * 0.1)
        self.Dy = nn.Parameter(torch.randn(n, d) * 0.1)
        
        # Dual Graphs
        # We simulate them as dense tensors for differentiability in this PoC
        # Initialize G_persist to Identity (strong persistence bias)
        self.init_persist = 0.5
        self.init_transition = 0.01
        
        self.mix_param = nn.Parameter(torch.tensor(0.0)) # Logit for Alpha
        
    def forward(self, x_seq, targets=None):
        batch_size, seq_len, d = x_seq.shape
        outputs = []
        
        # Init Graphs
        G_persist = torch.eye(self.n, device=x_seq.device).unsqueeze(0).repeat(batch_size, 1, 1) * self.init_persist
        G_transition = torch.zeros(self.n, self.n, device=x_seq.device).unsqueeze(0).repeat(batch_size, 1, 1) * self.init_transition
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            # Encode
            v_t = F.layer_norm(x_t @ self.E, normalized_shape=(self.n,))
            x_neurons = F.relu(v_t)
            
            # Reasoning
            # Persist: x @ G_persist
            # Transition: x @ G_transition
            
            # G matmul x
            # x_neurons: (B, n)
            
            # Persist reasoning
            r_persist = torch.bmm(x_neurons.unsqueeze(1), G_persist).squeeze(1)
            
            # Transition reasoning (multi-hop? Or just 1-hop for basic flow?)
            # User snippet implies just `G @ x`.
            # Let's do 1-hop for step-by-step unrolling.
            r_transition = torch.bmm(x_neurons.unsqueeze(1), G_transition).squeeze(1)
            
            # Mix
            alpha = torch.sigmoid(self.mix_param)
            # Output state
            reasoning_out = alpha * r_persist + (1 - alpha) * r_transition
            
            # Decode
            y_t = F.relu(reasoning_out @ self.Dy)
            outputs.append(y_t)
            
            # Update Graphs
            # We need targets for robust learning (Teacher Forcing)
            if targets is not None and t < seq_len - 1:
                target_next = targets[:, t+1, :]
                tv_t = F.layer_norm(target_next @ self.E, normalized_shape=(self.n,))
                target_neurons = F.relu(tv_t)
                
                # Update Transition: x_t -> target_next
                # Hebbian: x * target
                update_trans = torch.bmm(x_neurons.unsqueeze(2), target_neurons.unsqueeze(1))
                G_transition = torch.max(G_transition, update_trans * 0.5)
                
                # Update Persistence: x_t -> x_t (reinforce self)
                # Or x_t -> x_t?
                # Persistence means "I stay".
                # So update with x * x?
                update_persist = torch.bmm(x_neurons.unsqueeze(2), x_neurons.unsqueeze(1))
                G_persist = torch.max(G_persist, update_persist * 0.5)
                
        self.last_G_transition = G_transition
        self.last_G_persist = G_persist
        
        return torch.stack(outputs, dim=1)
