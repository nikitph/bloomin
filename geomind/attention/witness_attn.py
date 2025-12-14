import torch
import torch.nn as nn
import torch.nn.functional as F
from geomind.core.witness import WitnessExtractor, WitnessOverlap

class WitnessAttention(nn.Module):
    """
    Witness-Based Attention.
    Computes attention weights based on discrete witness overlap rather than dot products.
    """
    def __init__(self, dim, num_witnesses=64, tau=1.0):
        super().__init__()
        self.dim = dim
        self.num_witnesses = num_witnesses
        self.tau = tau
        
        self.extractor = WitnessExtractor(dim, num_witnesses)
        self.overlap_computer = WitnessOverlap()
        
        # Learnable projections to match Transformer expressivity
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, seq_q, dim)
            k: (batch, seq_k, dim)
            v: (batch, seq_k, dim_v)
            mask: (batch, seq_q, seq_k) Optional mask
            
        Returns:
            output: (batch, seq_q, dim_v)
            attn_weights: (batch, seq_q, seq_k)
        """
        # 1. Extract witnesses
        # We use signs as well for overlap?
        # The prompt said: "Similarity = discrete witness overlap".
        # Let's use the indices overlap for now.
        
        q_idx, q_sgn, q_val = self.extractor(q)
        k_idx, k_sgn, k_val = self.extractor(k)
        
        # 2. Compute overlap
        # Using weighted overlap to allow gradient flow
        scores = self.overlap_computer.compute_weighted_overlap(q_idx, q_val, k_idx, k_val, dim=self.dim) # (B, Sq, Sk)
        
        # 3. Scale
        scores = scores / self.tau
        
        # 4. Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 5. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. Aggregate
        # Project V first? Standard is V_proj(v)
        v_proj = self.value_proj(v)
        output = torch.matmul(attn_weights, v_proj)
        
        # 7. Output projection
        output = self.output_proj(output)
        
        return output, attn_weights
