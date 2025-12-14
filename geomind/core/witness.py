import torch
import torch.nn as nn

class WitnessExtractor(nn.Module):
    """
    Extracts the 'witnesses' of a vector: the dimensions with the largest magnitude.
    This converts a continuous vector into a discrete set of semantic features.
    """
    def __init__(self, dim, num_witnesses=64):
        super().__init__()
        self.dim = dim
        self.num_witnesses = num_witnesses

    def forward(self, x):
        """
        Args:
            x: (..., dim)
        Returns:
            indices: (..., num_witnesses) - Indices of top-k dimensions
            signs: (..., num_witnesses) - Signs of those dimensions (+1 or -1)
        """
        # Get top-k by absolute value
        abs_x = x.abs()
        topk_values, indices = torch.topk(abs_x, self.num_witnesses, dim=-1)
        
        # Get signs of those elements
        # gather signs from original x using the indices
        signs = torch.gather(x.sign(), -1, indices)
        
        # We return magnitude (topk_values) as well for weighted overlap
        return indices, signs, topk_values

class WitnessOverlap(nn.Module):
    """
    Computes overlap between witness sets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, indices_q, indices_k):
        """
        Compute pairwise overlap between query witnesses and key witnesses.
        
        Args:
            indices_q: (batch, seq_q, num_witnesses)
            indices_k: (batch, seq_k, num_witnesses)
            
        Returns:
            overlap: (batch, seq_q, seq_k) - Number of shared witnesses
        """
        # Naive exact overlap computation (can be slow for large batch/seq)
        # We want |W_q ∩ W_k|
        
        # Expand for broadcasting:
        # q: (batch, seq_q, 1, num_witnesses)
        # k: (batch, 1, seq_k, num_witnesses)
        
        q = indices_q.unsqueeze(2) # (B, Sq, 1, W)
        k = indices_k.unsqueeze(1) # (B, 1, Sk, W)
        
        # Currently doing exact match of INDICES only (ignoring signs for now, or match both?)
        # Proposal said "Similarity = discrete witness overlap".
        # Assuming just index overlap for now.
        
        # To count intersection:
        # For each pair (i, j), count how many indices match.
        # q_idx: (B, Sq, 1, W, 1)
        # k_idx: (B, 1, Sk, 1, W)
        # This is O(Sq * Sk * W^2) if done naively via broadcasting equality.
        
        # Better approach:
        # Use one-hot encoding or scatter add if W << dim?
        # Or sorting?
        
        # Since W (num_witnesses) is small (e.g. 64), O(W^2) per pair is 4096 ops.
        # seq len might be large.
        
        # Let's try the broadcasting approach for the prototype.
        # Equality check:
        # q: (B, Sq, 1, W, 1)
        # k: (B, 1, Sk, 1, W)
        # eq: (B, Sq, Sk, W, W) -> Boolean
        # sum over last two dims -> (B, Sq, Sk)
        
        B, Sq, W = indices_q.shape
        Sk = indices_k.shape[1]
        
        q_exp = indices_q.unsqueeze(2).unsqueeze(-1) # (B, Sq, 1, W, 1)
        k_exp = indices_k.unsqueeze(1).unsqueeze(-2) # (B, 1, Sk, 1, W)
        
        matches = (q_exp == k_exp) # (B, Sq, Sk, W, W)
        overlap = matches.float().sum(dim=(-1, -2)) # (B, Sq, Sk)
        
        return overlap

        return overlap

    def compute_weighted_overlap(self, indices_q, values_q, indices_k, values_k):
        """
        Compute pairwise overlap weighted by the magnitude of the witnesses.
        Score = sum_{matches} |q_i| * |k_j| (where indices match)
        
        Args:
            indices_q: (B, Sq, W)
            values_q: (B, Sq, W) - Absolute values/Magnitudes
            indices_k: (B, Sk, W)
            values_k: (B, Sk, W)
        """
        # Optimized implementation to avoid O(B*Sq*Sk*W*W) memory explosion.
        # We loop over W dimensions.
        
    def compute_weighted_overlap(self, indices_q, values_q, indices_k, values_k, dim=None):
        """
        Compute pairwise overlap weighted by the magnitude of the witnesses.
        Uses Scatter + Matmul approach (efficient for d < 10k).
        
        Args:
            indices_q: (B, Sq, W)
            values_q: (B, Sq, W)
            indices_k: (B, Sk, W)
            values_k: (B, Sk, W)
            dim: Dimension of the dense space. Required.
        """
        B, Sq, W = indices_q.shape
        Sk = indices_k.shape[1]
        
        if dim is None:
            # Fallback or error? Assuming dim is passed or inferred
            # Infer max dim from indices (unsafe if batch doesn't cover all dims)
            dim = max(indices_q.max().item(), indices_k.max().item()) + 1
            
        # 1. Create dense vectors
        # q_dense: (B, Sq, dim)
        q_dense = torch.zeros(B, Sq, dim, device=indices_q.device, dtype=values_q.dtype)
        
        # Scatter values. We use scatter_add in case multiple witnesses map to same dim (unlikely with topk but possible generically)
        q_dense.scatter_add_(2, indices_q, values_q)
        
        # k_dense: (B, Sk, dim)
        k_dense = torch.zeros(B, Sk, dim, device=indices_k.device, dtype=values_k.dtype)
        k_dense.scatter_add_(2, indices_k, values_k)
        
        # 2. Matmul
        # (B, Sq, dim) @ (B, dim, Sk) -> (B, Sq, Sk)
        score = torch.matmul(q_dense, k_dense.transpose(-2, -1))
        
        return score

