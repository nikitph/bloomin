import torch
import torch.nn as nn
import torch.nn.functional as F
from .semiring import AdaptiveSemiring

class BlockSparseGraph(nn.Module):
    """
    Simulates block-sparse graph operations.
    In a real optimized version, this would use custom CUDA kernels or 
    torch.sparse_csr_tensor with block-level sparsity patterns.
    
    For now, we use a dictionary of dense blocks for flexibility and correctness.
    """
    def __init__(self, n, block_size=32, device='cpu'):
        super().__init__()
        self.n = n
        self.block_size = block_size
        self.device = device
        self.n_blocks = (n + block_size - 1) // block_size
        
        # Dictionary mapping (block_i, block_j) -> Tensor[block_size, block_size]
        # We manually register parameters in a ModuleDict or ParameterDict isn't quite right 
        # because the number of blocks changes dynamically. 
        # For a differentiable "simulation", we can store them in a standard dict
        # but manage gradients carefully.
        # Alternatively, for the *initial* version, we can just use a masked dense tensor
        # if N is small (< 2000), or a true list of parameters if we want to be fancy.
        
        # Let's stick to the dictionary approach, but wrapped in a way that PyTorch sees.
        # Ideally, we'd use a sparse tensor, but PyTorch sparse support for custom gradients is tricky.
        # So for this PROTOTYPE, we will use a dense tensor MASKED to simulate sparsity,
        # or just the dictionary if we can afford the overhead.
        
        # User requested: "Force edges to exist in dense 32x32 blocks."
        self.blocks = nn.ParameterDict()
        
    def _get_block_key(self, bi, bj):
        return f"b_{bi}_{bj}"

    def add_edge(self, i, j, weight):
        """
        Adds an edge (semiring update). 
        NOTE: In a fully differentiable setting, 'weight' usually comes from a tensor
        tracking gradients. In-place modification of leaf variables is bad.
        We assume this is called during the 'forward' pass update logic.
        """
        block_i = i // self.block_size
        block_j = j // self.block_size
        key = self._get_block_key(block_i, block_j)
        
        if key not in self.blocks:
            # Initialize block
            self.blocks[key] = nn.Parameter(torch.zeros(self.block_size, self.block_size, device=self.device))
            
        local_i = i % self.block_size
        local_j = j % self.block_size
        
        # Differential update? 
        # If weight has grad, we can't just assign. 
        # But `add_edge` is usually an explicit discrete step in the user's logic.
        # For the differentiable closure, we operate on the *matrix*. 
        
        # Assuming we are building the graph state:
        with torch.no_grad():
            weight_tensor = torch.tensor(weight, device=self.device)
            self.blocks[key][local_i, local_j] = torch.max(self.blocks[key][local_i, local_j], weight_tensor)

    def to_dense(self):
        """Reconstruct full dense matrix for expensive operations"""
        full = torch.zeros(self.n, self.n, device=self.device)
        for key, block in self.blocks.items():
            # Parse key "b_0_1"
            parts = key.split('_')
            bi, bj = int(parts[1]), int(parts[2])
            
            i_start = bi * self.block_size
            i_end = min(i_start + self.block_size, self.n)
            j_start = bj * self.block_size
            j_end = min(j_start + self.block_size, self.n)
            
            # Handle edge cases where block might be clipped at boundaries
            h = i_end - i_start
            w = j_end - j_start
            
            full[i_start:i_end, j_start:j_end] = block[:h, :w]
        return full

class DifferentiableGraphUpdate:
    def __init__(self, n, sparsity_weight=0.01):
        self.sparsity_weight = sparsity_weight
        
    def update_edges(self, G, new_weights):
        """
        G: Current graph weights (n x n)
        new_weights: Proposed edge weights (n x n)
        """
        # Soft update (differentiable)
        G_updated = torch.max(G, new_weights)
        
        # Sparsity regularization
        sparsity_loss = (
            self.sparsity_weight * G_updated.abs().sum() +
            -self.sparsity_weight * (
                G_updated * torch.log(G_updated + 1e-8) +
                (1 - G_updated) * torch.log(1 - G_updated + 1e-8)
            ).sum()
        )
        
        return G_updated, sparsity_loss

def differentiable_closure(G, semiring, K=5):
    """
    Approximate G* = I + G + G^2 ... + G^K
    """
    n = G.size(0)
    G_star = torch.eye(n, device=G.device)
    G_power = G.clone()
    
    for k in range(1, K+1):
        # G_star = G_star (+) G_power
        # stack and semiring_max
        G_star = semiring.semiring_max(torch.stack([G_star, G_power]), dim=0)
        
        # G_power_next = G_power (x) G
        # Used semiring matmul
        G_power = semiring.semiring_matmul(G_power.unsqueeze(0), G.unsqueeze(0)).squeeze(0)
        
    return G_star
