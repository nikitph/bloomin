import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphCoarsening(nn.Module):
    def __init__(self, n_k, n_kplus1, d):
        super().__init__()
        self.n_k = n_k
        self.n_kplus1 = n_kplus1
        self.d = d
        
        # Learnable assignment matrix
        # Maps n_k nodes to n_kplus1 clusters
        self.assign = nn.Parameter(torch.randn(n_k, n_kplus1))
        
    def forward(self, G_k, features_k):
        """
        G_k: Sparse graph at level k (n_k x n_k) ? 
             Actually for now using dense or BlockSparse that can convert to dense.
             Let's accept either Token features or Graph features.
        features_k: Node features (n_k x d) (or activation "y")
        
        Returns:
        G_kplus1: Coarsened graph (n_kplus1 x n_kplus1)
        features_kplus1: Pooled features (n_kplus1 x d)
        """
        # Soft assignment: which nodes cluster together?
        # S: n_k x n_kplus1
        S = F.softmax(self.assign, dim=1) 
        
        # Pool features: weighted average
        # dimensions: (n_kplus1, n_k) @ (n_k, d) -> (n_kplus1, d)
        features_kplus1 = S.T @ features_k
        
        # Coarsen graph: S^T G S
        # G_k might be BlockSparse.
        if hasattr(G_k, 'to_dense'):
            G_dense = G_k.to_dense()
        else:
            G_dense = G_k
            
        # dimensions: (n_kplus1, n_k) @ (n_k, n_k) @ (n_k, n_kplus1) -> (n_kplus1, n_kplus1)
        G_kplus1 = S.T @ G_dense @ S
        
        return G_kplus1, features_kplus1

    def unpool(self, features_kplus1):
        """
        Map from coarse level back to fine level.
        features_kplus1: (B, n_kplus1) or (n_kplus1, d)
        """
        S = F.softmax(self.assign, dim=1) # n_k x n_kplus1
        
        # If input is (B, n_kplus1), we want (B, n_k).
        # We assume features are typically activations here.
        if features_kplus1.ndim == 2 and features_kplus1.shape[0] == 1:
            # (1, n_kplus1) @ S.T -> (1, n_k)
            return features_kplus1 @ S.T
        elif features_kplus1.ndim == 1:
             return S @ features_kplus1
        else:
            # Fallback for (n_kplus1, d) features
            return S @ features_kplus1
