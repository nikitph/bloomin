import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, mode='cosine', temperature=0.1):
        super().__init__()
        self.mode = mode
        self.tau = temperature
        
    def forward(self, features_1, features_2):
        """
        features_1, features_2: [B, D] corresponding positive pairs (views)
        """
        batch_size = features_1.shape[0]
        
        # Concatenate for full pairwise matrix: [2B, D]
        features = torch.cat([features_1, features_2], dim=0)
        
        if self.mode == 'cosine':
            # Inputs MUST be normalized already by model
            # logits = (x . y) / tau
            sim_matrix = torch.matmul(features, features.T)
            logits = sim_matrix / self.tau
            
        elif self.mode == 'euclidean':
            # Inputs are raw/REWA vectors
            # logits = -||x - y||^2 / (2*tau)
            # Efficient L2: ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy
            
            # [2B, 1]
            norms_sq = torch.sum(features**2, dim=1, keepdim=True)
            # [2B, 2B]
            dist_sq = norms_sq + norms_sq.T - 2 * torch.matmul(features, features.T)
            
            # Avoid numerical issues (negative distances)
            dist_sq = torch.clamp(dist_sq, min=0.0)
            
            logits = -dist_sq / (2 * self.tau)
        
        # Labels
        # Logic: 
        # If input is [v1_0, v1_1... | v2_0, v2_1...]
        # Anchor i (0..B-1) is pos with i+B
        # Anchor i+B (B..2B-1) is pos with i
        
        # Create mask for self-contrast (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        logits.masked_fill_(mask, -9e15) # Mask self-similarity
        
        # Ground Truth indices
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size, device=features.device),
            torch.arange(0, batch_size, device=features.device)
        ], dim=0)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
