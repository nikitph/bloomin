"""
Neural Encoder and Contrastive Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder: DIM_INPUT -> DIM_EMBED
    """
    def __init__(self):
        super().__init__()
        dim_input = CONFIG["DIM_INPUT"]
        dim_embed = CONFIG["DIM_EMBED"]
        
        # 3-layer MLP with ReLU activations
        self.network = nn.Sequential(
            nn.Linear(dim_input, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim_embed)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, DIM_INPUT]
        Returns:
            embeddings: [batch_size, DIM_EMBED] (normalized to unit sphere)
        """
        embeddings = self.network(x)
        # Normalize to unit sphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def contrastive_loss(anchors, positives, temperature=None):
    """
    Standard InfoNCE loss (NT-Xent)
    Theory says this approximates Natural Gradient Descent
    
    Args:
        anchors: [batch_size, dim] - normalized embeddings
        positives: [batch_size, dim] - normalized embeddings (augmented views)
        temperature: float - contrastive temperature
    
    Returns:
        loss: scalar
    """
    if temperature is None:
        temperature = CONFIG["TEMP"]
    
    batch_size = anchors.shape[0]
    
    # Compute similarity matrix
    # [batch_size, batch_size]
    sim_matrix = torch.matmul(anchors, positives.T) / temperature
    
    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=anchors.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


if __name__ == "__main__":
    # Test encoder
    encoder = MLPEncoder()
    x = torch.randn(32, CONFIG["DIM_INPUT"])
    embeddings = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output norm (should be ~1.0): {embeddings.norm(dim=1).mean():.3f}")
    
    # Test contrastive loss
    anchors = torch.randn(32, CONFIG["DIM_EMBED"])
    positives = torch.randn(32, CONFIG["DIM_EMBED"])
    anchors = F.normalize(anchors, p=2, dim=1)
    positives = F.normalize(positives, p=2, dim=1)
    loss = contrastive_loss(anchors, positives)
    print(f"Contrastive loss: {loss.item():.3f}")
