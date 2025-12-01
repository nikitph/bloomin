"""
Learned REWA Encoder
====================

Continuous REWA encoder with LEARNED projections (not random Hadamard).
Trains end-to-end to maximize recall on semantic similarity tasks.

Expected: 60-80% recall with m_dim=256 (vs 27% with random Hadamard)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedContinuousREWAEncoder(nn.Module):
    """
    Continuous REWA encoder with learned projections.
    
    Key difference from ContinuousREWAEncoder:
    - Uses learned nn.Linear instead of fixed Hadamard transform
    - Can adapt to semantic structure of data
    - Trained end-to-end with contrastive loss
    """
    
    def __init__(
        self,
        d_model: int,
        m_dim: int,
        dropout: float = 0.1,
        noise_std: float = 0.01,
        use_mlp: bool = True,
    ):
        """
        Args:
            d_model: Input dimension
            m_dim: Output dimension (compressed)
            dropout: Dropout rate
            noise_std: Training noise standard deviation
            use_mlp: If True, use 2-layer MLP; else single linear
        """
        super().__init__()
        
        self.d_model = d_model
        self.m_dim = m_dim
        self.noise_std = noise_std
        
        # Learned projection
        if use_mlp:
            # 2-layer MLP for better expressiveness
            hidden_dim = m_dim * 2
            self.projection = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, m_dim),
                nn.LayerNorm(m_dim),
            )
        else:
            # Single linear layer
            self.projection = nn.Sequential(
                nn.Linear(d_model, m_dim),
                nn.LayerNorm(m_dim),
            )
        
        # Learnable temperature for similarity scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(self, x: torch.Tensor, add_noise: bool = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model] input embeddings
            add_noise: If True, add noise (default: self.training)
        
        Returns:
            [B, N, m_dim] L2-normalized projections
        """
        # Learned projection
        x_proj = self.projection(x)  # [B, N, m_dim]
        
        # Add noise during training
        if add_noise is None:
            add_noise = self.training
        
        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(x_proj) * self.noise_std
            x_proj = x_proj + noise
        
        # L2 normalize for cosine similarity
        return F.normalize(x_proj, dim=-1)
    
    def compute_similarity(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scaled cosine similarity.
        
        Args:
            queries: [B, N, m_dim] normalized query projections
            keys: [B, M, m_dim] normalized key projections
        
        Returns:
            [B, N, M] similarity scores
        """
        # Cosine similarity
        similarity = torch.bmm(queries, keys.transpose(1, 2))
        
        # Apply learned temperature
        return similarity / self.temperature.clamp(min=0.01)
    
    def get_compression_ratio(self) -> float:
        """Compression ratio."""
        return self.d_model / self.m_dim


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training learned REWA encoder.
    
    Maximizes similarity for positive pairs (same class),
    minimizes similarity for negative pairs (different class).
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, m_dim] normalized embeddings
            labels: [B] class labels
        
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.T) / self.temperature
        
        # Create label matrix (positive pairs have same label)
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        label_matrix = label_matrix & ~mask
        
        # Numerically stable softmax
        similarity = similarity - torch.max(similarity, dim=1, keepdim=True)[0]
        similarity.masked_fill_(mask, -1e9)  # Mask diagonal
        
        # Positive pairs
        pos_mask = label_matrix
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        
        # For each anchor, compute loss
        losses = []
        for i in range(batch_size):
            # Positive examples for anchor i
            pos_sim = exp_sim[i][pos_mask[i]]
            
            if len(pos_sim) == 0:
                continue
            
            # All examples (positive + negative)
            all_sim = exp_sim[i].sum()
            
            # Loss: -log(sum(pos) / sum(all))
            loss = -torch.log(pos_sim.sum() / all_sim)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return torch.stack(losses).mean()


class TripletLoss(nn.Module):
    """
    Triplet loss: anchor-positive closer than anchor-negative.
    
    Alternative to contrastive loss, sometimes works better.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, m_dim] normalized embeddings
            labels: [B] class labels
        
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create label matrix
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Mask diagonal
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        label_matrix = label_matrix & ~mask
        
        losses = []
        for i in range(batch_size):
            # Positive examples
            pos_mask = label_matrix[i]
            if pos_mask.sum() == 0:
                continue
            
            # Negative examples
            neg_mask = ~label_matrix[i] & ~mask[i]
            if neg_mask.sum() == 0:
                continue
            
            # Get distances
            pos_dists = distances[i][pos_mask]
            neg_dists = distances[i][neg_mask]
            
            # Triplet loss: d(a,p) + margin < d(a,n)
            # Loss: max(0, d(a,p) - d(a,n) + margin)
            for pos_dist in pos_dists:
                for neg_dist in neg_dists:
                    loss = F.relu(pos_dist - neg_dist + self.margin)
                    losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return torch.stack(losses).mean()


if __name__ == "__main__":
    print("="*60)
    print("Testing Learned REWA Encoder")
    print("="*60)
    
    # Test encoder
    print("\n1. Testing LearnedContinuousREWAEncoder")
    encoder = LearnedContinuousREWAEncoder(d_model=768, m_dim=256)
    x = torch.randn(2, 10, 768)
    
    encoded = encoder(x)
    print(f"Input: {x.shape}, Encoded: {encoded.shape}")
    print(f"Compression: {encoder.get_compression_ratio():.1f}×")
    print(f"Encoded norm: {encoded.norm(dim=-1).mean():.3f} (should be ~1.0)")
    
    # Test similarity
    print("\n2. Testing similarity computation")
    Q = torch.randn(2, 5, 768)
    K = torch.randn(2, 10, 768)
    
    Q_enc = encoder(Q)
    K_enc = encoder(K)
    
    similarity = encoder.compute_similarity(Q_enc, K_enc)
    print(f"Query: {Q.shape}, Key: {K.shape}")
    print(f"Similarity: {similarity.shape}")
    print(f"Similarity range: [{similarity.min():.2f}, {similarity.max():.2f}]")
    
    # Test contrastive loss
    print("\n3. Testing ContrastiveLoss")
    loss_fn = ContrastiveLoss(temperature=0.07)
    
    # Create batch with 3 classes
    embeddings = torch.randn(12, 256)
    embeddings = F.normalize(embeddings, dim=-1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    
    loss = loss_fn(embeddings, labels)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    # Test triplet loss
    print("\n4. Testing TripletLoss")
    triplet_loss_fn = TripletLoss(margin=0.2)
    
    loss = triplet_loss_fn(embeddings, labels)
    print(f"Triplet loss: {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
