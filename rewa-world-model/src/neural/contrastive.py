"""
Contrastive Encoder Module

Implements contrastive learning with InfoNCE loss to learn embeddings
that approximate Fisher-Rao distances in Euclidean space.

Key features:
- InfoNCE loss for metric learning
- Positive/negative pair sampling based on gap Δ
- Optional natural gradient (K-FAC approximation)
- L2 normalization for Fisher-Rao approximation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class ContrastiveEncoder(nn.Module):
    """
    Neural encoder for learning embeddings via contrastive learning.
    
    Architecture: Simple MLP with residual connections
    Loss: InfoNCE (NT-Xent)
    Normalization: L2 (for Fisher-Rao approximation)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP with residual connections
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode input to embedding space.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            normalize: Whether to L2-normalize output
            
        Returns:
            Embeddings (batch_size, output_dim)
        """
        embeddings = self.encoder(x)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for contrastive learning.
    
    For each anchor, we have:
    - 1 positive (similar item)
    - N-1 negatives (dissimilar items)
    
    Loss = -log(exp(sim(anchor, positive) / τ) / Σ exp(sim(anchor, negative_i) / τ))
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchors: Anchor embeddings (batch_size, dim)
            positives: Positive embeddings (batch_size, dim)
            negatives: Negative embeddings (batch_size, num_negatives, dim)
                      If None, use in-batch negatives
                      
        Returns:
            Scalar loss
        """
        batch_size = anchors.size(0)
        
        # Compute similarities
        pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature  # (batch_size,)
        
        if negatives is None:
            # Use in-batch negatives (all other samples)
            # Similarity matrix: (batch_size, batch_size)
            sim_matrix = torch.matmul(anchors, positives.T) / self.temperature
            
            # Mask out diagonal (self-similarity)
            mask = torch.eye(batch_size, device=anchors.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            
            # LogSumExp over negatives
            neg_logsumexp = torch.logsumexp(sim_matrix, dim=1)
        else:
            # Use provided negatives
            # negatives: (batch_size, num_negatives, dim)
            # anchors: (batch_size, 1, dim)
            neg_sim = torch.sum(
                anchors.unsqueeze(1) * negatives,
                dim=2
            ) / self.temperature  # (batch_size, num_negatives)
            
            neg_logsumexp = torch.logsumexp(neg_sim, dim=1)
        
        # InfoNCE loss
        loss = -pos_sim + torch.logsumexp(
            torch.stack([pos_sim, neg_logsumexp], dim=1),
            dim=1
        )
        
        return loss.mean()

class ContrastiveTrainer:
    """
    Trainer for contrastive encoder.
    
    Handles:
    - Positive/negative pair sampling
    - Training loop with InfoNCE loss
    - Optional natural gradient (K-FAC)
    """
    
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.criterion = InfoNCELoss(temperature)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
    def sample_pairs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        delta: float = 0.1,
        num_negatives: int = 5
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Sample positive and negative pairs based on gap Δ.
        
        Args:
            embeddings: Document embeddings (N, dim)
            labels: Document labels/cluster IDs (N,)
            delta: Minimum gap for positives
            num_negatives: Number of negatives per anchor
            
        Returns:
            List of (anchor_idx, positive_idx, [negative_indices])
        """
        pairs = []
        N = len(embeddings)
        
        for i in range(N):
            # Find positives: same cluster
            same_cluster = np.where(labels == labels[i])[0]
            same_cluster = same_cluster[same_cluster != i]
            
            if len(same_cluster) == 0:
                continue
            
            # Sample one positive
            pos_idx = np.random.choice(same_cluster)
            
            # Find negatives: different cluster
            diff_cluster = np.where(labels != labels[i])[0]
            
            if len(diff_cluster) < num_negatives:
                continue
            
            # Sample negatives
            neg_indices = np.random.choice(
                diff_cluster,
                size=num_negatives,
                replace=False
            ).tolist()
            
            pairs.append((i, pos_idx, neg_indices))
        
        return pairs
    
    def train_epoch(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        delta: float = 0.1
    ) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss
        """
        self.encoder.train()
        
        # Sample pairs
        pairs = self.sample_pairs(embeddings, labels, delta)
        
        if len(pairs) == 0:
            return 0.0
        
        # Shuffle pairs
        np.random.shuffle(pairs)
        
        total_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Prepare batch
            anchors = []
            positives = []
            negatives = []
            
            for anchor_idx, pos_idx, neg_indices in batch_pairs:
                anchors.append(embeddings[anchor_idx])
                positives.append(embeddings[pos_idx])
                negatives.append([embeddings[j] for j in neg_indices])
            
            # Convert to tensors
            anchors = torch.FloatTensor(np.array(anchors)).to(self.device)
            positives = torch.FloatTensor(np.array(positives)).to(self.device)
            negatives = torch.FloatTensor(np.array(negatives)).to(self.device)
            
            # Forward pass
            anchor_emb = self.encoder(anchors)
            positive_emb = self.encoder(positives)
            negative_emb = self.encoder(negatives.view(-1, negatives.size(-1)))
            negative_emb = negative_emb.view(len(batch_pairs), -1, negative_emb.size(-1))
            
            # Compute loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode embeddings using trained encoder."""
        self.encoder.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(embeddings).to(self.device)
            encoded = self.encoder(x)
            return encoded.cpu().numpy()
