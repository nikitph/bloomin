"""
Ricci Flow implementation for concept stabilization
"""

import torch
import torch.nn.functional as F


def contrastive_loss(anchor, positive, negatives, temperature=0.1):
    """
    Contrastive loss: pull anchor toward positive, push away from negatives
    
    Args:
        anchor: Anchor distribution (n_witnesses,)
        positive: Positive distribution (n_witnesses,)
        negatives: List of negative distributions
        temperature: Temperature parameter
    
    Returns:
        Contrastive loss value
    """
    # KL divergence between anchor and positive (minimize)
    epsilon = 1e-10
    anchor_clipped = torch.clamp(anchor, epsilon, 1.0)
    positive_clipped = torch.clamp(positive, epsilon, 1.0)
    
    kl_positive = torch.sum(anchor_clipped * torch.log(anchor_clipped / positive_clipped))
    
    # KL divergence between anchor and negatives (maximize = minimize negative)
    kl_negatives = []
    for neg in negatives:
        neg_clipped = torch.clamp(neg, epsilon, 1.0)
        kl_neg = torch.sum(anchor_clipped * torch.log(anchor_clipped / neg_clipped))
        kl_negatives.append(kl_neg)
    
    # Contrastive objective: minimize distance to positive, maximize distance to negatives
    # Using temperature scaling
    loss = kl_positive / temperature
    
    # Add margin: we want negatives to be far away
    for kl_neg in kl_negatives:
        # Negative term: penalize if negative is too close
        margin = 1.0
        loss -= torch.clamp(kl_neg - margin, min=0.0) / temperature
    
    return loss


def curvature_penalty(witness_distributions, alpha=0.01):
    """
    Curvature regularization: encourage smooth manifold
    Penalize high variance in witness distributions
    
    Args:
        witness_distributions: List of witness distributions
        alpha: Regularization strength
    
    Returns:
        Curvature penalty
    """
    if len(witness_distributions) < 2:
        return torch.tensor(0.0)
    
    # Stack distributions
    dists = torch.stack(witness_distributions)
    
    # Compute pairwise KL divergences (measure of curvature)
    n = len(witness_distributions)
    total_curvature = 0.0
    epsilon = 1e-10
    
    for i in range(n):
        for j in range(i + 1, n):
            p = torch.clamp(dists[i], epsilon, 1.0)
            q = torch.clamp(dists[j], epsilon, 1.0)
            kl = torch.sum(p * torch.log(p / q))
            # Penalize very high curvature (distributions too different)
            total_curvature += torch.abs(kl)
    
    return alpha * total_curvature / (n * (n - 1) / 2)


def entropy(distribution):
    """
    Compute entropy of a probability distribution
    
    Args:
        distribution: Probability distribution (n_witnesses,)
    
    Returns:
        Entropy value
    """
    epsilon = 1e-10
    p = torch.clamp(distribution, epsilon, 1.0)
    return -torch.sum(p * torch.log(p))
