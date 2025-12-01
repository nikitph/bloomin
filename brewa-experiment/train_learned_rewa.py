"""
Train Learned REWA Encoder
===========================

Train the learned projection encoder end-to-end on semantic similarity task.

Expected results:
- Epoch 5: Recall@10 ~30-40%
- Epoch 10: Recall@10 ~50-60%
- Epoch 20: Recall@10 ~70-80%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from learned_rewa_encoder import (
    LearnedContinuousREWAEncoder,
    ContrastiveLoss,
    TripletLoss,
)


def generate_semantic_training_data(
    n_classes=50,
    samples_per_class=100,
    d_model=768,
    intra_class_std=0.3,
):
    """
    Generate training data with semantic structure.
    
    Args:
        n_classes: Number of semantic categories
        samples_per_class: Samples per category
        d_model: Embedding dimension
        intra_class_std: Within-class standard deviation
    
    Returns:
        embeddings: [n_samples, d_model]
        labels: [n_samples]
    """
    n_samples = n_classes * samples_per_class
    
    # Create class centroids (well-separated)
    centroids = torch.randn(n_classes, d_model)
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    centroids = centroids * 3.0  # Spread out
    
    embeddings = []
    labels = []
    
    for class_id in range(n_classes):
        centroid = centroids[class_id]
        
        for _ in range(samples_per_class):
            # Sample around centroid
            sample = centroid + torch.randn(d_model) * intra_class_std
            sample = sample / sample.norm()  # Normalize
            
            embeddings.append(sample)
            labels.append(class_id)
    
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)
    
    return embeddings, labels


def evaluate_recall(model, embeddings, labels, top_k=10):
    """
    Evaluate Recall@K on the dataset.
    
    Args:
        model: Learned encoder
        embeddings: [N, d_model] embeddings
        labels: [N] class labels
        top_k: K for Recall@K
    
    Returns:
        recall: Recall@K score
    """
    model.eval()
    
    with torch.no_grad():
        # Encode all embeddings
        encoded = model(embeddings.unsqueeze(0)).squeeze(0)  # [N, m_dim]
        
        # Compute similarity matrix
        similarity = torch.mm(encoded, encoded.T)  # [N, N]
        
        # Get top-k for each query
        _, indices = similarity.topk(top_k + 1, dim=1)  # +1 for self
        
        # Compute recall
        recall_sum = 0
        for i in range(len(labels)):
            # Get neighbors (excluding self)
            neighbors = indices[i, 1:]  # Remove self
            neighbor_labels = labels[neighbors]
            
            # Count correct (same class)
            correct = (neighbor_labels == labels[i]).sum().item()
            recall_sum += correct / min(top_k, (labels == labels[i]).sum().item() - 1)
        
        recall = recall_sum / len(labels)
    
    return recall


def train_learned_rewa(
    d_model=768,
    m_dim=256,
    n_classes=50,
    samples_per_class=100,
    num_epochs=20,
    batch_size=32,
    lr=1e-3,
    loss_type='contrastive',  # 'contrastive' or 'triplet'
):
    """
    Train learned REWA encoder.
    
    Returns:
        model: Trained encoder
        history: Training history
    """
    print("="*70)
    print("Training Learned REWA Encoder")
    print("="*70)
    print(f"\nParameters:")
    print(f"  d_model: {d_model}")
    print(f"  m_dim: {m_dim}")
    print(f"  n_classes: {n_classes}")
    print(f"  samples_per_class: {samples_per_class}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  lr: {lr}")
    print(f"  loss_type: {loss_type}")
    
    # Generate training data
    print("\nGenerating training data...")
    embeddings, labels = generate_semantic_training_data(
        n_classes, samples_per_class, d_model
    )
    
    print(f"Generated {len(embeddings)} samples across {n_classes} classes")
    
    # Create model
    print("\nInitializing model...")
    model = LearnedContinuousREWAEncoder(d_model, m_dim, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Loss function
    if loss_type == 'contrastive':
        criterion = ContrastiveLoss(temperature=0.07)
    else:
        criterion = TripletLoss(margin=0.2)
    
    # Training loop
    print("\nTraining...")
    history = {
        'train_loss': [],
        'recall': [],
    }
    
    best_recall = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        
        # Create dataloader
        dataset = TensorDataset(embeddings, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0
        num_batches = 0
        
        for batch_emb, batch_labels in dataloader:
            model.train()  # Ensure train mode
            
            # Encode (keep batch dimension)
            encoded = model(batch_emb.unsqueeze(0), add_noise=True).squeeze(0)  # [B, m_dim]
            
            # Compute loss
            loss = criterion(encoded, batch_labels)
            
            # Skip if loss is zero or nan
            if loss.item() == 0 or torch.isnan(loss):
                continue
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        # Evaluate
        recall = evaluate_recall(model, embeddings, labels, top_k=10)
        
        history['train_loss'].append(avg_loss)
        history['recall'].append(recall)
        
        if recall > best_recall:
            best_recall = recall
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Loss={avg_loss:.4f}, "
                  f"Recall@10={recall:.1%}, "
                  f"Best={best_recall:.1%}")
    
    print("\n" + "="*70)
    print(f"Training Complete!")
    print(f"Best Recall@10: {best_recall:.1%}")
    print("="*70)
    
    return model, history


def plot_training_history(history):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Recall curve
    ax2.plot(epochs, history['recall'], marker='s', linewidth=2, color='green')
    ax2.axhline(0.6, color='red', linestyle='--', label='60% target', alpha=0.5)
    ax2.axhline(0.8, color='orange', linestyle='--', label='80% target', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Recall@10 on Training Set')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('learned_rewa_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: learned_rewa_training.png")
    plt.close()


def quick_validation_test():
    """
    Quick test on simple 2D data to verify learning works.
    """
    print("="*70)
    print("Quick Validation Test (2D → 1D projection)")
    print("="*70)
    
    # Create simple 2D data with two clusters
    cluster1 = torch.randn(100, 2) + torch.tensor([3.0, 3.0])
    cluster2 = torch.randn(100, 2) + torch.tensor([-3.0, -3.0])
    data = torch.cat([cluster1, cluster2])
    labels = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    
    # Test random projection
    print("\n1. Random Projection (baseline)")
    random_proj = torch.randn(2, 1)
    random_proj = random_proj / random_proj.norm()
    
    projected_random = data @ random_proj
    class0_mean_random = projected_random[labels == 0].mean()
    class1_mean_random = projected_random[labels == 1].mean()
    separation_random = torch.abs(class0_mean_random - class1_mean_random).item()
    
    print(f"   Class separation: {separation_random:.3f}")
    
    # Test learned projection
    print("\n2. Learned Projection")
    model = LearnedContinuousREWAEncoder(d_model=2, m_dim=1, use_mlp=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = ContrastiveLoss(temperature=0.1)
    
    for epoch in range(100):
        model.train()
        
        # Encode
        encoded = model(data.unsqueeze(0)).squeeze(0)  # [200, 1]
        
        # Loss
        loss = criterion(encoded, labels)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                projected = model(data.unsqueeze(0)).squeeze(0)
                class0_mean = projected[labels == 0].mean()
                class1_mean = projected[labels == 1].mean()
                separation = torch.abs(class0_mean - class1_mean).item()
            print(f"   Epoch {epoch+1}: separation={separation:.3f}")
    
    print(f"\n   Improvement: {separation/separation_random:.1f}× better separation!")
    print("="*70)


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Quick validation test
    quick_validation_test()
    
    # Full training
    print("\n\n")
    model, history = train_learned_rewa(
        d_model=768,
        m_dim=256,
        n_classes=50,
        samples_per_class=100,
        num_epochs=20,
        batch_size=32,
        lr=1e-3,
        loss_type='contrastive',
    )
    
    # Plot results
    plot_training_history(history)
    
    # Save model
    torch.save(model.state_dict(), 'learned_rewa_encoder.pth')
    print("\nModel saved to: learned_rewa_encoder.pth")
