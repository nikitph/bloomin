"""
Train and Test Hybrid REWA
===========================

Train hybrid encoder with proper train/val/test splits.
Expected: 55-65% test recall with good generalization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from hybrid_rewa_encoder import HybridREWAEncoder, TripletLossTrainer


def generate_semantic_data(n_classes=50, samples_per_class=100, d_model=768):
    """Generate semantic clustering data."""
    centroids = torch.randn(n_classes, d_model)
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    centroids = centroids * 3.0
    
    embeddings = []
    labels = []
    
    for class_id in range(n_classes):
        centroid = centroids[class_id]
        for _ in range(samples_per_class):
            sample = centroid + torch.randn(d_model) * 0.3
            sample = sample / sample.norm()
            embeddings.append(sample)
            labels.append(class_id)
    
    return torch.stack(embeddings), torch.tensor(labels)


def evaluate_recall(model, embeddings, labels, top_k=10):
    """Evaluate Recall@K."""
    model.eval()
    
    with torch.no_grad():
        encoded = model(embeddings.unsqueeze(0), add_noise=False).squeeze(0)
        similarity = torch.mm(encoded, encoded.T)
        
        _, indices = similarity.topk(top_k + 1, dim=1)
        
        recall_sum = 0
        for i in range(len(labels)):
            neighbors = indices[i, 1:]
            neighbor_labels = labels[neighbors]
            correct = (neighbor_labels == labels[i]).sum().item()
            max_possible = min(top_k, (labels == labels[i]).sum().item() - 1)
            
            if max_possible > 0:
                recall_sum += correct / max_possible
        
        recall = recall_sum / len(labels)
    
    return recall


def train_hybrid_rewa():
    """Train hybrid REWA with proper splits and early stopping."""
    
    print("="*70)
    print("Training Hybrid REWA Encoder")
    print("="*70)
    
    # Generate data
    print("\nGenerating data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    all_embeddings, all_labels = generate_semantic_data(
        n_classes=50,
        samples_per_class=100,
        d_model=768
    )
    
    print(f"Total samples: {len(all_embeddings)}")
    
    # Split: 60% train, 20% val, 20% test
    N = len(all_embeddings)
    indices = torch.randperm(N)
    
    train_size = int(0.6 * N)
    val_size = int(0.2 * N)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_emb, train_labels = all_embeddings[train_idx], all_labels[train_idx]
    val_emb, val_labels = all_embeddings[val_idx], all_labels[val_idx]
    test_emb, test_labels = all_embeddings[test_idx], all_labels[test_idx]
    
    print(f"Train: {len(train_emb)}, Val: {len(val_emb)}, Test: {len(test_emb)}")
    
    # Create model
    print("\nInitializing Hybrid REWA...")
    model = HybridREWAEncoder(
        d_model=768,
        m_dim=256,
        random_ratio=0.5,  # 50% random, 50% learned
        dropout=0.3
    )
    
    print(f"Random dims: {model.m_random} (frozen)")
    print(f"Learned dims: {model.m_learned} (trainable)")
    
    # Trainer
    trainer = TripletLossTrainer(
        model,
        margin=1.0,
        lr=1e-3,
        weight_decay=0.05  # Strong regularization
    )
    
    # Training loop with early stopping
    print("\nTraining...")
    num_epochs = 50
    best_val_recall = 0
    patience = 10
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_recall': [],
        'test_recall': [],
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_emb, train_labels, num_triplets=500)
        
        # Validate
        val_recall = evaluate_recall(model, val_emb, val_labels, top_k=10)
        test_recall = evaluate_recall(model, test_emb, test_labels, top_k=10)
        
        history['train_loss'].append(train_loss)
        history['val_recall'].append(val_recall)
        history['test_recall'].append(test_recall)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Loss={train_loss:.4f}, "
                  f"Val={val_recall:.1%}, "
                  f"Test={test_recall:.1%}")
        
        # Early stopping
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            patience_counter = 0
            torch.save(model.state_dict(), 'hybrid_rewa_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('hybrid_rewa_best.pth'))
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)
    
    final_val_recall = evaluate_recall(model, val_emb, val_labels, top_k=10)
    final_test_recall = evaluate_recall(model, test_emb, test_labels, top_k=10)
    
    print(f"\nValidation Recall@10: {final_val_recall:.1%}")
    print(f"Test Recall@10: {final_test_recall:.1%}")
    print(f"Generalization gap: {abs(final_val_recall - final_test_recall)*100:.1f}%")
    
    # Test on completely NEW data
    print("\n" + "="*70)
    print("Testing on COMPLETELY NEW Data")
    print("="*70)
    
    torch.manual_seed(999)
    np.random.seed(999)
    
    new_emb, new_labels = generate_semantic_data(
        n_classes=30,  # Different number
        samples_per_class=20,
        d_model=768
    )
    
    new_recall = evaluate_recall(model, new_emb, new_labels, top_k=10)
    print(f"\nRecall on NEW clusters: {new_recall:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nTest Recall (held-out):     {final_test_recall:.1%}")
    print(f"New Clusters Recall:        {new_recall:.1%}")
    print(f"Compression:                {model.get_compression_ratio():.1f}×")
    
    # Compare with baselines
    print(f"\nComparison:")
    print(f"  Random REWA (baseline):   27%")
    print(f"  Hybrid REWA (this):       {final_test_recall:.1%}")
    print(f"  Improvement:              {final_test_recall/0.27:.1f}×")
    
    if final_test_recall > 0.55:
        print("\n✅ SUCCESS: Hybrid REWA achieves target performance!")
    elif final_test_recall > 0.45:
        print("\n⚠️  GOOD: Decent performance, room for improvement")
    else:
        print("\n❌ NEEDS WORK: Performance below target")
    
    # Plot results
    plot_training_history(history)
    
    return model, history, final_test_recall


def plot_training_history(history):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Triplet Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Recall curves
    ax2.plot(epochs, history['val_recall'], marker='s', linewidth=2, label='Validation', color='blue')
    ax2.plot(epochs, history['test_recall'], marker='^', linewidth=2, label='Test', color='green')
    ax2.axhline(0.27, color='red', linestyle='--', label='Random baseline', alpha=0.5)
    ax2.axhline(0.55, color='orange', linestyle='--', label='Target (55%)', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Validation & Test Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('hybrid_rewa_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: hybrid_rewa_training.png")
    plt.close()


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train and test
    model, history, test_recall = train_hybrid_rewa()
    
    print("\n" + "="*70)
    print("Hybrid REWA Training Complete!")
    print("="*70)
