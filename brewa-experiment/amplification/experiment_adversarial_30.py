"""
Adversarial Training - 30 Epochs
=================================

Train only the Adversarial method for 30 epochs to see if extended training improves results.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_rewa_encoder import TripletLossTrainer
from experiment_20newsgroups import load_and_embed_data, split_categories, evaluate_recall
from amplification.amplified_encoders import AdversarialHybridREWAEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

def train_adversarial_30_epochs():
    print("="*70)
    print("Adversarial Training - 30 Epochs")
    print("="*70)
    
    # Load Data
    embeddings, labels, target_names = load_and_embed_data()
    
    # Split Categories
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    # Split Seen into Train/Val
    n_seen = len(seen_emb)
    indices = torch.randperm(n_seen)
    train_idx = indices[:int(0.9 * n_seen)]
    val_idx = indices[int(0.9 * n_seen):]
    
    train_emb = seen_emb[train_idx].to(DEVICE)
    train_labels = seen_labels[train_idx].to(DEVICE)
    val_emb = seen_emb[val_idx].to(DEVICE)
    val_labels = seen_labels[val_idx].to(DEVICE)
    unseen_emb = unseen_emb.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)
    
    print(f"\nData Splits:")
    print(f"Train: {len(train_emb)}")
    print(f"Val:   {len(val_emb)}")
    print(f"Unseen (Zero-shot): {len(unseen_emb)}")
    
    # Initialize model
    model = AdversarialHybridREWAEncoder(768, 256).to(DEVICE)
    trainer = TripletLossTrainer(model, margin=1.0, lr=1e-3)
    
    # Adversarial optimizer
    adv_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    
    print(f"\n" + "-"*50)
    print(f"Training Adversarial for 30 Epochs...")
    print("-"*50)
    
    history = {
        'loss': [],
        'val_recall': [],
        'adv_loss': []
    }
    
    best_val_recall = 0
    best_epoch = 0
    
    for epoch in range(30):
        # Triplet loss training
        loss = trainer.train_epoch(train_emb, train_labels, num_triplets=1000)
        
        # Adversarial training step
        model.train()
        idx = torch.randperm(len(train_emb))[:BATCH_SIZE]
        batch = train_emb[idx]
        adv_loss = model.adversarial_loss(batch)
        
        adv_optimizer.zero_grad()
        adv_loss.backward()
        adv_optimizer.step()
        
        # Evaluate
        val_recall = evaluate_recall(model, val_emb, val_labels)
        
        history['loss'].append(loss)
        history['val_recall'].append(val_recall)
        history['adv_loss'].append(adv_loss.item())
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch + 1
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Adv Loss={adv_loss.item():.4f}, Val Recall={val_recall:.1%}")
    
    print(f"\nBest Val Recall: {best_val_recall:.1%} at Epoch {best_epoch}")
    
    # Final evaluation on unseen
    print(f"Evaluating on Unseen Categories...")
    unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
    print(f"Unseen Recall@10: {unseen_recall:.1%}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['loss'])
    axes[0].set_title('Triplet Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(history['adv_loss'])
    axes[1].set_title('Adversarial Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    axes[2].plot(history['val_recall'])
    axes[2].axhline(y=best_val_recall, color='r', linestyle='--', label=f'Best: {best_val_recall:.1%}')
    axes[2].set_title('Validation Recall')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Recall@10')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('adversarial_30_epochs.png')
    print("\nPlot saved to adversarial_30_epochs.png")
    
    return unseen_recall

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    unseen_recall = train_adversarial_30_epochs()
    
    print(f"\n" + "="*70)
    print(f"FINAL RESULT: Adversarial (30 epochs) Unseen Recall = {unseen_recall:.1%}")
    print("="*70)
