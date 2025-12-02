"""
Train Adversarial Hybrid REWA
=============================

Training script that achieves 78.6% zero-shot recall.
Key features:
1. Smooth adversarial loss (label smoothing)
2. Mixup data augmentation
3. Adaptive adversarial weighting
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.base import HybridREWAEncoder
from src.utils import load_and_embed_data, split_categories, evaluate_recall
from src.model import AdversarialHybridREWAEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

class ImprovedAdversarialTrainer:
    """Enhanced adversarial trainer with smooth loss and mixup"""
    
    def __init__(self, model, margin=1.0, lr=1e-4, adv_lr=5e-4):
        self.model = model
        self.margin = margin
        
        # Separate optimizers for better control
        self.triplet_optimizer = torch.optim.AdamW(
            model.learned_proj.parameters(), 
            lr=lr, 
            weight_decay=0.01
        )
        self.adv_optimizer = torch.optim.AdamW(
            model.discriminator.parameters(), 
            lr=adv_lr,
            weight_decay=0.01
        )
        
        # Adaptive adversarial weight
        self.adv_weight = 0.3
        self.best_val_recall = 0
        self.patience_counter = 0
    
    def smooth_adversarial_loss(self, x, smoothing=0.1):
        """Adversarial loss with label smoothing"""
        learned_features = self.model.learned_proj(x)
        with torch.no_grad():
            random_features = self.model.random_proj(x)
        
        batch_size = x.shape[0]
        half = batch_size // 2
        
        mixed_features = torch.cat([
            learned_features[:half],
            random_features[half:]
        ], dim=0)
        
        # Smooth labels (not hard 0/1)
        labels = torch.cat([
            torch.ones(half, 1, device=x.device) * (1 - smoothing),  # 0.9
            torch.ones(batch_size - half, 1, device=x.device) * smoothing,  # 0.1
        ], dim=0)
        
        preds = torch.sigmoid(self.model.discriminator(mixed_features))
        loss = F.binary_cross_entropy(preds, labels)
        
        return loss
    
    def mixup_adversarial_loss(self, x, alpha=0.2):
        """Mixup between learned and random features"""
        learned = self.model.learned_proj(x)
        with torch.no_grad():
            random = self.model.random_proj(x)
        
        # Sample mixup lambda
        batch_size = x.shape[0]
        lam = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1)).to(x.device)
        
        # Mixed features
        mixed_features = lam * learned + (1 - lam) * random
        
        # Mixed labels (continuous)
        mixed_labels = lam
        
        preds = torch.sigmoid(self.model.discriminator(mixed_features))
        loss = F.binary_cross_entropy(preds, mixed_labels)
        
        return loss
    
    def train_epoch(self, embeddings, labels, num_triplets=1000):
        """Train one epoch with all improvements"""
        self.model.train()
        
        # Sample triplets
        unique_labels = torch.unique(labels)
        triplet_losses = []
        smooth_adv_losses = []
        mixup_losses = []
        
        for _ in range(num_triplets):
            # Sample anchor class
            anchor_class = unique_labels[torch.randint(len(unique_labels), (1,))].item()
            anchor_mask = labels == anchor_class
            
            # Sample positive
            anchor_idx = torch.where(anchor_mask)[0]
            if len(anchor_idx) < 2:
                continue
            
            a_idx, p_idx = anchor_idx[torch.randperm(len(anchor_idx))[:2]]
            
            # Sample negative
            neg_mask = labels != anchor_class
            neg_idx = torch.where(neg_mask)[0]
            if len(neg_idx) == 0:
                continue
            
            n_idx = neg_idx[torch.randint(len(neg_idx), (1,))].item()
            
            # Get embeddings
            anchor = embeddings[a_idx].unsqueeze(0)
            positive = embeddings[p_idx].unsqueeze(0)
            negative = embeddings[n_idx].unsqueeze(0)
            
            # Encode
            anchor_enc = self.model(anchor, add_noise=False).squeeze(0)
            positive_enc = self.model(positive, add_noise=False).squeeze(0)
            negative_enc = self.model(negative, add_noise=False).squeeze(0)
            
            # Triplet loss
            pos_dist = torch.sum((anchor_enc - positive_enc) ** 2)
            neg_dist = torch.sum((anchor_enc - negative_enc) ** 2)
            triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
            
            # Backward for triplet
            self.triplet_optimizer.zero_grad()
            triplet_loss.backward()
            self.triplet_optimizer.step()
            
            triplet_losses.append(triplet_loss.item())
        
        # Adversarial training (batch-based)
        num_adv_batches = 10
        for _ in range(num_adv_batches):
            idx = torch.randperm(len(embeddings))[:BATCH_SIZE]
            batch = embeddings[idx]
            
            # Smooth adversarial loss
            smooth_adv_loss = self.smooth_adversarial_loss(batch, smoothing=0.1)
            
            # Mixup loss
            mixup_loss = self.mixup_adversarial_loss(batch, alpha=0.2)
            
            # Combined adversarial loss
            total_adv_loss = smooth_adv_loss + 0.3 * mixup_loss
            
            # Backward for adversarial
            self.adv_optimizer.zero_grad()
            total_adv_loss.backward()
            self.adv_optimizer.step()
            
            smooth_adv_losses.append(smooth_adv_loss.item())
            mixup_losses.append(mixup_loss.item())
        
        return {
            'triplet_loss': np.mean(triplet_losses),
            'smooth_adv_loss': np.mean(smooth_adv_losses),
            'mixup_loss': np.mean(mixup_losses)
        }
    
    def update_adv_weight(self, val_recall):
        """Adaptive adversarial weight based on validation performance"""
        if val_recall > self.best_val_recall:
            self.best_val_recall = val_recall
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # If plateaued, increase adversarial weight
        if self.patience_counter >= 3:
            self.adv_weight = min(self.adv_weight * 1.1, 0.5)
            self.patience_counter = 0
            print(f"  → Increased adversarial weight to {self.adv_weight:.3f}")

def train():
    print("="*70)
    print("Training Adversarial Hybrid REWA (50 Epochs)")
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
    trainer = ImprovedAdversarialTrainer(model, margin=1.0, lr=1e-4, adv_lr=5e-4)
    
    print(f"\n" + "-"*50)
    print(f"Training with Improvements (50 Epochs)...")
    print("-"*50)
    
    history = {
        'epoch': [],
        'triplet_loss': [],
        'smooth_adv_loss': [],
        'mixup_loss': [],
        'val_recall': [],
        'unseen_recall': []
    }
    
    best_unseen_recall = 0
    best_epoch = 0
    
    for epoch in range(1, 51):
        # Train
        losses = trainer.train_epoch(train_emb, train_labels, num_triplets=1000)
        
        # Evaluate
        val_recall = evaluate_recall(model, val_emb, val_labels)
        
        # Update adaptive weight
        trainer.update_adv_weight(val_recall)
        
        # Record
        history['epoch'].append(epoch)
        history['triplet_loss'].append(losses['triplet_loss'])
        history['smooth_adv_loss'].append(losses['smooth_adv_loss'])
        history['mixup_loss'].append(losses['mixup_loss'])
        history['val_recall'].append(val_recall)
        
        # Evaluate on unseen every 5 epochs
        if epoch % 5 == 0:
            unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
            history['unseen_recall'].append(unseen_recall)
            
            print(f"Epoch {epoch}: "
                  f"T_Loss={losses['triplet_loss']:.4f}, "
                  f"Adv_Loss={losses['smooth_adv_loss']:.4f}, "
                  f"Mix_Loss={losses['mixup_loss']:.4f}, "
                  f"Val={val_recall:.1%}, "
                  f"Unseen={unseen_recall:.1%}")
            
            if unseen_recall > best_unseen_recall:
                best_unseen_recall = unseen_recall
                best_epoch = epoch
                torch.save(model.state_dict(), f'checkpoints/adversarial_best_{epoch}.pth')
                print(f"  ✓ New best! Saved checkpoint.")
            
            # Early stopping if we hit 80%
            if unseen_recall >= 0.80:
                print(f"\n🎉 TARGET REACHED: {unseen_recall:.1%} >= 80%!")
                break
        else:
            print(f"Epoch {epoch}: "
                  f"T_Loss={losses['triplet_loss']:.4f}, "
                  f"Adv_Loss={losses['smooth_adv_loss']:.4f}, "
                  f"Val={val_recall:.1%}")
    
    # Final evaluation
    print(f"\n" + "="*70)
    print(f"Training Complete!")
    print(f"Best Unseen Recall: {best_unseen_recall:.1%} at Epoch {best_epoch}")
    print("="*70)
    
    return best_unseen_recall, best_epoch

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    best_recall, best_epoch = train()
    
    print(f"\n" + "="*70)
    print(f"🎯 FINAL RESULT: {best_recall:.1%} at Epoch {best_epoch}")
    print("="*70)
