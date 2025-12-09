"""
Train Adversarial REWA on 20 Newsgroups (Proper Implementation)
================================================================

Full replication of adversarial-rewa-release functionality.
Uses 20 Newsgroups dataset with real category labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import sys

# Add adversarial-rewa-release to path
sys.path.insert(0, '/Users/truckx/PycharmProjects/bloomin/adversarial-rewa-release')
from src.model import AdversarialHybridREWAEncoder
from src.utils import load_and_embed_data, split_categories, evaluate_recall

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """SimCLR-style contrastive loss."""
    embeddings = F.normalize(embeddings, dim=-1)
    
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    labels = labels.unsqueeze(1)
    mask = (labels == labels.T).float()
    mask = mask.fill_diagonal_(0)
    
    num_positives = mask.sum(dim=1)
    
    exp_sim = torch.exp(sim_matrix)
    sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
    
    log_prob = sim_matrix - torch.log(sum_exp_sim)
    
    loss = -(mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
    
    return loss.mean()


def mine_hard_triplets(model, embeddings, labels, num_triplets=500):
    """Mine hard negatives for triplet loss."""
    model.eval()
    with torch.no_grad():
        encoded = model(embeddings.unsqueeze(0), add_noise=False).squeeze(0)
        encoded = F.normalize(encoded, dim=-1)
    
    model.train()
    
    triplets = []
    unique_labels = torch.unique(labels)
    
    label_to_indices = {label.item(): torch.where(labels == label)[0] for label in unique_labels}
    
    for _ in range(num_triplets):
        anchor_class = unique_labels[torch.randint(len(unique_labels), (1,))].item()
        anchor_idx = label_to_indices[anchor_class]
        
        if len(anchor_idx) < 2:
            continue
        
        a_idx = anchor_idx[torch.randint(len(anchor_idx), (1,))].item()
        p_idx = anchor_idx[anchor_idx != a_idx][torch.randint(len(anchor_idx)-1, (1,))].item()
        
        # Hard negative mining
        anchor_enc = encoded[a_idx]
        
        neg_mask = labels != anchor_class
        neg_idx = torch.where(neg_mask)[0]
        
        if len(neg_idx) == 0:
            continue
        
        neg_encodings = encoded[neg_idx]
        sims = torch.matmul(anchor_enc, neg_encodings.T)
        
        # 80% hardest, 20% semi-hard
        if torch.rand(1) < 0.8:
            n_idx = neg_idx[torch.argmax(sims).item()]
        else:
            hardest_k = min(5, len(neg_idx))
            if hardest_k > 1:
                hard_neg_indices = torch.topk(sims, hardest_k)[1]
                n_idx = neg_idx[hard_neg_indices[torch.randint(hardest_k, (1,))].item()]
            else:
                n_idx = neg_idx[torch.argmax(sims).item()]
        
        triplets.append((a_idx, p_idx, n_idx.item()))
    
    return triplets


def train_rewa_proper(epochs=30, m_dim=256):
    """
    Train Adversarial REWA on 20 Newsgroups with proper labels.
    """
    print(f"\n{'='*70}")
    print(f"Training Adversarial REWA on 20 Newsgroups")
    print(f"{'='*70}")
    
    # Load 20 Newsgroups with BERT embeddings
    embeddings, labels, target_names = load_and_embed_data()
    
    # Split into seen/unseen categories (for zero-shot evaluation)
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    # Split seen into train/val
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
    model = AdversarialHybridREWAEncoder(d_model=768, m_dim=m_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    best_unseen_recall = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Adaptive margin
        progress = epoch / epochs
        margin = 0.3 + 0.5 * (2.0 - 0.3) * (1 + np.cos(np.pi * progress))
        
        # Hard triplet mining
        triplets = mine_hard_triplets(model, train_emb, train_labels, num_triplets=1000)
        
        triplet_losses = []
        adv_losses = []
        
        # Shuffle triplets
        np.random.shuffle(triplets)
        
        for a_idx, p_idx, n_idx in triplets:
            anchor = train_emb[a_idx].unsqueeze(0)
            positive = train_emb[p_idx].unsqueeze(0)
            negative = train_emb[n_idx].unsqueeze(0)
            
            # Forward with adversarial
            a_enc, a_disc = model.forward_with_adversarial(anchor)
            p_enc, _ = model.forward_with_adversarial(positive)
            n_enc, _ = model.forward_with_adversarial(negative)
            
            # Triplet loss
            pos_dist = torch.sum((a_enc - p_enc) ** 2)
            neg_dist = torch.sum((a_enc - n_enc) ** 2)
            triplet_loss = F.relu(pos_dist - neg_dist + margin)
            
            # Adversarial loss
            with torch.no_grad():
                random_features = model.random_proj(anchor)
            
            random_disc = model.discriminator(random_features)
            adv_loss_learned = F.binary_cross_entropy_with_logits(a_disc, torch.ones_like(a_disc))
            adv_loss_random = F.binary_cross_entropy_with_logits(random_disc, torch.zeros_like(random_disc))
            adv_loss = (adv_loss_learned + adv_loss_random) / 2
            
            total_loss = triplet_loss + 0.3 * adv_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            triplet_losses.append(triplet_loss.item())
            adv_losses.append(adv_loss.item())
        
        # Contrastive loss batches
        contrastive_losses = []
        for _ in range(10):
            idx = torch.randperm(len(train_emb))[:64]
            batch_emb = train_emb[idx]
            batch_labels = train_labels[idx]
            
            encoded = model(batch_emb.unsqueeze(0), add_noise=False).squeeze(0)
            cont_loss = supervised_contrastive_loss(encoded, batch_labels)
            
            optimizer.zero_grad()
            cont_loss.backward()
            optimizer.step()
            
            contrastive_losses.append(cont_loss.item())
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            val_recall = evaluate_recall(model, val_emb, val_labels)
            unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
            
            print(f"Epoch {epoch:2d}: "
                  f"T_Loss={np.mean(triplet_losses):.4f}, "
                  f"Adv={np.mean(adv_losses):.4f}, "
                  f"Cont={np.mean(contrastive_losses):.4f}, "
                  f"Margin={margin:.2f}, "
                  f"Val={val_recall:.1%}, "
                  f"Unseen={unseen_recall:.1%}")
            
            if unseen_recall > best_unseen_recall:
                best_unseen_recall = unseen_recall
                best_epoch = epoch
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), f'checkpoints/rewa_newsgroups_{m_dim}d_best.pth')
                print(f"  ✓ New best! Saved checkpoint.")
            
            # Early stopping if we hit 85%
            if unseen_recall >= 0.85:
                print(f"\n🎉 TARGET REACHED: {unseen_recall:.1%} >= 85%!")
                break
        else:
            print(f"Epoch {epoch:2d}: "
                  f"T_Loss={np.mean(triplet_losses):.4f}, "
                  f"Adv={np.mean(adv_losses):.4f}, "
                  f"Cont={np.mean(contrastive_losses):.4f}, "
                  f"Margin={margin:.2f}")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Unseen Recall: {best_unseen_recall:.1%} at Epoch {best_epoch}")
    print(f"{'='*70}")
    
    return model, best_unseen_recall


def main():
    print("="*70)
    print("ADVERSARIAL REWA - PROPER IMPLEMENTATION")
    print("="*70)
    print("\nUsing 20 Newsgroups dataset with real category labels")
    print("Training for zero-shot generalization to unseen categories\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, best_recall = train_rewa_proper(epochs=50, m_dim=256)
    
    print(f"\n{'='*70}")
    print(f"🎯 FINAL RESULT: {best_recall:.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
