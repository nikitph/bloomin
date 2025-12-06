"""
Quick Test: Hard Negative Mining on Old Checkpoint
===================================================

Test if hard negative mining alone improves the old baseline checkpoint.
"""

import torch
import torch.nn.functional as F
import numpy as np

from src.utils import load_and_embed_data, split_categories, evaluate_recall
from src.model import AdversarialHybridREWAEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def mine_hard_triplets(model, embeddings, labels, num_triplets=1000):
    """Sample hard negatives: those closest to anchor but with different label."""
    model.eval()
    with torch.no_grad():
        # Encode all embeddings
        encoded = model(embeddings.unsqueeze(0), add_noise=False).squeeze(0)
        encoded = F.normalize(encoded, dim=-1)
    
    model.train()
    
    triplets = []
    unique_labels = torch.unique(labels)
    
    # Pre-compute masks to speed up
    label_to_indices = {label.item(): torch.where(labels == label)[0] for label in unique_labels}
    
    for _ in range(num_triplets):
        # Sample anchor class
        anchor_class = unique_labels[torch.randint(len(unique_labels), (1,))].item()
        anchor_idx = label_to_indices[anchor_class]
        
        if len(anchor_idx) < 2:
            continue
        
        # Sample anchor and positive
        a_idx = anchor_idx[torch.randint(len(anchor_idx), (1,))].item()
        p_idx = anchor_idx[anchor_idx != a_idx][torch.randint(len(anchor_idx)-1, (1,))].item()
        
        # HARD NEGATIVE MINING: Find closest negative
        anchor_enc = encoded[a_idx]
        
        neg_mask = labels != anchor_class
        neg_idx = torch.where(neg_mask)[0]
        
        if len(neg_idx) == 0:
            continue
            
        # Compute similarities to all negatives
        neg_encodings = encoded[neg_idx]
        sims = torch.matmul(anchor_enc, neg_encodings.T)
        
        # Select hardest (most similar) negative
        # With 20% probability, use semi-hard (rank 2-5) for stability
        if torch.rand(1) < 0.2:
            # Semi-hard
            hardest_k = min(5, len(neg_idx))
            if hardest_k > 1:
                hard_neg_indices = torch.topk(sims, hardest_k)[1]
                n_idx = neg_idx[hard_neg_indices[torch.randint(hardest_k, (1,))].item()]
            else:
                n_idx = neg_idx[torch.argmax(sims).item()]
        else:
            # Hardest
            n_idx = neg_idx[torch.argmax(sims).item()]
        
        triplets.append((a_idx, p_idx, n_idx.item()))
    
    return triplets

def quick_test_hard_negatives():
    """Load old checkpoint and fine-tune with hard negatives only."""
    print("="*70)
    print("Quick Test: Hard Negative Mining on Old Checkpoint")
    print("="*70)
    
    # Load Data
    print("\nLoading data...")
    embeddings, labels, target_names = load_and_embed_data()
    
    # Split Categories (same seed for consistency)
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    # Split Seen into Train/Val
    n_seen = len(seen_emb)
    indices = torch.randperm(n_seen)
    train_idx = indices[:int(0.9 * n_seen)]
    
    train_emb = seen_emb[train_idx].to(DEVICE)
    train_labels = seen_labels[train_idx].to(DEVICE)
    unseen_emb = unseen_emb.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)
    
    print(f"Train: {len(train_emb)}")
    print(f"Unseen (Zero-shot): {len(unseen_emb)}")
    
    # Check which checkpoints exist
    import os
    checkpoints = []
    for epoch in [5, 30, 35, 40, 45, 50]:
        ckpt_path = f'checkpoints/adversarial_best_{epoch}.pth'
        if os.path.exists(ckpt_path):
            checkpoints.append((epoch, ckpt_path))
    
    if not checkpoints:
        print("\n❌ No checkpoints found! Please run training first.")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints: {[e for e, _ in checkpoints]}")
    
    # Test on the best checkpoint (last one)
    best_epoch, best_ckpt = checkpoints[-1]
    
    print(f"\n{'='*70}")
    print(f"Loading checkpoint from epoch {best_epoch}: {best_ckpt}")
    print(f"{'='*70}")
    
    # Load model
    model = AdversarialHybridREWAEncoder(768, 256).to(DEVICE)
    model.load_state_dict(torch.load(best_ckpt))
    
    # Evaluate baseline (before fine-tuning)
    print("\n📊 Baseline Performance (before hard negative fine-tuning):")
    baseline_recall = evaluate_recall(model, unseen_emb, unseen_labels)
    print(f"Unseen Recall: {baseline_recall:.1%}")
    
    # Setup optimizer for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Lower LR for fine-tuning
    
    print(f"\n{'='*70}")
    print("Fine-tuning with Hard Negative Mining (5 epochs)")
    print(f"{'='*70}\n")
    
    best_recall = baseline_recall
    
    for epoch in range(1, 6):
        model.train()
        
        # Mine hard triplets
        triplets = mine_hard_triplets(model, train_emb, train_labels, num_triplets=1000)
        
        triplet_losses = []
        
        # Train on triplets
        for a_idx, p_idx, n_idx in triplets:
            # Get embeddings
            anchor = train_emb[a_idx].unsqueeze(0)
            positive = train_emb[p_idx].unsqueeze(0)
            negative = train_emb[n_idx].unsqueeze(0)
            
            # Encode
            anchor_enc = model(anchor, add_noise=False).squeeze(0)
            positive_enc = model(positive, add_noise=False).squeeze(0)
            negative_enc = model(negative, add_noise=False).squeeze(0)
            
            # Triplet loss with margin=1.0
            pos_dist = torch.sum((anchor_enc - positive_enc) ** 2)
            neg_dist = torch.sum((anchor_enc - negative_enc) ** 2)
            triplet_loss = F.relu(pos_dist - neg_dist + 1.0)
            
            # Backward
            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()
            
            triplet_losses.append(triplet_loss.item())
        
        # Evaluate
        recall = evaluate_recall(model, unseen_emb, unseen_labels)
        
        improvement = recall - baseline_recall
        symbol = "✓" if recall > best_recall else " "
        
        print(f"{symbol} Epoch {epoch}: "
              f"T_Loss={np.mean(triplet_losses):.4f}, "
              f"Unseen={recall:.1%} "
              f"({improvement:+.1%} from baseline)")
        
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), f'checkpoints/hard_neg_finetuned_epoch_{epoch}.pth')
    
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    print(f"Baseline (epoch {best_epoch}):     {baseline_recall:.1%}")
    print(f"After Hard Neg Fine-tuning: {best_recall:.1%}")
    print(f"Improvement:                {best_recall - baseline_recall:+.1%}")
    print(f"{'='*70}")
    
    if best_recall > baseline_recall:
        print("\n✅ Hard negative mining improved performance!")
    else:
        print("\n⚠️  No improvement observed. Try:")
        print("   - Adjusting the hard/semi-hard ratio")
        print("   - Using a lower learning rate")
        print("   - Training for more epochs")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    quick_test_hard_negatives()
