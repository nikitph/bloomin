"""
Train Adversarial REWA for BERT/GPT-2 Compression
==================================================

Apply the Adversarial Hybrid REWA approach to compress BERT and GPT-2 embeddings
while maximizing retrieval recall.

Goal: Achieve 85%+ recall with 3x compression (768D → 256D)
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

# Local imports
from model_loaders import load_model_and_get_embeddings
from corpus_loaders import load_sentences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def create_category_labels(n_samples, n_categories=20):
    """
    Create synthetic category labels for Wikipedia sentences.
    In real use, you'd use actual document categories.
    """
    # Simple: assign sequential categories
    labels = torch.arange(n_samples) % n_categories
    return labels


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
        encoded = model(embeddings, add_noise=False)
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


def train_rewa(model_name, embeddings, labels, epochs=30, m_dim=256):
    """
    Train Adversarial REWA on embeddings.
    
    Args:
        model_name: 'bert' or 'gpt2'
        embeddings: (N, 768) tensor
        labels: (N,) tensor of category labels
        epochs: Number of training epochs
        m_dim: Compressed dimension
    """
    print(f"\n{'='*70}")
    print(f"Training Adversarial REWA for {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Input: 768D → Output: {m_dim}D ({768/m_dim:.1f}x compression)")
    print(f"Samples: {len(embeddings)}, Categories: {len(torch.unique(labels))}")
    
    # Split train/val
    n_samples = len(embeddings)
    indices = torch.randperm(n_samples)
    train_idx = indices[:int(0.8 * n_samples)]
    val_idx = indices[int(0.8 * n_samples):]
    
    train_emb = embeddings[train_idx].to(DEVICE)
    train_labels = labels[train_idx].to(DEVICE)
    val_emb = embeddings[val_idx].to(DEVICE)
    val_labels = labels[val_idx].to(DEVICE)
    
    print(f"Train: {len(train_emb)}, Val: {len(val_emb)}")
    
    # Initialize model
    model = AdversarialHybridREWAEncoder(d_model=768, m_dim=m_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    best_val_recall = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Adaptive margin
        progress = epoch / epochs
        margin = 0.3 + 0.5 * (2.0 - 0.3) * (1 + np.cos(np.pi * progress))
        
        # Hard triplet mining
        triplets = mine_hard_triplets(model, train_emb, train_labels, num_triplets=500)
        
        triplet_losses = []
        adv_losses = []
        
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
            
            encoded = model(batch_emb, add_noise=False)
            cont_loss = supervised_contrastive_loss(encoded, batch_labels)
            
            optimizer.zero_grad()
            cont_loss.backward()
            optimizer.step()
            
            contrastive_losses.append(cont_loss.item())
        
        # Evaluate on validation
        if epoch % 5 == 0:
            val_recall = evaluate_recall(model, val_emb, val_labels)
            
            print(f"Epoch {epoch:2d}: "
                  f"T_Loss={np.mean(triplet_losses):.4f}, "
                  f"Adv={np.mean(adv_losses):.4f}, "
                  f"Cont={np.mean(contrastive_losses):.4f}, "
                  f"Val_Recall={val_recall:.1%}")
            
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), f'checkpoints/rewa_{model_name}_{m_dim}d_best.pth')
                print(f"  ✓ New best! Saved checkpoint.")
    
    print(f"\nTraining complete! Best val recall: {best_val_recall:.1%}")
    
    return model, best_val_recall


def evaluate_recall(model, embeddings, labels, k=10):
    """Evaluate recall@k."""
    model.eval()
    
    with torch.no_grad():
        encoded = model(embeddings, add_noise=False)
        encoded = F.normalize(encoded, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(encoded, encoded.T)
    
    recalls = []
    for i in range(len(embeddings)):
        # Get top-k most similar (excluding self)
        sims = sim_matrix[i]
        sims[i] = -float('inf')  # Exclude self
        top_k_idx = torch.topk(sims, k)[1]
        
        # Check if any have same label
        same_label = (labels[top_k_idx] == labels[i]).float()
        recall = same_label.sum().item() / min(k, (labels == labels[i]).sum().item() - 1)
        recalls.append(recall)
    
    return np.mean(recalls)


def main():
    print("="*70)
    print("ADVERSARIAL REWA COMPRESSION BENCHMARK")
    print("="*70)
    print("\nLoading Wikipedia embeddings...")
    
    # Load sentences
    texts = load_sentences(n_samples=5000)  # Smaller for faster training
    print(f"  ✓ Loaded {len(texts)} sentences")
    
    # Create synthetic category labels (in practice, use real categories)
    labels = create_category_labels(len(texts), n_categories=20)
    
    # Train for BERT
    print("\n" + "="*70)
    print("BERT EMBEDDINGS")
    print("="*70)
    
    bert_config = {'source': 'transformers', 'model': 'bert-base-uncased', 'dim': 768}
    bert_embeddings = load_model_and_get_embeddings(bert_config, texts)
    bert_embeddings = torch.from_numpy(bert_embeddings).float()
    
    bert_model, bert_recall = train_rewa('bert', bert_embeddings, labels, epochs=30, m_dim=256)
    
    # Train for GPT-2
    print("\n" + "="*70)
    print("GPT-2 EMBEDDINGS")
    print("="*70)
    
    gpt2_config = {'source': 'transformers', 'model': 'gpt2', 'dim': 768}
    gpt2_embeddings = load_model_and_get_embeddings(gpt2_config, texts)
    gpt2_embeddings = torch.from_numpy(gpt2_embeddings).float()
    
    gpt2_model, gpt2_recall = train_rewa('gpt2', gpt2_embeddings, labels, epochs=30, m_dim=256)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nBERT (768D → 256D):")
    print(f"  Recall@10: {bert_recall:.1%}")
    print(f"  Compression: 3.0x")
    
    print(f"\nGPT-2 (768D → 256D):")
    print(f"  Recall@10: {gpt2_recall:.1%}")
    print(f"  Compression: 3.0x")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
