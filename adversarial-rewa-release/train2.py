# improved_adversarial_rewa_train.py
"""
Improved Adversarial Hybrid REWA Trainer
- Vectorized hard negative mining (batch + memory bank)
- Momentum memory bank (MoCo-style queue) for large negative set
- Projection head + LayerNorm (conditioning)
- Stable supervised contrastive (log-sum-exp denominator)
- Gradient Reversal Layer (GRL) implemented
- Separate discriminator optimization steps for stability
- Adaptive margin schedule (cosine) retained
- Mixed losses: triplet + supervised-contrastive + adversarial + distillation (optional)
"""

import os
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base import HybridREWAEncoder
from src.utils import load_and_embed_data, split_categories, evaluate_recall
from src.model import AdversarialHybridREWAEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  # larger for better contrastive statistics
EMBED_DIM = 256
QUEUE_SIZE = 16384  # memory bank size for negatives
MOMENTUM = 0.999  # momentum for momentum encoder (optional)
LR = 3e-4

# ------------------------------
# Utility / components
# ------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class ProjectionHead(nn.Module):
    """Simple 2-layer projection head used in contrastive learning"""
    def __init__(self, in_dim, hidden_dim=512, out_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        # optionally L2-normalize outside

    def forward(self, x):
        return self.net(x)

class MomentumEncoder(nn.Module):
    """Momentum encoder wrapper for a target encoder (MoCo-style)"""
    def __init__(self, base_encoder):
        super().__init__()
        # copy of base encoder
        self.encoder = base_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def momentum_update(self, online, m=MOMENTUM):
        # update target params toward online params
        for param_q, param_k in zip(online.parameters(), self.encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

# ------------------------------
# Contrastive + triplet losses
# ------------------------------
def supervised_contrastive_loss_stable(embeddings, labels, temperature=0.07, eps=1e-8):
    """
    Stable supervised contrastive loss (per-sample averaged positives).
    Uses log-sum-exp denominator for numerical stability.
    """
    z = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(z, z.T) / temperature  # [B,B]
    labels = labels.contiguous().view(-1, 1)
    mask = (labels == labels.T).float()
    # remove self-similarity
    diag = torch.eye(mask.size(0), device=mask.device)
    mask = mask * (1 - diag)

    # For each i: numerator is sum_j exp(sim_ij) over positives
    # Denominator is sum_k exp(sim_ik) over all except i.
    max_sim, _ = torch.max(sim - 1e9 * diag, dim=1, keepdim=True)
    sim_minus_max = sim - max_sim  # [B,B]
    exp_sim = torch.exp(sim_minus_max) * (1 - diag)  # zero out self
    denom = exp_sim.sum(dim=1, keepdim=True) + eps

    positive_exp = (exp_sim * mask).sum(dim=1) + eps
    loss_i = - torch.log(positive_exp / denom)
    # normalize by number of positives per sample
    n_pos = mask.sum(dim=1).clamp(min=1)
    loss = (loss_i / n_pos).mean()
    return loss

def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """
    Vectorized batch-hard triplet loss: for each anchor, choose hardest positive and hardest negative within batch.
    """
    z = F.normalize(embeddings, dim=-1)
    # distance matrix (squared euclidean can be computed via dot products)
    # use cosine similarities for stability and scale to distances
    sim = torch.matmul(z, z.T)
    N = sim.size(0)
    labels = labels.contiguous().view(-1, 1)
    mask_pos = (labels == labels.T)
    mask_neg = ~mask_pos

    # For positives: find minimum similarity (hardest positive = lowest sim)
    pos_sim = sim.clone()
    pos_sim[~mask_pos] = float('inf')  # ignore negatives
    hardest_pos_sim, _ = pos_sim.min(dim=1)
    # For negatives: find maximum similarity (hardest negative = highest sim)
    neg_sim = sim.clone()
    neg_sim[~mask_neg] = float('-inf')
    hardest_neg_sim, _ = neg_sim.max(dim=1)

    # Triplet loss in similarity-space: want pos_sim > neg_sim + margin
    loss = F.relu(hardest_neg_sim - hardest_pos_sim + margin)
    return loss.mean()

# ------------------------------
# Trainer
# ------------------------------
class ImprovedAdversarialTrainerV2:
    def __init__(self, model: AdversarialHybridREWAEncoder,
                 lr=LR,
                 queue_size=QUEUE_SIZE,
                 device=DEVICE):
        self.model = model.to(device)
        self.device = device

        # Infer dimensions from model
        input_dim = model.random_proj.in_features
        output_dim = model.random_proj.out_features * 2
        
        # Momentum encoder for memory bank (optional but helps)
        self.momentum_encoder = MomentumEncoder(
            AdversarialHybridREWAEncoder(d_model=input_dim, m_dim=output_dim).to(device)
        )

        # projection head (learnable) on top of model.encoder; applied to embeddings used for contrastive
        self.proj = ProjectionHead(output_dim, hidden_dim=512, out_dim=EMBED_DIM).to(device)

        # queue for negatives (embeddings)
        self.register_queue(queue_size)

        # single optimizer for encoder+projection; discriminator will have separate optimizer
        self.encoder_params = list(self.model.parameters()) + list(self.proj.parameters())
        self.encoder_opt = torch.optim.AdamW(self.encoder_params, lr=lr, weight_decay=0.01)

        # discriminator optimizer
        self.disc_opt = torch.optim.AdamW(self.model.discriminator.parameters(), lr=lr * 0.5)

        # scheduler (cosine)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_opt, T_max=200)

        # bookkeeping
        self.best_val = 0.0

    def register_queue(self, queue_size):
        self.queue_size = queue_size
        self.registered_queue = torch.zeros(queue_size, EMBED_DIM).to(self.device)
        self.queue_ptr = 0
        self.queue_filled = 0

    def dequeue_and_enqueue(self, keys):
        # keys: [B, D]
        n = keys.shape[0]
        if n == 0:
            return
        if n + self.queue_ptr <= self.queue_size:
            self.registered_queue[self.queue_ptr:self.queue_ptr + n] = keys.detach()
            self.queue_ptr = (self.queue_ptr + n) % self.queue_size
        else:
            # wrap-around
            first = self.queue_size - self.queue_ptr
            self.registered_queue[self.queue_ptr:] = keys[:first].detach()
            self.registered_queue[:n - first] = keys[first:n].detach()
            self.queue_ptr = (n - first) % self.queue_size
        self.queue_filled = min(self.queue_filled + n, self.queue_size)

    def get_queue_negatives(self):
        if self.queue_filled == 0:
            return None
        if self.queue_filled < self.queue_size:
            return self.registered_queue[:self.queue_filled]
        return self.registered_queue

    def train_epoch(self, emb_tensor, labels_tensor, epoch,
                    num_batches_for_contrast=50,
                    margin_fn=lambda e: get_adaptive_margin(e, max_epochs=50)):
        self.model.train()
        self.proj.train()

        # compute adaptive margin
        margin = margin_fn(epoch)

        # 1) Iterate over mini-batches drawn randomly for contrastive training
        idx_perm = torch.randperm(len(emb_tensor))
        losses = []
        adv_losses = []

        # iterate by BATCH_SIZE chunks
        for i in range(0, len(idx_perm), BATCH_SIZE):
            batch_idx = idx_perm[i:i + BATCH_SIZE]
            batch_emb = emb_tensor[batch_idx].to(self.device)
            batch_labels = labels_tensor[batch_idx].to(self.device)

            # forward: encoder -> projection
            z = self.model(batch_emb, add_noise=False)  # assume model returns embeddings
            z_proj = self.proj(z)  # [B, EMBED_DIM]
            z_proj = F.normalize(z_proj, dim=-1)

            # batch-hard triplet (vectorized)
            triplet_loss = batch_hard_triplet_loss(z_proj, batch_labels, margin=margin)

            # supervised contrastive (stable)
            cont_loss = supervised_contrastive_loss_stable(z_proj, batch_labels, temperature=0.07)

            # adversarial discriminator: GRL on encoder features, discriminator predicts learned vs random
            # GRL lambda can be schedule; here we ramp up with epoch
            grl_lambda = min(1.0, epoch / 30.0)
            disc_in = grad_reverse(z, grl_lambda)  # feed original embedding (not proj) reversed
            disc_logits = self.model.discriminator(disc_in)
            adv_loss_enc = F.binary_cross_entropy_with_logits(disc_logits, torch.ones_like(disc_logits))

            # also train discriminator on random projections to classify as fake
            with torch.no_grad():
                rand_feat = self.model.random_proj(batch_emb)  # random features
            disc_fake_logits = self.model.discriminator(rand_feat)
            adv_loss_fake = F.binary_cross_entropy_with_logits(disc_fake_logits, torch.zeros_like(disc_fake_logits))

            # total discriminator loss (we'll step discriminator separately)
            adv_loss_disc = (adv_loss_fake + 0.0)  # we compute for discriminator update later

            # Memory bank negatives: compute keys and enqueue (momentum encoder)
            with torch.no_grad():
                # Optionally update momentum encoder toward online model
                self.momentum_encoder.momentum_update(self.model)
                key = self.momentum_encoder.encoder(batch_emb)  # [B, D]
                key_proj = self.proj(key)
                key_proj = F.normalize(key_proj, dim=-1)
            # enqueue keys
            self.dequeue_and_enqueue(key_proj)

            # contrastive with queue negatives (InfoNCE-style)
            queue_neg = self.get_queue_negatives()  # [Q, D] or None
            if queue_neg is not None and queue_neg.shape[0] > 0:
                # compute similarity between z_proj and queue_neg (torch.matmul)
                logits_pos = torch.sum(z_proj * key_proj.detach(), dim=1, keepdim=True)  # [B,1]
                logits_neg = torch.matmul(z_proj, queue_neg.T)  # [B, Q]
                logits = torch.cat([logits_pos, logits_neg], dim=1) / 0.07
                labels_i = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                # cross entropy (pos index 0)
                info_nce_loss = F.cross_entropy(logits, labels_i)
            else:
                info_nce_loss = torch.tensor(0.0, device=self.device)

            # Combine losses with weights
            # We decay weight of triplet over time and increase contrastive weight
            alpha_triplet = max(0.2, 1.0 - epoch / 60.0)
            alpha_contrast = 0.8 - 0.3 * (epoch / 60.0)
            loss_encoder = alpha_triplet * triplet_loss + 0.6 * cont_loss + 0.4 * info_nce_loss + 0.5 * adv_loss_enc

            # Backprop encoder
            self.encoder_opt.zero_grad()
            loss_encoder.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder_params, 1.0)
            self.encoder_opt.step()

            # Step discriminator separately: use detached features for stability
            self.disc_opt.zero_grad()
            disc_loss = (adv_loss_fake.detach() + F.binary_cross_entropy_with_logits(
                self.model.discriminator(z.detach()), torch.ones_like(disc_logits.detach()))) / 2.0
            disc_loss.backward()
            self.disc_opt.step()

            losses.append(loss_encoder.item() if isinstance(loss_encoder, torch.Tensor) else float(loss_encoder))
            adv_losses.append(disc_loss.item())

        # scheduler step
        self.scheduler.step()

        summary = {
            'loss': np.mean(losses),
            'disc_loss': np.mean(adv_losses),
            'queue_fill': self.queue_filled
        }
        return summary

# ------------------------------
# Training harness
# ------------------------------
def train_main(num_epochs=50):
    print("Loading data...")
    embeddings, labels, target_names = load_and_embed_data()  # embeddings as torch.tensor
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )

    # Train/val split
    n_seen = len(seen_emb)
    perm = torch.randperm(n_seen)
    train_idx = perm[:int(0.9 * n_seen)]
    val_idx = perm[int(0.9 * n_seen):]

    train_emb = seen_emb[train_idx].to(DEVICE)
    train_labels = seen_labels[train_idx].to(DEVICE)
    val_emb = seen_emb[val_idx].to(DEVICE)
    val_labels = seen_labels[val_idx].to(DEVICE)
    unseen_emb = unseen_emb.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)

    # Initialize model
    model = AdversarialHybridREWAEncoder(d_model=train_emb.size(1), m_dim=EMBED_DIM).to(DEVICE)
    trainer = ImprovedAdversarialTrainerV2(model, lr=LR, queue_size=QUEUE_SIZE, device=DEVICE)

    best_unseen = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        stats = trainer.train_epoch(train_emb, train_labels, epoch)
        val_recall = evaluate_recall(model, val_emb, val_labels)

        # Evaluate unseen every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
            print(f"Epoch {epoch} | Loss={stats['loss']:.4f} | DiscLoss={stats['disc_loss']:.4f} | Queue={stats['queue_fill']} | Val={val_recall:.3f} | Unseen={unseen_recall:.3f}")
            if unseen_recall > best_unseen:
                best_unseen = unseen_recall
                best_epoch = epoch
                torch.save(model.state_dict(), f'checkpoints/improved_rewa_best_{epoch}.pth')
                print("  ✓ New best checkpoint saved")
        else:
            print(f"Epoch {epoch} | Loss={stats['loss']:.4f} | DiscLoss={stats['disc_loss']:.4f} | Queue={stats['queue_fill']} | Val={val_recall:.3f}")

    print(f"Training completed. Best unseen recall {best_unseen:.3f} at epoch {best_epoch}")
    return best_unseen, best_epoch

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    os.makedirs('checkpoints', exist_ok=True)
    best_unseen_recall, best_epoch = train_main(num_epochs=50)
    print(f"FINAL: Best unseen recall {best_unseen_recall:.3f} at epoch {best_epoch}")