"""
REWA-C for BERT: Geometric Preservation in Continual Learning

Key insight from ricci-continual-learning:
- Preserve the STRUCTURE of class representations (distances, angles)
- Not absolute positions, but relative geometry

For BERT:
- Extract [CLS] embeddings as class representations
- Compute centroid geometry per task
- Regularize to preserve structure when learning new tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TaskGeometry:
    """Stores geometric structure of a task's class representations."""
    task_name: str
    num_classes: int
    centroids: torch.Tensor  # (num_classes, hidden_dim)
    centroid_distances: torch.Tensor  # (num_classes, num_classes)
    centroid_angles: torch.Tensor  # (num_classes, num_classes) cosine similarities
    class_spreads: torch.Tensor  # (num_classes,) avg distance to centroid


class GeometricPreserver(nn.Module):
    """
    Preserves geometric structure of class representations.

    Reuses the structural preservation approach from ricci-continual-learning:
    - Preserve relative distances between class centroids
    - Preserve angular arrangement of classes
    - Preserve relative spreads (how tight each class is)
    """

    def __init__(
        self,
        distance_weight: float = 1.0,
        angle_weight: float = 1.0,
        spread_weight: float = 0.5
    ):
        super().__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.spread_weight = spread_weight

        self.task_geometries: List[TaskGeometry] = []

    def compute_geometry(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        task_name: str
    ) -> TaskGeometry:
        """Compute geometric structure from embeddings."""
        device = embeddings.device
        dim = embeddings.shape[1]

        centroids = torch.zeros(num_classes, dim, device=device)
        spreads = torch.zeros(num_classes, device=device)

        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_emb = embeddings[mask]
                centroid = class_emb.mean(dim=0)
                centroids[c] = centroid
                spreads[c] = torch.norm(class_emb - centroid, dim=1).mean()

        # For normalized centroids: ||a-b||² = 2 - 2(a·b)
        # So we just need dot products for relative distance comparison
        centroids_norm = F.normalize(centroids, dim=1, eps=1e-8)
        similarities = centroids_norm @ centroids_norm.T  # Cosine similarity matrix
        # distances² ∝ (1 - similarity), but we'll just preserve the similarity matrix directly
        distances = 1 - similarities  # This is proportional to squared distance

        # Angular structure (same as similarities for normalized vectors)
        angles = similarities

        return TaskGeometry(
            task_name=task_name,
            num_classes=num_classes,
            centroids=centroids.detach(),
            centroid_distances=distances.detach(),
            centroid_angles=angles.detach(),
            class_spreads=spreads.detach()
        )

    def store_geometry(self, geometry: TaskGeometry):
        """Store geometry for a completed task."""
        self.task_geometries.append(geometry)

    def compute_preservation_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Compute loss to preserve all stored task geometries."""
        if not self.task_geometries:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        device = embeddings.device
        dim = embeddings.shape[1]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Compute current geometry
        curr_centroids = torch.zeros(num_classes, dim, device=device)
        curr_spreads = torch.zeros(num_classes, device=device)

        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_emb = embeddings[mask]
                centroid = class_emb.mean(dim=0)
                curr_centroids[c] = centroid
                curr_spreads[c] = torch.norm(class_emb - centroid, dim=1).mean()

        # Current structure (using dot products for MPS compatibility)
        curr_centroids_norm = F.normalize(curr_centroids, dim=1, eps=1e-8)
        curr_similarities = curr_centroids_norm @ curr_centroids_norm.T
        curr_distances = 1 - curr_similarities
        curr_angles = curr_similarities

        # Compare against all stored geometries
        for ref_geom in self.task_geometries:
            # Match dimensions if different number of classes
            min_classes = min(num_classes, ref_geom.num_classes)

            # Distance structure preservation (normalized for scale invariance)
            ref_dist_norm = ref_geom.centroid_distances[:min_classes, :min_classes]
            curr_dist_norm = curr_distances[:min_classes, :min_classes]

            ref_dist_norm = ref_dist_norm / (ref_dist_norm.mean() + 1e-8)
            curr_dist_norm = curr_dist_norm / (curr_dist_norm.mean() + 1e-8)

            dist_loss = F.mse_loss(curr_dist_norm, ref_dist_norm.to(device))
            total_loss = total_loss + self.distance_weight * dist_loss

            # Angular structure preservation
            ref_angles = ref_geom.centroid_angles[:min_classes, :min_classes]
            curr_angles_subset = curr_angles[:min_classes, :min_classes]

            angle_loss = F.mse_loss(curr_angles_subset, ref_angles.to(device))
            total_loss = total_loss + self.angle_weight * angle_loss

            # Spread preservation (normalized)
            ref_spread_norm = ref_geom.class_spreads[:min_classes]
            curr_spread_norm = curr_spreads[:min_classes]

            ref_spread_norm = ref_spread_norm / (ref_spread_norm.mean() + 1e-8)
            curr_spread_norm = curr_spread_norm / (curr_spread_norm.mean() + 1e-8)

            spread_loss = F.mse_loss(curr_spread_norm, ref_spread_norm.to(device))
            total_loss = total_loss + self.spread_weight * spread_loss

        return total_loss / len(self.task_geometries)


class REWABert(nn.Module):
    """
    BERT with REWA-C geometric preservation for continual learning.

    Architecture:
    - DistilBERT encoder (faster than full BERT)
    - Classification head per task (or shared)
    - Geometric preserver for regularization
    """

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_classes: int = 2,
        lambda_geom: float = 100.0
    ):
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Geometric preserver
        self.geom_preserver = GeometricPreserver()
        self.lambda_geom = lambda_geom

        self.num_classes = num_classes
        self.current_task = None

    def get_embeddings(self, input_ids, attention_mask) -> torch.Tensor:
        """Get [CLS] embeddings."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token is the first token
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with optional loss computation."""
        embeddings = self.get_embeddings(input_ids, attention_mask)
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            # Task loss
            ce_loss = F.cross_entropy(logits, labels)

            # Geometric preservation loss
            geom_loss = self.geom_preserver.compute_preservation_loss(
                embeddings, labels, self.num_classes
            )

            loss = ce_loss + self.lambda_geom * geom_loss

        return {'loss': loss, 'logits': logits, 'embeddings': embeddings}

    def store_task_geometry(self, dataloader: DataLoader, task_name: str):
        """Extract and store geometry for completed task."""
        self.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(next(self.parameters()).device)
                attention_mask = batch['attention_mask'].to(next(self.parameters()).device)
                labels = batch['labels']

                embeddings = self.get_embeddings(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)

        geometry = self.geom_preserver.compute_geometry(
            all_embeddings, all_labels, self.num_classes, task_name
        )
        self.geom_preserver.store_geometry(geometry)

        print(f"Stored geometry for {task_name}: {self.num_classes} classes, "
              f"avg distance={geometry.centroid_distances.mean():.3f}")

    def update_classifier(self, num_classes: int):
        """Update classifier for new task with different number of classes."""
        if num_classes != self.num_classes:
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(
                next(self.parameters()).device
            )
            self.num_classes = num_classes


class BaselineBert(nn.Module):
    """Baseline BERT without any continual learning protection."""

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_classes: int = 2
    ):
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = self.dropout(cls_embeddings)
        logits = self.classifier(cls_embeddings)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {'loss': loss, 'logits': logits, 'embeddings': cls_embeddings}

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(
                next(self.parameters()).device
            )
            self.num_classes = num_classes


class EWCBert(nn.Module):
    """BERT with Elastic Weight Consolidation."""

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_classes: int = 2,
        ewc_lambda: float = 1000.0
    ):
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        self.ewc_lambda = ewc_lambda
        self.fisher_info = {}
        self.optimal_params = {}

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = self.dropout(cls_embeddings)
        logits = self.classifier(cls_embeddings)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            ewc_loss = self._ewc_loss()
            loss = ce_loss + self.ewc_lambda * ewc_loss

        return {'loss': loss, 'logits': logits, 'embeddings': cls_embeddings}

    def _ewc_loss(self) -> torch.Tensor:
        if not self.fisher_info:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.named_parameters():
            # Skip classifier (changes size between tasks)
            if 'classifier' in name:
                continue
            if name in self.fisher_info:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                # Only apply if shapes match
                if fisher.shape == param.shape:
                    loss = loss + (fisher * (param - optimal).pow(2)).sum()

        return loss

    def compute_fisher(self, dataloader: DataLoader, num_samples: int = 200):
        """Compute Fisher information matrix (encoder only, skip classifier)."""
        self.eval()
        # Only compute Fisher for encoder params (not classifier which changes size)
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()
                  if p.requires_grad and 'classifier' not in n}

        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break

            input_ids = batch['input_ids'].to(next(self.parameters()).device)
            attention_mask = batch['attention_mask'].to(next(self.parameters()).device)
            labels = batch['labels'].to(next(self.parameters()).device)

            self.zero_grad()
            outputs = self.forward(input_ids, attention_mask, labels)
            outputs['loss'].backward()

            for name, param in self.named_parameters():
                if param.grad is not None and 'classifier' not in name:
                    fisher[name] += param.grad.pow(2)

            count += len(labels)

        # Average and store
        for name in fisher:
            fisher[name] /= count
            if name in self.fisher_info:
                self.fisher_info[name] = self.fisher_info[name] + fisher[name]
            else:
                self.fisher_info[name] = fisher[name]

        # Store optimal params (encoder only)
        self.optimal_params = {n: p.detach().clone() for n, p in self.named_parameters()
                               if 'classifier' not in n}

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(
                next(self.parameters()).device
            )
            self.num_classes = num_classes


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return correct / total


if __name__ == "__main__":
    # Quick test
    print("Testing REWA-BERT...")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    model = REWABert(num_classes=2, lambda_geom=100.0).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Test forward pass
    text = "This is a test sentence for BERT."
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    print("Test passed!")
