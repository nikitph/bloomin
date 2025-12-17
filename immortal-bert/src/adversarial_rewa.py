"""
Adversarial REWA Variants for Continual Learning

Four adversarial approaches to maximize forward transfer:
1. Task Discriminator (v1) - Domain adaptation style, make shared subspace task-agnostic
2. Witness Adversary (v2) - Push different tasks to use different witness dimensions
3. Contrastive Adversarial (v3) - Pull related tasks together, push unrelated apart
4. GAN-REWA (v4) - Full GAN approach with generator/discriminator

All variants build on Subspace-REWA with layer-wise lambda annealing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Function
from transformers import BertModel
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# GRADIENT REVERSAL LAYER (for v1)
# ============================================================================

class GradientReversalFunction(Function):
    """Gradient Reversal Layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ============================================================================
# VERSION 1: TASK DISCRIMINATOR (Domain Adaptation Style)
# ============================================================================

class TaskDiscriminator(nn.Module):
    """Discriminator that tries to identify which task an example came from."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, max_tasks: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_tasks)
        )

    def forward(self, x):
        return self.classifier(x)


class AdversarialREWA_TaskDiscriminator(nn.Module):
    """
    Version 1: Task Discriminator

    Forces the shared subspace to be task-invariant by training a discriminator
    to identify task identity while the encoder tries to fool it.

    Goal: I(h_shared; task_id) -> 0
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        lambda_adv: float = 0.5,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        max_tasks: int = 10,
        grl_alpha: float = 1.0
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        # Hyperparams
        self.lambda_max = lambda_max
        self.lambda_adv = lambda_adv
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        self.anneal_rate = anneal_rate
        self.min_lambda = min_lambda
        self.grl_alpha = grl_alpha

        # Task discriminator with gradient reversal
        self.task_discriminator = TaskDiscriminator(
            input_dim=subspace_dim,
            hidden_dim=256,
            max_tasks=max_tasks
        )
        self.grl = GradientReversalLayer(alpha=grl_alpha)

        # State
        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False
        self.frozen_encoder = None
        self.current_task_id = 0
        self.tasks_completed = 0

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1

    def set_task_id(self, task_id: int):
        """Set current task ID for discriminator training."""
        self.current_task_id = task_id

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces from first task."""
        self.eval()
        self.to(device)

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:self.subspace_dim, :]
            self.pca_components[i] = components.T.to(device)

        self.subspace_identified = True

        # Freeze encoder copy
        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        device = input_ids.device
        batch_size = input_ids.size(0)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        rewa_loss = torch.tensor(0.0, device=device)
        adv_loss = torch.tensor(0.0, device=device)

        if self.subspace_identified and labels is not None:
            # Standard subspace distillation loss
            if self.frozen_encoder is not None:
                with torch.no_grad():
                    frozen_outputs = self.frozen_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                for i in range(self.num_layers):
                    if i < self.layer_focus_start or i not in self.pca_components:
                        continue

                    U = self.pca_components[i]
                    curr_h = outputs.hidden_states[i+1][:, 0, :]
                    target_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                    curr_proj = curr_h @ U
                    target_proj = target_h @ U

                    layer_loss = F.mse_loss(curr_proj, target_proj)
                    lambda_l = self.layer_lambdas[i].to(device)
                    rewa_loss = rewa_loss + lambda_l * layer_loss

            # Adversarial task discriminator loss (on last focused layer)
            if self.num_layers - 1 in self.pca_components:
                U = self.pca_components[self.num_layers - 1]
                h_last = outputs.hidden_states[-1][:, 0, :]
                h_proj = h_last @ U  # (B, d_s)

                # Apply gradient reversal
                h_reversed = self.grl(h_proj)

                # Task prediction
                task_logits = self.task_discriminator(h_reversed)
                task_labels = torch.full((batch_size,), self.current_task_id,
                                        dtype=torch.long, device=device)

                # Cross entropy with only valid task classes
                adv_loss = F.cross_entropy(
                    task_logits[:, :self.current_task_id + 1],
                    task_labels
                )

        if loss is not None:
            loss = loss + rewa_loss + self.lambda_adv * adv_loss

        return {
            'loss': loss,
            'logits': logits,
            'rewa_loss': rewa_loss,
            'adv_loss': adv_loss
        }

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes


# ============================================================================
# VERSION 2: WITNESS ADVERSARY
# ============================================================================

class AdversarialREWA_WitnessAdversary(nn.Module):
    """
    Version 2: Witness Adversary

    Pushes different tasks to use DIFFERENT witness dimensions while
    maintaining the same shared subspace alignment.

    Goal: Maximize witness diversity across tasks for greater capacity.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        lambda_witness: float = 1.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        witness_overlap_penalty: float = 0.5
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        # Hyperparams
        self.lambda_max = lambda_max
        self.lambda_witness = lambda_witness
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        self.anneal_rate = anneal_rate
        self.min_lambda = min_lambda
        self.witness_overlap_penalty = witness_overlap_penalty

        # State
        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False
        self.frozen_encoder = None
        self.tasks_completed = 0

        # Witness tracking per task
        self.task_witnesses: Dict[int, Dict[int, torch.Tensor]] = {}  # task_id -> layer -> witness importance
        self.current_task_id = 0

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces and initial witness importance."""
        self.eval()
        self.to(device)

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            # SVD to get principal components and their importance
            U_full, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:self.subspace_dim, :]
            self.pca_components[i] = components.T.to(device)

            # Store witness importance (normalized singular values)
            witness_importance = (S[:self.subspace_dim] ** 2) / (S[:self.subspace_dim] ** 2).sum()

            if self.current_task_id not in self.task_witnesses:
                self.task_witnesses[self.current_task_id] = {}
            self.task_witnesses[self.current_task_id][i] = witness_importance.to(device)

        self.subspace_identified = True

        # Freeze encoder copy
        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

    def update_witnesses(self, dataloader: DataLoader, device: str):
        """Update witness importance for current task."""
        self.eval()

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    if i < self.layer_focus_start or i not in self.pca_components:
                        continue
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        if self.current_task_id not in self.task_witnesses:
            self.task_witnesses[self.current_task_id] = {}

        for i in range(self.num_layers):
            if i < self.layer_focus_start or i not in self.pca_components:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            U = self.pca_components[i].cpu()

            # Project to subspace
            X_proj = X @ U  # (N, d_s)

            # Compute variance per dimension as importance
            var_per_dim = X_proj.var(dim=0)
            witness_importance = var_per_dim / var_per_dim.sum()
            self.task_witnesses[self.current_task_id][i] = witness_importance.to(device)

        self.train()

    def compute_witness_adversary_loss(self, h_proj: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute loss that encourages using different witness dimensions than previous tasks.
        """
        device = h_proj.device

        if len(self.task_witnesses) <= 1:
            return torch.tensor(0.0, device=device)

        # Current batch variance per dimension
        batch_var = h_proj.var(dim=0)  # (d_s,)
        batch_importance = batch_var / (batch_var.sum() + 1e-8)

        # Compute overlap with previous tasks' witnesses
        overlap_loss = torch.tensor(0.0, device=device)

        for prev_task_id, prev_witnesses in self.task_witnesses.items():
            if prev_task_id >= self.current_task_id:
                continue
            if layer_idx not in prev_witnesses:
                continue

            prev_importance = prev_witnesses[layer_idx]

            # Penalize overlap: high similarity means both tasks use same dimensions
            # Use cosine similarity or dot product
            overlap = (batch_importance * prev_importance).sum()
            overlap_loss = overlap_loss + overlap

        return overlap_loss * self.witness_overlap_penalty

    def forward(self, input_ids, attention_mask, labels=None):
        device = input_ids.device

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        rewa_loss = torch.tensor(0.0, device=device)
        witness_loss = torch.tensor(0.0, device=device)

        if self.subspace_identified and labels is not None:
            # Standard subspace distillation loss
            if self.frozen_encoder is not None:
                with torch.no_grad():
                    frozen_outputs = self.frozen_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                for i in range(self.num_layers):
                    if i < self.layer_focus_start or i not in self.pca_components:
                        continue

                    U = self.pca_components[i]
                    curr_h = outputs.hidden_states[i+1][:, 0, :]
                    target_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                    curr_proj = curr_h @ U
                    target_proj = target_h @ U

                    # Standard REWA loss
                    layer_loss = F.mse_loss(curr_proj, target_proj)
                    lambda_l = self.layer_lambdas[i].to(device)
                    rewa_loss = rewa_loss + lambda_l * layer_loss

                    # Witness adversary loss
                    witness_loss = witness_loss + self.compute_witness_adversary_loss(curr_proj, i)

        if loss is not None:
            loss = loss + rewa_loss + self.lambda_witness * witness_loss

        return {
            'loss': loss,
            'logits': logits,
            'rewa_loss': rewa_loss,
            'witness_loss': witness_loss
        }

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes


# ============================================================================
# VERSION 3: CONTRASTIVE ADVERSARIAL
# ============================================================================

class AdversarialREWA_Contrastive(nn.Module):
    """
    Version 3: Contrastive Adversarial

    Pulls representations of related tasks together while pushing
    unrelated tasks apart in the shared subspace.

    Goal: Create explicit clustering structure in shared subspace.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        lambda_contrast: float = 0.5,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        margin: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        # Hyperparams
        self.lambda_max = lambda_max
        self.lambda_contrast = lambda_contrast
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        self.anneal_rate = anneal_rate
        self.min_lambda = min_lambda
        self.margin = margin
        self.temperature = temperature

        # State
        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False
        self.frozen_encoder = None
        self.tasks_completed = 0
        self.current_task_id = 0

        # Task centroids in subspace (for contrastive learning)
        self.task_centroids: Dict[int, torch.Tensor] = {}  # task_id -> centroid in subspace

        # Task relatedness (can be predefined or learned)
        # Sentiment tasks: AG_News(0), IMDB(1), SST2(2), Yelp(3)
        # NLI tasks: CoLA(4), MNLI(5), QNLI(6), QQP(7), RTE(8), MRPC(9)
        self.task_groups = {
            'sentiment': [0, 1, 2, 3],  # AG_News, IMDB, SST2, Yelp
            'nli': [5, 6, 8],            # MNLI, QNLI, RTE
            'paraphrase': [7, 9],        # QQP, MRPC
            'linguistic': [4],           # CoLA
        }

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def get_related_tasks(self, task_id: int) -> List[int]:
        """Get task IDs that are related to current task."""
        for group_name, group_ids in self.task_groups.items():
            if task_id in group_ids:
                return [t for t in group_ids if t != task_id and t in self.task_centroids]
        return []

    def get_unrelated_tasks(self, task_id: int) -> List[int]:
        """Get task IDs that are unrelated to current task."""
        related = set(self.get_related_tasks(task_id))
        related.add(task_id)
        return [t for t in self.task_centroids.keys() if t not in related]

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces and initial task centroid."""
        self.eval()
        self.to(device)

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:self.subspace_dim, :]
            self.pca_components[i] = components.T.to(device)

        self.subspace_identified = True

        # Compute and store task centroid (using last layer)
        if self.num_layers - 1 in self.pca_components:
            X_last = torch.cat(layer_activations[self.num_layers - 1], dim=0).to(device)
            U = self.pca_components[self.num_layers - 1]
            X_proj = X_last @ U
            centroid = X_proj.mean(dim=0)
            self.task_centroids[self.current_task_id] = centroid.detach()

        # Freeze encoder copy
        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

    def update_task_centroid(self, dataloader: DataLoader, device: str):
        """Update centroid for current task after training."""
        self.eval()

        projections = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                if self.num_layers - 1 in self.pca_components:
                    h = outputs.hidden_states[-1][:, 0, :]
                    U = self.pca_components[self.num_layers - 1]
                    h_proj = h @ U
                    projections.append(h_proj)

        if projections:
            all_proj = torch.cat(projections, dim=0)
            self.task_centroids[self.current_task_id] = all_proj.mean(dim=0).detach()

        self.train()

    def compute_contrastive_loss(self, h_proj: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss: pull towards related tasks, push from unrelated.
        """
        device = h_proj.device

        if len(self.task_centroids) == 0:
            return torch.tensor(0.0, device=device)

        # Current batch centroid
        batch_centroid = h_proj.mean(dim=0)

        related_tasks = self.get_related_tasks(self.current_task_id)
        unrelated_tasks = self.get_unrelated_tasks(self.current_task_id)

        loss = torch.tensor(0.0, device=device)

        # Pull towards related task centroids
        for related_id in related_tasks:
            related_centroid = self.task_centroids[related_id]
            # Negative distance (we want to minimize distance)
            similarity = F.cosine_similarity(
                batch_centroid.unsqueeze(0),
                related_centroid.unsqueeze(0)
            )
            loss = loss - similarity.mean()  # Maximize similarity

        # Push away from unrelated task centroids
        for unrelated_id in unrelated_tasks:
            unrelated_centroid = self.task_centroids[unrelated_id]
            similarity = F.cosine_similarity(
                batch_centroid.unsqueeze(0),
                unrelated_centroid.unsqueeze(0)
            )
            # Triplet-style margin loss
            loss = loss + F.relu(similarity - self.margin).mean()

        return loss

    def forward(self, input_ids, attention_mask, labels=None):
        device = input_ids.device

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        rewa_loss = torch.tensor(0.0, device=device)
        contrast_loss = torch.tensor(0.0, device=device)

        if self.subspace_identified and labels is not None:
            # Standard subspace distillation loss
            if self.frozen_encoder is not None:
                with torch.no_grad():
                    frozen_outputs = self.frozen_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                for i in range(self.num_layers):
                    if i < self.layer_focus_start or i not in self.pca_components:
                        continue

                    U = self.pca_components[i]
                    curr_h = outputs.hidden_states[i+1][:, 0, :]
                    target_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                    curr_proj = curr_h @ U
                    target_proj = target_h @ U

                    layer_loss = F.mse_loss(curr_proj, target_proj)
                    lambda_l = self.layer_lambdas[i].to(device)
                    rewa_loss = rewa_loss + lambda_l * layer_loss

            # Contrastive loss on last layer
            if self.num_layers - 1 in self.pca_components:
                U = self.pca_components[self.num_layers - 1]
                h_last = outputs.hidden_states[-1][:, 0, :]
                h_proj = h_last @ U
                contrast_loss = self.compute_contrastive_loss(h_proj)

        if loss is not None:
            loss = loss + rewa_loss + self.lambda_contrast * contrast_loss

        return {
            'loss': loss,
            'logits': logits,
            'rewa_loss': rewa_loss,
            'contrast_loss': contrast_loss
        }

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes


# ============================================================================
# VERSION 4: GAN-REWA
# ============================================================================

class RepresentationGenerator(nn.Module):
    """Generator that creates 'ideal' task representations."""
    def __init__(self, noise_dim: int = 64, hidden_dim: int = 256, output_dim: int = 64, num_tasks: int = 10):
        super().__init__()
        self.noise_dim = noise_dim
        self.task_embedding = nn.Embedding(num_tasks, 32)

        self.net = nn.Sequential(
            nn.Linear(noise_dim + 32, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Normalize output
        )

    def forward(self, batch_size: int, task_id: int, device: torch.device):
        z = torch.randn(batch_size, self.noise_dim, device=device)
        task_emb = self.task_embedding(torch.full((batch_size,), task_id, device=device))
        x = torch.cat([z, task_emb], dim=1)
        return self.net(x)


class RepresentationDiscriminator(nn.Module):
    """Discriminator that distinguishes real vs generated representations."""
    def __init__(self, input_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class AdversarialREWA_GAN(nn.Module):
    """
    Version 4: GAN-REWA

    Full GAN where generator creates 'ideal' task representations
    and encoder is trained to match them.

    Goal: Discover optimal transfer structure automatically.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        lambda_gan: float = 0.1,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        noise_dim: int = 64,
        max_tasks: int = 10
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        # Hyperparams
        self.lambda_max = lambda_max
        self.lambda_gan = lambda_gan
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        self.anneal_rate = anneal_rate
        self.min_lambda = min_lambda
        self.noise_dim = noise_dim

        # GAN components
        self.generator = RepresentationGenerator(
            noise_dim=noise_dim,
            hidden_dim=256,
            output_dim=subspace_dim,
            num_tasks=max_tasks
        )
        self.discriminator = RepresentationDiscriminator(
            input_dim=subspace_dim,
            hidden_dim=256
        )

        # State
        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False
        self.frozen_encoder = None
        self.tasks_completed = 0
        self.current_task_id = 0

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def get_gan_optimizers(self, lr: float = 1e-4):
        """Return separate optimizers for generator and discriminator."""
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return gen_optimizer, disc_optimizer

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces."""
        self.eval()
        self.to(device)

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:self.subspace_dim, :]
            self.pca_components[i] = components.T.to(device)

        self.subspace_identified = True

        # Freeze encoder copy
        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

    def train_discriminator_step(self, h_real: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Train discriminator to distinguish real from fake."""
        batch_size = h_real.size(0)

        # Real samples
        real_labels = torch.ones(batch_size, 1, device=device)
        real_pred = self.discriminator(h_real.detach())
        real_loss = F.binary_cross_entropy(real_pred, real_labels)

        # Fake samples
        fake_labels = torch.zeros(batch_size, 1, device=device)
        h_fake = self.generator(batch_size, self.current_task_id, device)
        fake_pred = self.discriminator(h_fake.detach())
        fake_loss = F.binary_cross_entropy(fake_pred, fake_labels)

        return (real_loss + fake_loss) / 2

    def train_generator_step(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Train generator to fool discriminator."""
        real_labels = torch.ones(batch_size, 1, device=device)
        h_fake = self.generator(batch_size, self.current_task_id, device)
        fake_pred = self.discriminator(h_fake)
        return F.binary_cross_entropy(fake_pred, real_labels)

    def forward(self, input_ids, attention_mask, labels=None, train_gan=True):
        device = input_ids.device
        batch_size = input_ids.size(0)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        rewa_loss = torch.tensor(0.0, device=device)
        gan_loss = torch.tensor(0.0, device=device)
        disc_loss = torch.tensor(0.0, device=device)
        gen_loss = torch.tensor(0.0, device=device)

        if self.subspace_identified and labels is not None:
            # Standard subspace distillation loss
            if self.frozen_encoder is not None:
                with torch.no_grad():
                    frozen_outputs = self.frozen_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                for i in range(self.num_layers):
                    if i < self.layer_focus_start or i not in self.pca_components:
                        continue

                    U = self.pca_components[i]
                    curr_h = outputs.hidden_states[i+1][:, 0, :]
                    target_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                    curr_proj = curr_h @ U
                    target_proj = target_h @ U

                    layer_loss = F.mse_loss(curr_proj, target_proj)
                    lambda_l = self.layer_lambdas[i].to(device)
                    rewa_loss = rewa_loss + lambda_l * layer_loss

            # GAN losses (on last layer projection)
            if train_gan and self.num_layers - 1 in self.pca_components:
                U = self.pca_components[self.num_layers - 1]
                h_last = outputs.hidden_states[-1][:, 0, :]
                h_proj = h_last @ U  # Real representations

                # Normalize for stability
                h_proj_norm = F.normalize(h_proj, dim=1)

                # Encoder adversarial loss: want real to look like generator output
                # (encoder tries to match the "ideal" representations)
                real_pred = self.discriminator(h_proj_norm)
                # Encoder wants discriminator to think real samples are fake
                # This pushes encoder towards generator's learned distribution
                gan_loss = -torch.log(1 - real_pred + 1e-8).mean()

        if loss is not None:
            loss = loss + rewa_loss + self.lambda_gan * gan_loss

        return {
            'loss': loss,
            'logits': logits,
            'rewa_loss': rewa_loss,
            'gan_loss': gan_loss,
            'h_proj': h_proj_norm if self.subspace_identified and self.num_layers - 1 in self.pca_components else None
        }

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes


# ============================================================================
# COMBINED MAXIMAL ADVERSARIAL (All techniques together)
# ============================================================================

class MaximalAdversarialREWA(nn.Module):
    """
    Combined version using all adversarial techniques:
    - Task Discriminator (v1)
    - Witness Adversary (v2)
    - Contrastive Learning (v3)

    (GAN excluded for training stability)
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        lambda_adv: float = 0.3,
        lambda_witness: float = 0.3,
        lambda_contrast: float = 0.3,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        max_tasks: int = 10
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        # Hyperparams
        self.lambda_max = lambda_max
        self.lambda_adv = lambda_adv
        self.lambda_witness = lambda_witness
        self.lambda_contrast = lambda_contrast
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        self.anneal_rate = anneal_rate
        self.min_lambda = min_lambda

        # V1: Task discriminator
        self.task_discriminator = TaskDiscriminator(subspace_dim, 256, max_tasks)
        self.grl = GradientReversalLayer(alpha=1.0)

        # V2: Witness tracking
        self.task_witnesses: Dict[int, Dict[int, torch.Tensor]] = {}

        # V3: Task centroids for contrastive
        self.task_centroids: Dict[int, torch.Tensor] = {}
        self.task_groups = {
            'sentiment': [0, 1, 2, 3],
            'nli': [5, 6, 8],
            'paraphrase': [7, 9],
            'linguistic': [4],
        }

        # State
        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False
        self.frozen_encoder = None
        self.tasks_completed = 0
        self.current_task_id = 0

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1

    def set_task_id(self, task_id: int):
        self.current_task_id = task_id

    def get_related_tasks(self, task_id: int) -> List[int]:
        for group_ids in self.task_groups.values():
            if task_id in group_ids:
                return [t for t in group_ids if t != task_id and t in self.task_centroids]
        return []

    def get_unrelated_tasks(self, task_id: int) -> List[int]:
        related = set(self.get_related_tasks(task_id))
        related.add(task_id)
        return [t for t in self.task_centroids.keys() if t not in related]

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        self.eval()
        self.to(device)

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            U_full, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:self.subspace_dim, :]
            self.pca_components[i] = components.T.to(device)

            # Store witness importance
            witness_importance = (S[:self.subspace_dim] ** 2) / (S[:self.subspace_dim] ** 2).sum()
            if self.current_task_id not in self.task_witnesses:
                self.task_witnesses[self.current_task_id] = {}
            self.task_witnesses[self.current_task_id][i] = witness_importance.to(device)

        # Store task centroid
        if self.num_layers - 1 in self.pca_components:
            X_last = torch.cat(layer_activations[self.num_layers - 1], dim=0).to(device)
            U = self.pca_components[self.num_layers - 1]
            X_proj = X_last @ U
            self.task_centroids[self.current_task_id] = X_proj.mean(dim=0).detach()

        self.subspace_identified = True

        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

    def update_task_info(self, dataloader: DataLoader, device: str):
        """Update witness importance and centroid for current task."""
        self.eval()

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                for i in range(self.num_layers):
                    if i >= self.layer_focus_start and i in self.pca_components:
                        hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                        layer_activations[i].append(hidden)

        if self.current_task_id not in self.task_witnesses:
            self.task_witnesses[self.current_task_id] = {}

        for i in range(self.num_layers):
            if i < self.layer_focus_start or i not in self.pca_components:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            U = self.pca_components[i].cpu()
            X_proj = X @ U

            var_per_dim = X_proj.var(dim=0)
            witness_importance = var_per_dim / var_per_dim.sum()
            self.task_witnesses[self.current_task_id][i] = witness_importance.to(device)

        # Update centroid
        if self.num_layers - 1 in self.pca_components:
            X_last = torch.cat(layer_activations[self.num_layers - 1], dim=0).to(device)
            U = self.pca_components[self.num_layers - 1]
            X_proj = X_last @ U
            self.task_centroids[self.current_task_id] = X_proj.mean(dim=0).detach()

        self.train()

    def forward(self, input_ids, attention_mask, labels=None):
        device = input_ids.device
        batch_size = input_ids.size(0)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        rewa_loss = torch.tensor(0.0, device=device)
        adv_loss = torch.tensor(0.0, device=device)
        witness_loss = torch.tensor(0.0, device=device)
        contrast_loss = torch.tensor(0.0, device=device)

        if self.subspace_identified and labels is not None and self.frozen_encoder is not None:
            with torch.no_grad():
                frozen_outputs = self.frozen_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            for i in range(self.num_layers):
                if i < self.layer_focus_start or i not in self.pca_components:
                    continue

                U = self.pca_components[i]
                curr_h = outputs.hidden_states[i+1][:, 0, :]
                target_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                curr_proj = curr_h @ U
                target_proj = target_h @ U

                # Standard REWA
                layer_loss = F.mse_loss(curr_proj, target_proj)
                lambda_l = self.layer_lambdas[i].to(device)
                rewa_loss = rewa_loss + lambda_l * layer_loss

                # V2: Witness adversary
                if len(self.task_witnesses) > 1:
                    batch_var = curr_proj.var(dim=0)
                    batch_importance = batch_var / (batch_var.sum() + 1e-8)

                    for prev_task_id, prev_witnesses in self.task_witnesses.items():
                        if prev_task_id >= self.current_task_id or i not in prev_witnesses:
                            continue
                        prev_importance = prev_witnesses[i]
                        overlap = (batch_importance * prev_importance).sum()
                        witness_loss = witness_loss + overlap * 0.5

            # V1: Task discriminator (on last layer)
            if self.num_layers - 1 in self.pca_components:
                U = self.pca_components[self.num_layers - 1]
                h_last = outputs.hidden_states[-1][:, 0, :]
                h_proj = h_last @ U

                h_reversed = self.grl(h_proj)
                task_logits = self.task_discriminator(h_reversed)
                task_labels = torch.full((batch_size,), self.current_task_id,
                                        dtype=torch.long, device=device)
                adv_loss = F.cross_entropy(
                    task_logits[:, :self.current_task_id + 1],
                    task_labels
                )

                # V3: Contrastive
                if len(self.task_centroids) > 0:
                    batch_centroid = h_proj.mean(dim=0)

                    for related_id in self.get_related_tasks(self.current_task_id):
                        related_centroid = self.task_centroids[related_id]
                        sim = F.cosine_similarity(
                            batch_centroid.unsqueeze(0),
                            related_centroid.unsqueeze(0)
                        )
                        contrast_loss = contrast_loss - sim.mean()

                    for unrelated_id in self.get_unrelated_tasks(self.current_task_id):
                        unrelated_centroid = self.task_centroids[unrelated_id]
                        sim = F.cosine_similarity(
                            batch_centroid.unsqueeze(0),
                            unrelated_centroid.unsqueeze(0)
                        )
                        contrast_loss = contrast_loss + F.relu(sim - 0.5).mean()

        if loss is not None:
            loss = (loss + rewa_loss +
                   self.lambda_adv * adv_loss +
                   self.lambda_witness * witness_loss +
                   self.lambda_contrast * contrast_loss)

        return {
            'loss': loss,
            'logits': logits,
            'rewa_loss': rewa_loss,
            'adv_loss': adv_loss,
            'witness_loss': witness_loss,
            'contrast_loss': contrast_loss
        }

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes
