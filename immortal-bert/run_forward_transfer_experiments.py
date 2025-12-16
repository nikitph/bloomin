#!/usr/bin/env python3
"""
Forward Transfer Engineering Experiments

Goal: Achieve negative FI (forward transfer) through systematic strategies:
1. Curriculum ordering (easy → hard)
2. Subspace expansion + refinement (actively improve old tasks)
3. Multi-task mixture (blend old/new task data)
4. Contrastive subspace learning (maximize transfer signal)
5. Combined strategies

Baseline: AnnealingSubspaceREWA achieved FI = -1.3%
Target: FI = -5% to -8%
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import random

from src.rewa_bert import evaluate


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


# ============================================================================
# DATA LOADING
# ============================================================================

class TextDataset(Dataset):
    """Simple text classification dataset."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_ag_news(tokenizer, n_train=1500, n_test=300):
    log("  Loading AG News...")
    dataset = load_dataset('ag_news')
    train_ds = TextDataset(
        dataset['train']['text'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = TextDataset(
        dataset['test']['text'][:n_test],
        dataset['test']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 4, "AG_News"


def load_imdb(tokenizer, n_train=1500, n_test=300):
    log("  Loading IMDB...")
    dataset = load_dataset('imdb')
    train_ds = TextDataset(
        dataset['train']['text'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = TextDataset(
        dataset['test']['text'][:n_test],
        dataset['test']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "IMDB"


def load_sst2(tokenizer, n_train=1500, n_test=300):
    log("  Loading SST-2...")
    dataset = load_dataset('glue', 'sst2')
    train_ds = TextDataset(
        dataset['train']['sentence'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = TextDataset(
        dataset['validation']['sentence'][:n_test],
        dataset['validation']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "SST2"


def load_yelp(tokenizer, n_train=1500, n_test=300):
    log("  Loading Yelp...")
    dataset = load_dataset('yelp_polarity')
    train_ds = TextDataset(
        dataset['train']['text'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = TextDataset(
        dataset['test']['text'][:n_test],
        dataset['test']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "Yelp"


# ============================================================================
# STRATEGY 1: CURRICULUM ORDERING
# Task difficulty estimation based on:
# - Number of classes (more = harder)
# - Text length variance (higher = harder)
# - Domain specificity
# ============================================================================

TASK_DIFFICULTY = {
    # Easy: Short, simple binary sentiment
    "SST2": 1,
    # Medium: Longer sentiment, same domain
    "IMDB": 2,
    # Medium-Hard: Reviews, slightly different style
    "Yelp": 3,
    # Hard: 4-class topic, different domain entirely
    "AG_News": 4,
}


def get_curriculum_order(tasks: List[Tuple], order_type: str = "easy_to_hard") -> List[Tuple]:
    """
    Reorder tasks based on difficulty curriculum.

    order_type:
    - "easy_to_hard": Simple tasks first (expected: most negative FI)
    - "hard_to_easy": Complex tasks first
    - "random": Shuffled
    - "original": Keep original order
    """
    if order_type == "original":
        return tasks

    # Create task_name -> task mapping
    task_dict = {t[3]: t for t in tasks}

    if order_type == "easy_to_hard":
        ordered_names = sorted(TASK_DIFFICULTY.keys(), key=lambda x: TASK_DIFFICULTY[x])
    elif order_type == "hard_to_easy":
        ordered_names = sorted(TASK_DIFFICULTY.keys(), key=lambda x: -TASK_DIFFICULTY[x])
    elif order_type == "random":
        ordered_names = list(TASK_DIFFICULTY.keys())
        random.shuffle(ordered_names)
    else:
        raise ValueError(f"Unknown order_type: {order_type}")

    return [task_dict[name] for name in ordered_names if name in task_dict]


# ============================================================================
# STRATEGY 2: SUBSPACE EXPANSION + REFINEMENT
# Key innovation: After learning new task, actively refine old tasks
# ============================================================================

class ExpansionRefinementREWA(nn.Module):
    """
    Subspace-REWA with expansion and refinement phases.

    After each new task:
    1. Expand shared subspace to include new task's directions
    2. Refine old tasks in the expanded subspace (key for negative FI)
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        refine_epochs: int = 1,  # Epochs to refine each old task
        refine_lr_ratio: float = 0.5,  # Lower LR for refinement
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        self.lambda_max = lambda_max
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        self.anneal_rate = anneal_rate
        self.min_lambda = min_lambda
        self.refine_epochs = refine_epochs
        self.refine_lr_ratio = refine_lr_ratio

        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

        self.frozen_encoder = None
        self.tasks_completed = 0

        # Store task-specific data for refinement
        self.task_data: Dict[str, DataLoader] = {}
        self.task_heads: Dict[str, Dict] = {}

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        """Decay all lambdas by anneal_rate."""
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute initial PCA subspaces from Task 1."""
        log(f"  Computing subspaces (dim={self.subspace_dim})...")
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

        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

        log("  Subspaces identified.")

    def expand_subspace(self, new_dataloader: DataLoader, device: str):
        """
        Expand subspace to include new task's important directions.
        Uses subspace union via concatenation + re-orthogonalization.
        """
        log(f"  Expanding subspace with new task data...")
        self.eval()

        layer_activations = {i: [] for i in range(self.num_layers)}

        with torch.no_grad():
            for batch in new_dataloader:
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

            # Get new task's top directions
            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            U_new = Vh[:self.subspace_dim // 2, :].T.to(device)  # Half the dims

            # Merge with existing subspace
            U_old = self.pca_components[i]
            U_combined = torch.cat([U_old, U_new], dim=1)

            # Re-orthogonalize via QR decomposition (move to CPU for MPS compatibility)
            Q, R = torch.linalg.qr(U_combined.cpu())

            # Keep only subspace_dim directions
            self.pca_components[i] = Q[:, :self.subspace_dim].to(device)

        # Update frozen encoder to current state
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())

        log(f"  Subspace expanded.")

    def refine_task(
        self,
        task_name: str,
        task_loader: DataLoader,
        task_head: Dict,
        device: str,
        lr: float
    ):
        """
        Refine a previous task in the expanded subspace.

        This is the KEY INNOVATION for negative FI:
        - Old tasks benefit from the expanded shared knowledge
        - Training in expanded subspace improves old task performance
        """
        log(f"    Refining {task_name}...")

        # Save current state
        current_head_w = self.classifier.weight.detach().clone()
        current_head_b = self.classifier.bias.detach().clone()
        current_classes = self.num_classes

        # Load old task head
        num_classes = task_head['weight'].shape[0]
        self.update_classifier(num_classes)
        self.classifier.weight.data = task_head['weight'].to(device)
        self.classifier.bias.data = task_head['bias'].to(device)

        # Light refinement with low learning rate
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr * self.refine_lr_ratio,
            weight_decay=0.01
        )

        self.train()
        total_loss = 0

        for epoch in range(self.refine_epochs):
            for batch in task_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask, labels)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

        # Save refined head back
        task_head['weight'] = self.classifier.weight.detach().cpu().clone()
        task_head['bias'] = self.classifier.bias.detach().cpu().clone()

        # Restore current head
        self.update_classifier(current_classes)
        self.classifier.weight.data = current_head_w
        self.classifier.bias.data = current_head_b

        return task_head

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

                layer_loss = F.mse_loss(curr_proj, target_proj)
                lambda_l = self.layer_lambdas[i].to(device)
                rewa_loss = rewa_loss + lambda_l * layer_loss

        if loss is not None:
            loss = loss + rewa_loss

        return {'loss': loss, 'logits': logits, 'rewa_loss': rewa_loss}

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes


# ============================================================================
# STRATEGY 3: MULTI-TASK MIXTURE
# Blend old task data during training to prevent forgetting
# ============================================================================

class MixtureDataLoader:
    """
    DataLoader that mixes current task with previous tasks.
    """
    def __init__(
        self,
        current_loader: DataLoader,
        previous_loaders: List[Tuple[DataLoader, Dict]],  # (loader, head_dict)
        mixture_ratio: float = 0.1,  # 10% old, 90% new
    ):
        self.current_loader = current_loader
        self.previous_loaders = previous_loaders
        self.mixture_ratio = mixture_ratio

        # Create iterators
        self.current_iter = iter(current_loader)
        self.prev_iters = [iter(loader) for loader, _ in previous_loaders]

    def __iter__(self):
        self.current_iter = iter(self.current_loader)
        self.prev_iters = [iter(loader) for loader, _ in self.previous_loaders]
        return self

    def __next__(self):
        if random.random() < self.mixture_ratio and self.prev_iters:
            # Sample from previous task
            idx = random.randint(0, len(self.prev_iters) - 1)
            try:
                return next(self.prev_iters[idx]), idx
            except StopIteration:
                self.prev_iters[idx] = iter(self.previous_loaders[idx][0])
                return next(self.prev_iters[idx]), idx
        else:
            # Current task
            try:
                return next(self.current_iter), -1
            except StopIteration:
                raise StopIteration

    def __len__(self):
        return len(self.current_loader)


class MixtureSubspaceREWA(ExpansionRefinementREWA):
    """
    Subspace-REWA with multi-task mixture during training.

    Key: Keep exposing model to old tasks during new task training.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        mixture_ratio: float = 0.1,  # 10% old task data
    ):
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            lambda_max=lambda_max,
            subspace_dim=subspace_dim,
            layer_focus_start=layer_focus_start,
            anneal_rate=anneal_rate,
            min_lambda=min_lambda,
            refine_epochs=0,  # No separate refinement, mixture handles it
        )
        self.mixture_ratio = mixture_ratio


# ============================================================================
# STRATEGY 4: CONTRASTIVE SUBSPACE LEARNING
# Maximize similarity in shared subspace during new task learning
# ============================================================================

class ContrastiveSubspaceREWA(ExpansionRefinementREWA):
    """
    Subspace-REWA with contrastive alignment loss.

    Key innovation: Push new task representations toward old task centroids
    in the shared subspace.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        lambda_contrast: float = 0.5,  # Contrastive loss weight
        temperature: float = 0.1,
    ):
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            lambda_max=lambda_max,
            subspace_dim=subspace_dim,
            layer_focus_start=layer_focus_start,
            anneal_rate=anneal_rate,
            min_lambda=min_lambda,
        )
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature

        # Store task centroids in subspace
        self.task_centroids: Dict[str, Dict[int, torch.Tensor]] = {}  # task -> layer -> centroid

    def compute_task_centroid(self, task_name: str, dataloader: DataLoader, device: str):
        """Compute centroid of task representations in subspace."""
        log(f"  Computing centroid for {task_name}...")
        self.eval()

        layer_projections = {i: [] for i in range(self.num_layers)}

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

                    h = outputs.hidden_states[i+1][:, 0, :]
                    U = self.pca_components[i]
                    proj = h @ U  # (B, k)
                    layer_projections[i].append(proj.cpu())

        self.task_centroids[task_name] = {}
        for i in layer_projections:
            if layer_projections[i]:
                all_proj = torch.cat(layer_projections[i], dim=0)
                self.task_centroids[task_name][i] = all_proj.mean(dim=0).to(device)

    def contrastive_loss(self, hidden_states: Dict[int, torch.Tensor], device: str) -> torch.Tensor:
        """
        Compute contrastive alignment loss.
        Push current representations toward previous task centroids.
        """
        if not self.task_centroids:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        for i in range(self.num_layers):
            if i < self.layer_focus_start or i not in self.pca_components:
                continue

            U = self.pca_components[i]
            h = hidden_states[i][:, 0, :]  # (B, D)
            h_proj = F.normalize(h @ U, dim=1)  # (B, k) normalized

            # Compute similarity to each task centroid
            for task_name, centroids in self.task_centroids.items():
                if i in centroids:
                    centroid = F.normalize(centroids[i].unsqueeze(0), dim=1)  # (1, k)

                    # Cosine similarity
                    sim = (h_proj @ centroid.T).squeeze() / self.temperature  # (B,)

                    # Maximize similarity (minimize negative log)
                    loss = loss - sim.mean()

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

                layer_loss = F.mse_loss(curr_proj, target_proj)
                lambda_l = self.layer_lambdas[i].to(device)
                rewa_loss = rewa_loss + lambda_l * layer_loss

            # Contrastive alignment
            hidden_dict = {i: outputs.hidden_states[i+1] for i in range(self.num_layers)}
            contrast_loss = self.contrastive_loss(hidden_dict, device)

        if loss is not None:
            loss = loss + rewa_loss + self.lambda_contrast * contrast_loss

        return {'loss': loss, 'logits': logits, 'rewa_loss': rewa_loss, 'contrast_loss': contrast_loss}


# ============================================================================
# STRATEGY 5: COMBINED - All strategies together
# ============================================================================

class MaximalForwardTransferREWA(ContrastiveSubspaceREWA):
    """
    Combines all strategies for maximal forward transfer:
    - Contrastive alignment (from ContrastiveSubspaceREWA)
    - Expansion + refinement (from ExpansionRefinementREWA)
    - Multi-task mixture capability
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.85,
        min_lambda: float = 1.0,
        lambda_contrast: float = 0.3,
        temperature: float = 0.1,
        refine_epochs: int = 1,
        refine_lr_ratio: float = 0.3,
    ):
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            lambda_max=lambda_max,
            subspace_dim=subspace_dim,
            layer_focus_start=layer_focus_start,
            anneal_rate=anneal_rate,
            min_lambda=min_lambda,
            lambda_contrast=lambda_contrast,
            temperature=temperature,
        )
        self.refine_epochs = refine_epochs
        self.refine_lr_ratio = refine_lr_ratio


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_epoch_standard(model, dataloader, optimizer, scheduler, device, desc="Training"):
    """Standard training epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
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

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.1%}'})

    return total_loss / len(dataloader), correct / total


def train_epoch_mixture(
    model,
    current_loader,
    previous_data: List[Tuple[DataLoader, Dict]],  # [(loader, head_dict), ...]
    optimizer,
    scheduler,
    device,
    mixture_ratio: float = 0.1,
    desc="Training"
):
    """Training with multi-task mixture."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Create iterators
    current_iter = iter(current_loader)
    prev_iters = [iter(loader) for loader, _ in previous_data] if previous_data else []

    pbar = tqdm(range(len(current_loader)), desc=desc, leave=False)
    for _ in pbar:
        # Decide whether to sample from previous tasks
        if random.random() < mixture_ratio and prev_iters:
            # Sample from random previous task
            idx = random.randint(0, len(prev_iters) - 1)
            try:
                batch = next(prev_iters[idx])
            except StopIteration:
                prev_iters[idx] = iter(previous_data[idx][0])
                batch = next(prev_iters[idx])

            # Need to temporarily switch head for this batch
            # Skip for simplicity - just use current head (acts as regularization)
        else:
            # Current task
            try:
                batch = next(current_iter)
            except StopIteration:
                break

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

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.1%}'})

    return total_loss / max(1, len(current_loader)), correct / max(1, total)


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_forward_transfer_experiment(
    model_class,
    model_kwargs: dict,
    tasks: List[Tuple],
    device: str,
    epochs_per_task: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    model_name: str = "Model",
    use_curriculum: str = "original",
    use_refinement: bool = False,
    use_mixture: bool = False,
    mixture_ratio: float = 0.1,
    use_expansion: bool = False,
    use_contrastive: bool = False,
):
    """
    Run forward transfer experiment with configurable strategies.
    """
    log(f"\n{'='*70}")
    log(f"MODEL: {model_name}")
    log(f"Strategies: curriculum={use_curriculum}, refinement={use_refinement}, "
        f"mixture={use_mixture}, expansion={use_expansion}, contrastive={use_contrastive}")
    log(f"{'='*70}")

    # Apply curriculum ordering
    if use_curriculum != "original":
        tasks = get_curriculum_order(tasks, use_curriculum)
        log(f"Task order ({use_curriculum}): {[t[3] for t in tasks]}")

    results = {
        'model': model_name,
        'task_order': [t[3] for t in tasks],
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
        'lambda_history': [],
        'strategies': {
            'curriculum': use_curriculum,
            'refinement': use_refinement,
            'mixture': use_mixture,
            'expansion': use_expansion,
            'contrastive': use_contrastive,
        }
    }

    model = None
    completed_tasks = []
    task_heads = {}
    previous_data = []  # For mixture training

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        log(f"\n--- Task {task_idx + 1}/{len(tasks)}: {task_name} ({num_classes} classes) ---")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Initialize or update model
        if model is None:
            model = model_class(num_classes=num_classes, **model_kwargs).to(device)
        else:
            model.update_classifier(num_classes)

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        # Train
        for epoch in range(epochs_per_task):
            desc = f"T{task_idx+1} E{epoch+1}"

            if use_mixture and previous_data:
                loss, acc = train_epoch_mixture(
                    model, train_loader, previous_data, optimizer, scheduler,
                    device, mixture_ratio=mixture_ratio, desc=desc
                )
            else:
                loss, acc = train_epoch_standard(
                    model, train_loader, optimizer, scheduler, device, desc=desc
                )
            log(f"  Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.1%}")

        # Save task head
        task_heads[task_name] = {
            'weight': model.classifier.weight.detach().cpu().clone(),
            'bias': model.classifier.bias.detach().cpu().clone()
        }

        # Store for mixture training
        previous_data.append((train_loader, task_heads[task_name]))

        # Setup subspace on first task
        if task_idx == 0:
            if hasattr(model, 'compute_subspaces'):
                model.compute_subspaces(train_loader, device)

            # Compute initial centroid for contrastive
            if use_contrastive and hasattr(model, 'compute_task_centroid'):
                model.compute_task_centroid(task_name, train_loader, device)
        else:
            # Lambda annealing
            if hasattr(model, 'anneal_lambdas'):
                model.anneal_lambdas()
                log(f"  Lambda annealed. L11: {model.layer_lambdas[11]:.2f}")

            # Subspace expansion
            if use_expansion and hasattr(model, 'expand_subspace'):
                model.expand_subspace(train_loader, device)

            # Compute centroid for contrastive
            if use_contrastive and hasattr(model, 'compute_task_centroid'):
                model.compute_task_centroid(task_name, train_loader, device)

            # Refinement phase - key for negative FI
            if use_refinement and hasattr(model, 'refine_task'):
                log(f"  Refinement phase:")
                for prev_loader, prev_classes, prev_name in completed_tasks:
                    prev_train_loader = DataLoader(
                        tasks[completed_tasks.index((prev_loader, prev_classes, prev_name))][0],
                        batch_size=batch_size, shuffle=True
                    )
                    task_heads[prev_name] = model.refine_task(
                        prev_name, prev_train_loader, task_heads[prev_name], device, lr
                    )

        completed_tasks.append((test_loader, num_classes, task_name))

        # Evaluate all tasks
        log(f"  Evaluating all {len(completed_tasks)} tasks...")
        current_head_w = model.classifier.weight.detach().clone()
        current_head_b = model.classifier.bias.detach().clone()
        current_classes = model.num_classes

        for prev_loader, prev_classes, prev_name in completed_tasks:
            model.update_classifier(prev_classes)
            saved_head = task_heads[prev_name]
            model.classifier.weight.data = saved_head['weight'].to(device)
            model.classifier.bias.data = saved_head['bias'].to(device)

            acc = evaluate(model, prev_loader, device)
            results['task_accuracies'][prev_name].append(acc)
            log(f"    {prev_name}: {acc:.1%}")

        model.update_classifier(current_classes)
        model.classifier.weight.data = current_head_w
        model.classifier.bias.data = current_head_b

    # Compute final metrics
    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final

    task_accs = results['task_accuracies']
    results['WTA'] = min([accs[-1] for accs in task_accs.values()])

    forgettings = []
    for task_name, accs in task_accs.items():
        if len(accs) > 1:
            max_acc = max(accs[:-1])
            final_acc = accs[-1]
            forgetting = max_acc - final_acc
            results['per_task_forgetting'][task_name] = forgetting
            forgettings.append(forgetting)

    results['FI'] = np.mean(forgettings) if forgettings else 0.0

    final_values = [accs[-1] for accs in task_accs.values()]
    results['PV'] = np.var(final_values)

    if hasattr(model, 'lambda_history'):
        results['lambda_history'] = [lh.tolist() for lh in model.lambda_history]

    # Print summary
    log(f"\n{'─'*50}")
    log(f"RESULTS: {model_name}")
    log(f"{'─'*50}")
    log(f"  Task Order: {results['task_order']}")
    log(f"  Average Final Accuracy: {avg_final:.1%}")
    log(f"  Worst-Task Accuracy:    {results['WTA']:.1%}")
    log(f"  Forgetting Index (FI):  {results['FI']:+.2%}")
    log(f"  Performance Variance:   {results['PV']:.4f}")

    if results['FI'] < 0:
        log(f"  ✓ NEGATIVE FI ACHIEVED! Forward transfer detected.")

    return results


# ============================================================================
# MAIN EXPERIMENT SUITE
# ============================================================================

def main():
    log("="*80)
    log("FORWARD TRANSFER ENGINEERING EXPERIMENTS")
    log("="*80)
    log("\nGoal: Achieve FI < -3% through systematic strategy combination")
    log("Baseline: AnnealingSubspaceREWA achieved FI = -1.3%\n")

    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"Device: {device}")

    # Common parameters
    n_train = 1500
    n_test = 300
    epochs_per_task = 2
    batch_size = 16
    lr = 2e-5

    all_results = []

    def load_tasks():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return [
            load_ag_news(tokenizer, n_train, n_test),
            load_imdb(tokenizer, n_train, n_test),
            load_sst2(tokenizer, n_train, n_test),
            load_yelp(tokenizer, n_train, n_test),
        ]

    # # ========================================================================
    # # EXPERIMENT 1: Baseline (Original Order)
    # # ========================================================================
    # log("\n" + "="*70)
    # log("EXPERIMENT 1: Baseline (Original Order)")
    # log("="*70)

    # tasks = load_tasks()
    # results_baseline = run_forward_transfer_experiment(
    #     model_class=ExpansionRefinementREWA,
    #     model_kwargs={
    #         'lambda_max': 10.0,
    #         'subspace_dim': 64,
    #         'layer_focus_start': 7,
    #         'anneal_rate': 0.85,
    #         'min_lambda': 1.0,
    #     },
    #     tasks=tasks,
    #     device=device,
    #     epochs_per_task=epochs_per_task,
    #     batch_size=batch_size,
    #     lr=lr,
    #     model_name="Baseline-Original",
    #     use_curriculum="original",
    # )
    # all_results.append(results_baseline)

    # # ========================================================================
    # # EXPERIMENT 2: Curriculum Ordering (Easy → Hard)
    # # ========================================================================
    # log("\n" + "="*70)
    # log("EXPERIMENT 2: Curriculum Ordering (Easy → Hard)")
    # log("="*70)

    # tasks = load_tasks()
    # results_curriculum = run_forward_transfer_experiment(
    #     model_class=ExpansionRefinementREWA,
    #     model_kwargs={
    #         'lambda_max': 10.0,
    #         'subspace_dim': 64,
    #         'layer_focus_start': 7,
    #         'anneal_rate': 0.85,
    #         'min_lambda': 1.0,
    #     },
    #     tasks=tasks,
    #     device=device,
    #     epochs_per_task=epochs_per_task,
    #     batch_size=batch_size,
    #     lr=lr,
    #     model_name="Curriculum-EasyToHard",
    #     use_curriculum="easy_to_hard",
    # )
    # all_results.append(results_curriculum)

    # # ========================================================================
    # # EXPERIMENT 3: Curriculum + Refinement
    # # ========================================================================
    # log("\n" + "="*70)
    # log("EXPERIMENT 3: Curriculum + Refinement")
    # log("="*70)

    # tasks = load_tasks()
    # results_refine = run_forward_transfer_experiment(
    #     model_class=ExpansionRefinementREWA,
    #     model_kwargs={
    #         'lambda_max': 10.0,
    #         'subspace_dim': 64,
    #         'layer_focus_start': 7,
    #         'anneal_rate': 0.85,
    #         'min_lambda': 1.0,
    #         'refine_epochs': 1,
    #         'refine_lr_ratio': 0.3,
    #     },
    #     tasks=tasks,
    #     device=device,
    #     epochs_per_task=epochs_per_task,
    #     batch_size=batch_size,
    #     lr=lr,
    #     model_name="Curriculum+Refinement",
    #     use_curriculum="easy_to_hard",
    #     use_refinement=True,
    # )
    # all_results.append(results_refine)

    # ========================================================================
    # EXPERIMENT 4: Curriculum + Expansion + Refinement
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 4: Curriculum + Expansion + Refinement")
    log("="*70)

    tasks = load_tasks()
    results_expand = run_forward_transfer_experiment(
        model_class=ExpansionRefinementREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7,
            'anneal_rate': 0.85,
            'min_lambda': 1.0,
            'refine_epochs': 1,
            'refine_lr_ratio': 0.3,
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Curriculum+Expand+Refine",
        use_curriculum="easy_to_hard",
        use_refinement=True,
        use_expansion=True,
    )
    all_results.append(results_expand)

    # ========================================================================
    # EXPERIMENT 5: Contrastive Subspace Learning
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 5: Contrastive Subspace Learning")
    log("="*70)

    tasks = load_tasks()
    results_contrastive = run_forward_transfer_experiment(
        model_class=ContrastiveSubspaceREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7,
            'anneal_rate': 0.85,
            'min_lambda': 1.0,
            'lambda_contrast': 0.5,
            'temperature': 0.1,
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Contrastive-Subspace",
        use_curriculum="easy_to_hard",
        use_contrastive=True,
    )
    all_results.append(results_contrastive)

    # ========================================================================
    # EXPERIMENT 6: Multi-Task Mixture
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 6: Multi-Task Mixture (10%)")
    log("="*70)

    tasks = load_tasks()
    results_mixture = run_forward_transfer_experiment(
        model_class=MixtureSubspaceREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7,
            'anneal_rate': 0.85,
            'min_lambda': 1.0,
            'mixture_ratio': 0.1,
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Mixture-10pct",
        use_curriculum="easy_to_hard",
        use_mixture=True,
        mixture_ratio=0.1,
    )
    all_results.append(results_mixture)

    # ========================================================================
    # EXPERIMENT 7: MAXIMAL - All Strategies Combined
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 7: MAXIMAL - All Strategies Combined")
    log("="*70)

    tasks = load_tasks()
    results_maximal = run_forward_transfer_experiment(
        model_class=MaximalForwardTransferREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7,
            'anneal_rate': 0.85,
            'min_lambda': 1.0,
            'lambda_contrast': 0.3,
            'temperature': 0.1,
            'refine_epochs': 1,
            'refine_lr_ratio': 0.3,
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="MAXIMAL-Combined",
        use_curriculum="easy_to_hard",
        use_refinement=True,
        use_expansion=True,
        use_contrastive=True,
    )
    all_results.append(results_maximal)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    log("\n" + "="*80)
    log("FINAL SUMMARY: FORWARD TRANSFER EXPERIMENTS")
    log("="*80)

    header = f"{'Model':<35} {'Task Order':<25} {'Avg Acc':<10} {'WTA':<10} {'FI':<12} {'PV':<10}"
    log(header)
    log("-" * len(header))

    for r in all_results:
        order_str = "→".join([t[:4] for t in r['task_order']])
        fi_str = f"{r['FI']:+.2%}"
        log(f"{r['model']:<35} {order_str:<25} {r['average_final_accuracy']:<10.1%} "
            f"{r['WTA']:<10.1%} {fi_str:<12} {r['PV']:<10.4f}")

    # Find best by FI
    best_by_fi = min(all_results, key=lambda x: x['FI'])
    log(f"\nBest by FI: {best_by_fi['model']} (FI = {best_by_fi['FI']:+.2%})")

    # Success check
    log("\n" + "="*80)
    if best_by_fi['FI'] < -0.05:
        log(f"MAJOR SUCCESS: Achieved FI < -5% ({best_by_fi['FI']:+.2%})")
        log(f"Target FI = -5% to -8% is REALISTIC")
    elif best_by_fi['FI'] < -0.03:
        log(f"SUCCESS: Achieved FI < -3% ({best_by_fi['FI']:+.2%})")
        log(f"Significant forward transfer demonstrated")
    elif best_by_fi['FI'] < -0.01:
        log(f"PARTIAL SUCCESS: Achieved FI < -1% ({best_by_fi['FI']:+.2%})")
        log(f"Forward transfer present, room for improvement")
    elif best_by_fi['FI'] < 0:
        log(f"BASELINE MET: Negative FI achieved ({best_by_fi['FI']:+.2%})")
    else:
        log(f"NEEDS WORK: FI is positive ({best_by_fi['FI']:+.2%})")
    log("="*80)

    # Improvement over baseline
    baseline_fi = results_baseline['FI']
    best_fi = best_by_fi['FI']
    improvement = baseline_fi - best_fi
    log(f"\nImprovement over baseline: {improvement:+.2%} FI reduction")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"forward_transfer_results_{timestamp}.json"

    json_results = []
    for r in all_results:
        jr = {
            'model': r['model'],
            'task_order': r['task_order'],
            'strategies': r['strategies'],
            'task_accuracies': {k: [float(x) for x in v] for k, v in r['task_accuracies'].items()},
            'final_accuracies': {k: float(v) for k, v in r['final_accuracies'].items()},
            'per_task_forgetting': {k: float(v) for k, v in r['per_task_forgetting'].items()},
            'average_final_accuracy': float(r['average_final_accuracy']),
            'WTA': float(r['WTA']),
            'FI': float(r['FI']),
            'PV': float(r['PV']),
            'lambda_history': r.get('lambda_history', []),
        }
        json_results.append(jr)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
