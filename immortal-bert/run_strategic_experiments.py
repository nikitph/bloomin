#!/usr/bin/env python3
"""
Strategic Experiments for Subspace-REWA (Point 8 from Analysis)

Implements 4 high-leverage experiments:
1. Adaptive subspace dimensionality - grow d_s when FI starts increasing
2. Layer-wise lambda annealing - decay lambda after task convergence
3. Subspace intersection across tasks - track how shared manifold evolves
4. Representation drift curves - plot ||h_l^(s) - h_{l,frozen}^(s)||
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.rewa_bert import evaluate


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def train_epoch_with_progress(model, dataloader, optimizer, scheduler, device, desc="Training"):
    """Train for one epoch with progress bar."""
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


def load_ag_news(tokenizer, n_train=2000, n_test=500):
    log("Loading AG News...")
    dataset = load_dataset('ag_news')
    texts_train = dataset['train']['text'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['test']['text'][:n_test]
    labels_test = dataset['test']['label'][:n_test]
    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)
    return train_ds, test_ds, 4, "AG_News"


def load_imdb(tokenizer, n_train=2000, n_test=500):
    log("Loading IMDB...")
    dataset = load_dataset('imdb')
    texts_train = dataset['train']['text'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['test']['text'][:n_test]
    labels_test = dataset['test']['label'][:n_test]
    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)
    return train_ds, test_ds, 2, "IMDB"


def load_sst2(tokenizer, n_train=2000, n_test=500):
    log("Loading SST-2...")
    dataset = load_dataset('glue', 'sst2')
    texts_train = dataset['train']['sentence'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['validation']['sentence'][:n_test]
    labels_test = dataset['validation']['label'][:n_test]
    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)
    return train_ds, test_ds, 2, "SST2"


def load_yelp(tokenizer, n_train=2000, n_test=500):
    log("Loading Yelp...")
    dataset = load_dataset('yelp_polarity')
    texts_train = dataset['train']['text'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['test']['text'][:n_test]
    labels_test = dataset['test']['label'][:n_test]
    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)
    return train_ds, test_ds, 2, "Yelp"


# ============================================================================
# EXPERIMENT 1: ADAPTIVE SUBSPACE DIMENSIONALITY
# ============================================================================

@dataclass
class AdaptiveSubspaceConfig:
    """Configuration for adaptive subspace growth."""
    initial_dim: int = 32
    max_dim: int = 128
    growth_step: int = 16
    fi_threshold: float = 0.02  # FI increase threshold to trigger growth


class AdaptiveSubspaceREWA(nn.Module):
    """
    Subspace-REWA with adaptive dimensionality.

    Key innovation: Monitor forgetting index during training and grow
    subspace dimension when FI starts increasing beyond threshold.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        config: AdaptiveSubspaceConfig = None,
        layer_focus_start: int = 7
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
        self.layer_focus_start = layer_focus_start
        self.adaptive_config = config or AdaptiveSubspaceConfig()

        # Track current subspace dimension per layer
        self.current_dims: Dict[int, int] = {}
        self.pca_components: Dict[int, torch.Tensor] = {}
        self.pca_full_components: Dict[int, torch.Tensor] = {}  # Store full PCA for expansion
        self.subspace_identified = False

        # Tracking metrics
        self.fi_history: List[float] = []
        self.dim_history: List[Dict[int, int]] = []

        self.layer_lambdas = self._compute_lambda_schedule()
        self.frozen_encoder = None

    def _compute_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces and store full components for later expansion."""
        print(f"Computing adaptive subspaces (initial dim={self.adaptive_config.initial_dim})...")
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

            # Store full components (up to max_dim)
            max_k = min(self.adaptive_config.max_dim, Vh.shape[0])
            self.pca_full_components[i] = Vh[:max_k, :].T.to(device)  # (D, max_dim)

            # Initialize with initial_dim
            init_dim = self.adaptive_config.initial_dim
            self.pca_components[i] = self.pca_full_components[i][:, :init_dim]  # (D, init_dim)
            self.current_dims[i] = init_dim

        self.subspace_identified = True

        # Freeze encoder snapshot
        self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)

        print(f"Subspaces initialized: dims={list(self.current_dims.values())[:3]}...")

    def expand_subspace(self, layer_idx: int):
        """Expand subspace dimension for a specific layer."""
        if layer_idx not in self.current_dims:
            return False

        current = self.current_dims[layer_idx]
        new_dim = min(current + self.adaptive_config.growth_step,
                      self.adaptive_config.max_dim)

        if new_dim > current:
            self.pca_components[layer_idx] = self.pca_full_components[layer_idx][:, :new_dim]
            self.current_dims[layer_idx] = new_dim
            return True
        return False

    def check_and_adapt(self, current_fi: float):
        """Check if FI is increasing and adapt subspace if needed."""
        self.fi_history.append(current_fi)

        if len(self.fi_history) < 2:
            return False

        fi_increase = current_fi - self.fi_history[-2]

        if fi_increase > self.adaptive_config.fi_threshold:
            print(f"  FI increased by {fi_increase:.3f} > threshold. Expanding subspaces...")
            expanded = False
            for layer_idx in self.current_dims:
                if self.expand_subspace(layer_idx):
                    expanded = True
            if expanded:
                print(f"  New dims: {list(self.current_dims.values())[:3]}...")
            return expanded
        return False

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
# EXPERIMENT 2: LAYER-WISE LAMBDA ANNEALING
# ============================================================================

class AnnealingSubspaceREWA(nn.Module):
    """
    Subspace-REWA with layer-wise lambda annealing.

    Key innovation: Decay lambda after task convergence to allow
    more plasticity for new tasks while maintaining protection.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7,
        anneal_rate: float = 0.8,  # Multiply lambda by this after each task
        min_lambda: float = 1.0    # Floor for lambda
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

        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False

        # Per-layer lambda tracking with history
        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

        self.frozen_encoder = None
        self.tasks_completed = 0

    def _compute_initial_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def anneal_lambdas(self):
        """Decay all lambdas by anneal_rate (called after task convergence)."""
        for i in range(self.num_layers):
            self.layer_lambdas[i] = max(
                self.layer_lambdas[i] * self.anneal_rate,
                self.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        self.lambda_history.append(self.layer_lambdas.clone())
        self.tasks_completed += 1
        print(f"  Lambdas annealed (task {self.tasks_completed}). L11 lambda: {self.layer_lambdas[10]:.2f}")

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces."""
        print(f"Computing subspaces (dim={self.subspace_dim})...")
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

        print("Subspaces identified.")

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
# EXPERIMENT 3: SUBSPACE INTERSECTION TRACKING
# ============================================================================

@dataclass
class SubspaceAnalysis:
    """Tracks subspace evolution across tasks."""
    task_name: str
    layer_subspaces: Dict[int, torch.Tensor]  # layer -> U matrix
    layer_singular_values: Dict[int, torch.Tensor]

    # Intersection metrics with previous tasks
    intersection_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    principal_angles: Dict[str, Dict[int, List[float]]] = field(default_factory=dict)


class SubspaceTrackingREWA(nn.Module):
    """
    Subspace-REWA with full subspace intersection tracking.

    Tracks how the shared manifold evolves across tasks:
    - Computes principal angles between task subspaces
    - Measures subspace intersection/overlap
    - Identifies stable vs. task-specific dimensions
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7
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

        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False

        # Tracking across tasks
        self.task_subspaces: List[SubspaceAnalysis] = []

        self.layer_lambdas = self._compute_lambda_schedule()
        self.frozen_encoder = None

    def _compute_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    @staticmethod
    def compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
        """
        Compute principal angles between two subspaces.

        U1, U2: (D, k) orthonormal basis matrices
        Returns: tensor of angles in radians
        """
        # SVD of U1^T @ U2 gives cos(principal angles)
        M = U1.T @ U2
        _, S, _ = torch.linalg.svd(M)
        # Clamp to valid range for arccos
        S = torch.clamp(S, -1.0, 1.0)
        angles = torch.acos(S)
        return angles

    @staticmethod
    def compute_subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
        """
        Compute overlap/intersection score between subspaces.

        Returns: overlap in [0, 1], where 1 = identical subspaces
        """
        angles = SubspaceTrackingREWA.compute_principal_angles(U1, U2)
        # Average cosine of principal angles
        overlap = torch.cos(angles).mean().item()
        return overlap

    def compute_subspaces_with_tracking(self, dataloader: DataLoader, device: str, task_name: str):
        """Compute PCA subspaces and track evolution."""
        print(f"Computing and tracking subspaces for {task_name}...")
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

        task_subspaces = {}
        task_singular_values = {}

        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:self.subspace_dim, :]
            U = components.T  # (D, k)

            task_subspaces[i] = U
            task_singular_values[i] = S[:self.subspace_dim]

            # Only update PCA on first task
            if not self.subspace_identified:
                self.pca_components[i] = U.to(device)

        # Compute intersection with all previous tasks
        analysis = SubspaceAnalysis(
            task_name=task_name,
            layer_subspaces=task_subspaces,
            layer_singular_values=task_singular_values
        )

        for prev_analysis in self.task_subspaces:
            analysis.intersection_scores[prev_analysis.task_name] = {}
            analysis.principal_angles[prev_analysis.task_name] = {}

            for layer_idx in task_subspaces:
                if layer_idx in prev_analysis.layer_subspaces:
                    U_prev = prev_analysis.layer_subspaces[layer_idx]
                    U_curr = task_subspaces[layer_idx]

                    overlap = self.compute_subspace_overlap(U_prev, U_curr)
                    angles = self.compute_principal_angles(U_prev, U_curr)

                    analysis.intersection_scores[prev_analysis.task_name][layer_idx] = overlap
                    analysis.principal_angles[prev_analysis.task_name][layer_idx] = angles.tolist()

        self.task_subspaces.append(analysis)

        if not self.subspace_identified:
            self.subspace_identified = True
            self.frozen_encoder = BertModel.from_pretrained(self.config._name_or_path)
            self.frozen_encoder.load_state_dict(self.encoder.state_dict())
            self.frozen_encoder.eval()
            for param in self.frozen_encoder.parameters():
                param.requires_grad = False
            self.frozen_encoder.to(device)

        # Print intersection analysis
        if len(self.task_subspaces) > 1:
            print(f"\n  Subspace intersection with previous tasks:")
            for prev_name, scores in analysis.intersection_scores.items():
                avg_overlap = np.mean(list(scores.values()))
                print(f"    vs {prev_name}: avg overlap = {avg_overlap:.3f}")

        return analysis

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
# EXPERIMENT 4: REPRESENTATION DRIFT CURVES
# ============================================================================

@dataclass
class DriftMeasurement:
    """Single drift measurement."""
    task_name: str
    epoch: int
    layer_drifts: Dict[int, float]  # layer -> ||h - h_frozen||
    subspace_drifts: Dict[int, float]  # layer -> ||U'h - U'h_frozen||
    null_drifts: Dict[int, float]  # layer -> drift in null space


class DriftTrackingREWA(nn.Module):
    """
    Subspace-REWA with comprehensive drift tracking.

    Tracks:
    - Full representation drift: ||h - h_frozen||
    - Subspace drift: ||U'h - U'h_frozen|| (what we're protecting)
    - Null space drift: drift in directions we allow to change
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 7
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

        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False

        # Drift tracking
        self.drift_history: List[DriftMeasurement] = []
        self.reference_loader = None  # Small loader for drift measurement

        self.layer_lambdas = self._compute_lambda_schedule()
        self.frozen_encoder = None

    def _compute_lambda_schedule(self) -> torch.Tensor:
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
        return lambdas

    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """Compute PCA subspaces and store reference loader."""
        print(f"Computing subspaces (dim={self.subspace_dim})...")
        self.eval()
        self.to(device)

        # Store first few batches as reference for drift measurement
        self.reference_loader = dataloader

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

        print("Subspaces identified.")

    def measure_drift(self, device: str, task_name: str, epoch: int) -> DriftMeasurement:
        """Measure representation drift across all layers."""
        if self.reference_loader is None or self.frozen_encoder is None:
            return None

        self.eval()

        layer_drifts = {i: 0.0 for i in range(self.num_layers)}
        subspace_drifts = {i: 0.0 for i in range(self.num_layers)}
        null_drifts = {i: 0.0 for i in range(self.num_layers)}
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.reference_loader):
                if batch_idx >= 5:  # Only use first 5 batches
                    break

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Current model
                curr_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Frozen model
                frozen_outputs = self.frozen_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                batch_size = input_ids.shape[0]
                count += batch_size

                for i in range(self.num_layers):
                    curr_h = curr_outputs.hidden_states[i+1][:, 0, :]  # (B, D)
                    frozen_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                    # Full drift
                    full_diff = curr_h - frozen_h
                    layer_drifts[i] += full_diff.norm(dim=1).sum().item()

                    # Subspace drift (if we have PCA for this layer)
                    if i in self.pca_components:
                        U = self.pca_components[i]  # (D, k)

                        # Project to subspace
                        curr_proj = curr_h @ U  # (B, k)
                        frozen_proj = frozen_h @ U
                        subspace_diff = curr_proj - frozen_proj
                        subspace_drifts[i] += subspace_diff.norm(dim=1).sum().item()

                        # Null space drift: ||h - UU'h||
                        # Project to null space: (I - UU')h
                        curr_null = curr_h - (curr_proj @ U.T)
                        frozen_null = frozen_h - (frozen_proj @ U.T)
                        null_diff = curr_null - frozen_null
                        null_drifts[i] += null_diff.norm(dim=1).sum().item()

        # Average
        for i in range(self.num_layers):
            layer_drifts[i] /= count
            subspace_drifts[i] /= count
            null_drifts[i] /= count

        measurement = DriftMeasurement(
            task_name=task_name,
            epoch=epoch,
            layer_drifts=layer_drifts,
            subspace_drifts=subspace_drifts,
            null_drifts=null_drifts
        )
        self.drift_history.append(measurement)

        return measurement

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
# EXPERIMENT RUNNERS
# ============================================================================

def run_experiment_with_tracking(
    model_class,
    model_kwargs,
    tasks,
    device,
    epochs_per_task=2,
    batch_size=16,
    lr=2e-5,
    model_name="Model",
    track_drift=False,
    track_subspace=False,
    anneal_lambda=False
):
    """Run continual learning with optional tracking features."""
    log(f"\n{'='*60}")
    log(f"Running: {model_name}")
    log(f"{'='*60}")

    results = {
        'model': model_name,
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'drift_measurements': [],
        'subspace_analyses': [],
        'lambda_history': [],
    }

    model = None
    completed_tasks = []
    task_heads = {}

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        log(f"\n--- Task {task_idx + 1}/{len(tasks)}: {task_name} ({num_classes} classes) ---")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        if model is None:
            log(f"  Initializing model...")
            model = model_class(num_classes=num_classes, **model_kwargs).to(device)
            log(f"  Model ready on {device}")
        else:
            model.update_classifier(num_classes)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        # Train
        for epoch in range(epochs_per_task):
            desc = f"T{task_idx+1} E{epoch+1}"
            loss, acc = train_epoch_with_progress(model, train_loader, optimizer, scheduler, device, desc=desc)
            log(f"  Epoch {epoch + 1}/{epochs_per_task}: Loss={loss:.4f}, Acc={acc:.1%}")

            # Track drift if enabled
            if track_drift and hasattr(model, 'measure_drift') and model.subspace_identified:
                drift = model.measure_drift(device, task_name, epoch)
                if drift:
                    # Print summary
                    avg_subspace_drift = np.mean([v for k, v in drift.subspace_drifts.items()
                                                   if k >= model.layer_focus_start])
                    avg_null_drift = np.mean([v for k, v in drift.null_drifts.items()
                                              if k >= model.layer_focus_start])
                    log(f"    Drift - Subspace: {avg_subspace_drift:.4f}, Null: {avg_null_drift:.4f}")

        # Save head
        task_heads[task_name] = {
            'weight': model.classifier.weight.detach().cpu().clone(),
            'bias': model.classifier.bias.detach().cpu().clone()
        }

        # Setup subspace (Task 1 only for most experiments)
        if task_idx == 0:
            if track_subspace and hasattr(model, 'compute_subspaces_with_tracking'):
                analysis = model.compute_subspaces_with_tracking(train_loader, device, task_name)
                results['subspace_analyses'].append({
                    'task': task_name,
                    'intersections': {k: dict(v) for k, v in analysis.intersection_scores.items()}
                })
            elif hasattr(model, 'compute_subspaces'):
                model.compute_subspaces(train_loader, device)
        elif track_subspace and hasattr(model, 'compute_subspaces_with_tracking'):
            # Track subspace evolution for later tasks too
            analysis = model.compute_subspaces_with_tracking(train_loader, device, task_name)
            results['subspace_analyses'].append({
                'task': task_name,
                'intersections': {k: dict(v) for k, v in analysis.intersection_scores.items()}
            })

        # Lambda annealing after task convergence
        if anneal_lambda and hasattr(model, 'anneal_lambdas') and task_idx > 0:
            model.anneal_lambdas()
            if hasattr(model, 'lambda_history'):
                results['lambda_history'] = [lh.tolist() for lh in model.lambda_history]

        # Adaptive dimension check
        if hasattr(model, 'check_and_adapt'):
            # Compute current FI
            current_fi = compute_current_fi(results['task_accuracies'])
            model.check_and_adapt(current_fi)

        completed_tasks.append((test_loader, num_classes, task_name))

        # Evaluate all tasks
        log(f"  Evaluating on all tasks...")
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

    # Final metrics
    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final

    task_accs = results['task_accuracies']
    results['WTA'] = min([accs[-1] for accs in task_accs.values()])

    forgettings = []
    for accs in task_accs.values():
        if len(accs) > 1:
            forgettings.append(max(accs[:-1]) - accs[-1])
    results['FI'] = np.mean(forgettings) if forgettings else 0.0

    final_values = [accs[-1] for accs in task_accs.values()]
    results['PV'] = np.var(final_values)

    # Get drift data if available
    if track_drift and hasattr(model, 'drift_history'):
        results['drift_measurements'] = [
            {
                'task': d.task_name,
                'epoch': d.epoch,
                'layer_drifts': d.layer_drifts,
                'subspace_drifts': d.subspace_drifts,
                'null_drifts': d.null_drifts
            }
            for d in model.drift_history
        ]

    log(f"\n  Average final accuracy: {avg_final:.1%}")
    log(f"  Worst-Task Accuracy:    {results['WTA']:.1%}")
    log(f"  Forgetting Index:       {results['FI']:.1%}")
    log(f"  Performance Variance:   {results['PV']:.4f}")

    return results, model


def compute_current_fi(task_accuracies: Dict[str, List[float]]) -> float:
    """Compute current forgetting index from task accuracies."""
    forgettings = []
    for accs in task_accuracies.values():
        if len(accs) > 1:
            forgettings.append(max(accs[:-1]) - accs[-1])
    return np.mean(forgettings) if forgettings else 0.0


def plot_drift_curves(drift_measurements: List[dict], save_path: str = None):
    """Plot representation drift curves."""
    if not drift_measurements:
        log("No drift measurements to plot.")
        return

    # Organize by layer
    layers = sorted(drift_measurements[0]['subspace_drifts'].keys())
    focus_layers = [l for l in layers if l >= 7]  # Focus on layers 7-11

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create x-axis (measurement index)
    x = range(len(drift_measurements))
    task_boundaries = []
    current_task = None
    for i, m in enumerate(drift_measurements):
        if m['task'] != current_task:
            task_boundaries.append(i)
            current_task = m['task']

    # Plot 1: Full drift
    ax1 = axes[0]
    for layer in focus_layers:
        drifts = [m['layer_drifts'][layer] for m in drift_measurements]
        ax1.plot(x, drifts, label=f'L{layer}', alpha=0.7)
    ax1.set_title('Full Representation Drift')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('||h - h_frozen||')
    ax1.legend()
    for tb in task_boundaries[1:]:
        ax1.axvline(x=tb, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Subspace drift
    ax2 = axes[1]
    for layer in focus_layers:
        drifts = [m['subspace_drifts'][layer] for m in drift_measurements]
        ax2.plot(x, drifts, label=f'L{layer}', alpha=0.7)
    ax2.set_title('Subspace Drift (Protected)')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel("||U'h - U'h_frozen||")
    ax2.legend()
    for tb in task_boundaries[1:]:
        ax2.axvline(x=tb, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Null space drift
    ax3 = axes[2]
    for layer in focus_layers:
        drifts = [m['null_drifts'][layer] for m in drift_measurements]
        ax3.plot(x, drifts, label=f'L{layer}', alpha=0.7)
    ax3.set_title('Null Space Drift (Allowed)')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Null space drift')
    ax3.legend()
    for tb in task_boundaries[1:]:
        ax3.axvline(x=tb, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"Drift curves saved to {save_path}")
    plt.close()


def plot_subspace_evolution(subspace_analyses: List[dict], save_path: str = None):
    """Plot subspace intersection evolution."""
    if len(subspace_analyses) < 2:
        log("Need at least 2 tasks for subspace evolution plot.")
        return

    # Extract intersection data
    tasks = [a['task'] for a in subspace_analyses]

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each task after the first, plot intersection with Task 1
    for i, analysis in enumerate(subspace_analyses[1:], start=1):
        if 'AG_News' in analysis['intersections']:
            overlaps = analysis['intersections']['AG_News']
            layers = sorted(overlaps.keys())
            values = [overlaps[l] for l in layers]
            ax.plot(layers, values, 'o-', label=f'{tasks[i]} vs AG_News', markersize=8)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Subspace Overlap')
    ax.set_title('Subspace Intersection with Task 1 (AG_News)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"Subspace evolution plot saved to {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    log("="*70)
    log("STRATEGIC EXPERIMENTS: Subspace-REWA Deep Analysis")
    log("="*70)

    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"\nDevice: {device}")

    # Load datasets
    n_train = 1500
    n_test = 300

    log("\nPreparing tasks for BERT models...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tasks = [
        load_ag_news(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_sst2(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
    ]

    epochs_per_task = 2
    batch_size = 16
    lr = 2e-5

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================================
    # EXPERIMENT 1: Adaptive Subspace Dimensionality
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 1: Adaptive Subspace Dimensionality")
    log("="*70)

    adaptive_config = AdaptiveSubspaceConfig(
        initial_dim=32,
        max_dim=128,
        growth_step=16,
        fi_threshold=0.02
    )

    adaptive_results, adaptive_model = run_experiment_with_tracking(
        model_class=AdaptiveSubspaceREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'config': adaptive_config,
            'layer_focus_start': 7
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Adaptive-Subspace-REWA"
    )
    adaptive_results['dim_history'] = [
        {k: v for k, v in adaptive_model.current_dims.items()}
    ]
    all_results.append(adaptive_results)

    # Reload tasks for next experiment
    tasks = [
        load_ag_news(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_sst2(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
    ]

    # ========================================================================
    # EXPERIMENT 2: Lambda Annealing
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 2: Layer-wise Lambda Annealing")
    log("="*70)

    annealing_results, annealing_model = run_experiment_with_tracking(
        model_class=AnnealingSubspaceREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7,
            'anneal_rate': 0.8,
            'min_lambda': 1.0
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Annealing-Subspace-REWA",
        anneal_lambda=True
    )
    all_results.append(annealing_results)

    # Reload tasks
    tasks = [
        load_ag_news(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_sst2(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
    ]

    # ========================================================================
    # EXPERIMENT 3: Subspace Intersection Tracking
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 3: Subspace Intersection Tracking")
    log("="*70)

    tracking_results, tracking_model = run_experiment_with_tracking(
        model_class=SubspaceTrackingREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Tracking-Subspace-REWA",
        track_subspace=True
    )
    all_results.append(tracking_results)

    # Plot subspace evolution
    if tracking_results['subspace_analyses']:
        plot_subspace_evolution(
            tracking_results['subspace_analyses'],
            f"subspace_evolution_{timestamp}.png"
        )

    # Reload tasks
    tasks = [
        load_ag_news(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_sst2(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
    ]

    # ========================================================================
    # EXPERIMENT 4: Representation Drift Curves
    # ========================================================================
    log("\n" + "="*70)
    log("EXPERIMENT 4: Representation Drift Curves")
    log("="*70)

    drift_results, drift_model = run_experiment_with_tracking(
        model_class=DriftTrackingREWA,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7
        },
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Drift-Tracking-REWA",
        track_drift=True
    )
    all_results.append(drift_results)

    # Plot drift curves
    if drift_results['drift_measurements']:
        plot_drift_curves(
            drift_results['drift_measurements'],
            f"drift_curves_{timestamp}.png"
        )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    log("\n" + "="*80)
    log("STRATEGIC EXPERIMENTS SUMMARY")
    log("="*80)

    header = f"{'Experiment':<30} {'Avg Acc':<10} {'WTA':<10} {'FI':<10} {'PV':<10}"
    log(header)
    log("-" * len(header))

    for r in all_results:
        log(f"{r['model']:<30} {r['average_final_accuracy']:<10.1%} "
            f"{r['WTA']:<10.1%} {r['FI']:<10.1%} {r['PV']:<10.4f}")

    # Save results
    results_file = f"strategic_results_{timestamp}.json"

    json_results = []
    for r in all_results:
        jr = r.copy()
        jr['task_accuracies'] = {k: [float(x) for x in v] for k, v in r['task_accuracies'].items()}
        jr['final_accuracies'] = {k: float(v) for k, v in r['final_accuracies'].items()}
        jr['average_final_accuracy'] = float(r['average_final_accuracy'])
        jr['WTA'] = float(r['WTA'])
        jr['FI'] = float(r['FI'])
        jr['PV'] = float(r['PV'])
        json_results.append(jr)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
