#!/usr/bin/env python3
"""
Multi-Sheaf + Maximal Adversarial Combined Benchmark (v2)

Combines:
1. Sheaf-aware task grouping (semantic compatibility)
2. Centroid attraction contrastive loss (from forward transfer experiments)
3. Subspace expansion + refinement
4. Per-sheaf FI computation

Key change from v1: Uses cosine similarity centroid attraction instead of KL divergence.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

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
from tqdm import tqdm
from dataclasses import dataclass, field

from src.rewa_bert import evaluate


def log(msg):
    print(msg, flush=True)


# ============================================================================
# SEMANTIC SHEAF DEFINITIONS
# ============================================================================

@dataclass
class AdversarialSheaf:
    """A semantic sheaf with contrastive training components."""
    name: str
    task_names: List[str]
    invariant_type: str

    # Hyperparameters
    subspace_dim: int = 64
    lambda_max: float = 10.0
    lambda_contrast: float = 0.3
    temperature: float = 0.1
    anneal_rate: float = 0.85
    min_lambda: float = 1.0

    # State
    pca_components: Dict[int, torch.Tensor] = field(default_factory=dict)
    layer_lambdas: Optional[torch.Tensor] = None
    tasks_completed: int = 0
    task_heads: Dict = field(default_factory=dict)
    task_train_datasets: Dict = field(default_factory=dict)
    task_centroids: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)  # task_name -> {layer -> centroid}
    initialized: bool = False


ADVERSARIAL_SHEAVES = {
    "sentiment": AdversarialSheaf(
        name="Sentiment",
        task_names=["SST2", "IMDB", "Yelp"],
        invariant_type="polarity",
        lambda_max=10.0,
        lambda_contrast=0.3,
        temperature=0.1,
        anneal_rate=0.9,
    ),
    "paraphrase": AdversarialSheaf(
        name="Paraphrase",
        task_names=["QQP", "MRPC"],
        invariant_type="symmetry",
        lambda_max=10.0,
        lambda_contrast=0.3,
        temperature=0.1,
        anneal_rate=0.88,
    ),
    "entailment": AdversarialSheaf(
        name="Entailment",
        task_names=["RTE", "QNLI", "MNLI"],
        invariant_type="directionality",
        lambda_max=8.0,
        lambda_contrast=0.25,
        temperature=0.1,
        anneal_rate=0.85,
    ),
    "syntax": AdversarialSheaf(
        name="Syntax",
        task_names=["CoLA"],
        invariant_type="grammar",
        lambda_max=5.0,
        lambda_contrast=0.2,
        anneal_rate=0.9,
    ),
    "topic": AdversarialSheaf(
        name="Topic",
        task_names=["AG_News"],
        invariant_type="clustering",
        lambda_max=5.0,
        lambda_contrast=0.2,
        anneal_rate=0.9,
    ),
}


def get_task_sheaf(task_name: str) -> Optional[AdversarialSheaf]:
    for sheaf in ADVERSARIAL_SHEAVES.values():
        if task_name in sheaf.task_names:
            return sheaf
    return None


# ============================================================================
# DATA LOADING
# ============================================================================

class TextDataset(Dataset):
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


class PairTextDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=128):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts1[idx],
            self.texts2[idx],
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


def load_qqp(tokenizer, n_train=1500, n_test=300):
    log("  Loading QQP...")
    dataset = load_dataset('glue', 'qqp')
    train_ds = PairTextDataset(
        dataset['train']['question1'][:n_train],
        dataset['train']['question2'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = PairTextDataset(
        dataset['validation']['question1'][:n_test],
        dataset['validation']['question2'][:n_test],
        dataset['validation']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "QQP"


def load_mrpc(tokenizer, n_train=1500, n_test=300):
    log("  Loading MRPC...")
    dataset = load_dataset('glue', 'mrpc')
    actual_train = min(n_train, len(dataset['train']))
    actual_test = min(n_test, len(dataset['validation']))
    train_ds = PairTextDataset(
        dataset['train']['sentence1'][:actual_train],
        dataset['train']['sentence2'][:actual_train],
        dataset['train']['label'][:actual_train],
        tokenizer
    )
    test_ds = PairTextDataset(
        dataset['validation']['sentence1'][:actual_test],
        dataset['validation']['sentence2'][:actual_test],
        dataset['validation']['label'][:actual_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "MRPC"


def load_rte(tokenizer, n_train=1500, n_test=300):
    log("  Loading RTE...")
    dataset = load_dataset('glue', 'rte')
    actual_train = min(n_train, len(dataset['train']))
    actual_test = min(n_test, len(dataset['validation']))
    train_ds = PairTextDataset(
        dataset['train']['sentence1'][:actual_train],
        dataset['train']['sentence2'][:actual_train],
        dataset['train']['label'][:actual_train],
        tokenizer
    )
    test_ds = PairTextDataset(
        dataset['validation']['sentence1'][:actual_test],
        dataset['validation']['sentence2'][:actual_test],
        dataset['validation']['label'][:actual_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "RTE"


def load_qnli(tokenizer, n_train=1500, n_test=300):
    log("  Loading QNLI...")
    dataset = load_dataset('glue', 'qnli')
    train_ds = PairTextDataset(
        dataset['train']['question'][:n_train],
        dataset['train']['sentence'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = PairTextDataset(
        dataset['validation']['question'][:n_test],
        dataset['validation']['sentence'][:n_test],
        dataset['validation']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 2, "QNLI"


def load_mnli(tokenizer, n_train=1500, n_test=300):
    log("  Loading MNLI...")
    dataset = load_dataset('glue', 'mnli')
    train_ds = PairTextDataset(
        dataset['train']['premise'][:n_train],
        dataset['train']['hypothesis'][:n_train],
        dataset['train']['label'][:n_train],
        tokenizer
    )
    test_ds = PairTextDataset(
        dataset['validation_matched']['premise'][:n_test],
        dataset['validation_matched']['hypothesis'][:n_test],
        dataset['validation_matched']['label'][:n_test],
        tokenizer
    )
    return train_ds, test_ds, 3, "MNLI"


def load_cola(tokenizer, n_train=1500, n_test=300):
    log("  Loading CoLA...")
    dataset = load_dataset('glue', 'cola')
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
    return train_ds, test_ds, 2, "CoLA"


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


# ============================================================================
# MULTI-SHEAF MAXIMAL MODEL (v2 - Centroid Attraction)
# ============================================================================

class MultiSheafMaximalREWA(nn.Module):
    """
    Multi-Sheaf REWA with Centroid Attraction Contrastive Learning.

    Key change from v1: Uses cosine similarity to pull representations
    toward task centroids within the same sheaf.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'bert-base-uncased',
        layer_focus_start: int = 7,
        refine_epochs: int = 1,
        refine_lr_ratio: float = 0.3,
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes

        self.layer_focus_start = layer_focus_start
        self.refine_epochs = refine_epochs
        self.refine_lr_ratio = refine_lr_ratio

        # Per-sheaf frozen encoders
        self.sheaf_frozen_encoders: Dict[str, BertModel] = {}

        # Current active sheaf and task
        self.active_sheaf: Optional[AdversarialSheaf] = None
        self.current_task_name: str = ""

    def initialize_sheaf(self, sheaf: AdversarialSheaf, dataloader: DataLoader, device: str, task_name: str):
        """Initialize a sheaf's subspace and compute first task centroid."""
        log(f"  Initializing {sheaf.name} sheaf (dim={sheaf.subspace_dim})...")
        self.eval()

        # Compute lambda schedule
        sheaf.layer_lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            sheaf.layer_lambdas[i] = sheaf.lambda_max * (ratio ** 2)

        # Collect activations for PCA
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

        # Compute PCA components per layer
        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vh[:sheaf.subspace_dim, :]
            sheaf.pca_components[i] = components.T.to(device)

        # Create frozen encoder for this sheaf
        frozen = BertModel.from_pretrained(self.config._name_or_path)
        frozen.load_state_dict(self.encoder.state_dict())
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad = False
        frozen.to(device)
        self.sheaf_frozen_encoders[sheaf.name] = frozen

        # Compute centroid for this task
        self._compute_task_centroid(sheaf, dataloader, device, task_name)

        sheaf.initialized = True
        log(f"    {sheaf.name} sheaf initialized with centroid attraction")

    def _compute_task_centroid(self, sheaf: AdversarialSheaf, dataloader: DataLoader, device: str, task_name: str):
        """Compute centroid for a task in the sheaf's subspace."""
        log(f"    Computing centroid for {task_name}...")
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
                    if i < self.layer_focus_start or i not in sheaf.pca_components:
                        continue

                    h = outputs.hidden_states[i+1][:, 0, :]
                    U = sheaf.pca_components[i]
                    proj = h @ U  # (B, k)
                    layer_projections[i].append(proj.cpu())

        sheaf.task_centroids[task_name] = {}
        for i in layer_projections:
            if layer_projections[i]:
                all_proj = torch.cat(layer_projections[i], dim=0)
                sheaf.task_centroids[task_name][i] = all_proj.mean(dim=0).to(device)

    def expand_sheaf_subspace(self, sheaf: AdversarialSheaf, dataloader: DataLoader, device: str, task_name: str):
        """Expand sheaf's subspace and compute new task centroid."""
        log(f"    Expanding {sheaf.name} subspace...")
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
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)

        # Expand PCA subspace
        for i in range(self.num_layers):
            if i < self.layer_focus_start or i not in sheaf.pca_components:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            U_new = Vh[:sheaf.subspace_dim // 2, :].T.to(device)

            U_old = sheaf.pca_components[i]
            U_combined = torch.cat([U_old, U_new], dim=1)

            Q, R = torch.linalg.qr(U_combined.cpu())
            sheaf.pca_components[i] = Q[:, :sheaf.subspace_dim].to(device)

        # Update frozen encoder
        self.sheaf_frozen_encoders[sheaf.name].load_state_dict(self.encoder.state_dict())

        # Compute centroid for new task
        self._compute_task_centroid(sheaf, dataloader, device, task_name)

    def anneal_sheaf_lambdas(self, sheaf: AdversarialSheaf):
        """Decay lambdas for a specific sheaf."""
        for i in range(self.num_layers):
            sheaf.layer_lambdas[i] = max(
                sheaf.layer_lambdas[i] * sheaf.anneal_rate,
                sheaf.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        sheaf.tasks_completed += 1

    def contrastive_centroid_loss(self, hidden_states: Dict[int, torch.Tensor], sheaf: AdversarialSheaf, device: str) -> torch.Tensor:
        """
        Contrastive loss: Pull representations toward sheaf task centroids.
        Uses cosine similarity (same as forward transfer experiments).
        """
        if not sheaf.task_centroids:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        for i in range(self.num_layers):
            if i < self.layer_focus_start or i not in sheaf.pca_components:
                continue

            U = sheaf.pca_components[i]
            h = hidden_states[i][:, 0, :]  # (B, D)
            h_proj = F.normalize(h @ U, dim=1)  # (B, k) normalized

            # Compute similarity to each task centroid in this sheaf
            for task_name, centroids in sheaf.task_centroids.items():
                if i in centroids:
                    centroid = F.normalize(centroids[i].unsqueeze(0), dim=1)  # (1, k)

                    # Cosine similarity
                    sim = (h_proj @ centroid.T).squeeze() / sheaf.temperature  # (B,)

                    # Maximize similarity (minimize negative)
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
        cls_embedding_dropped = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding_dropped)

        loss = F.cross_entropy(logits, labels) if labels is not None else None
        rewa_loss = torch.tensor(0.0, device=device)
        contrast_loss = torch.tensor(0.0, device=device)

        # Apply sheaf-local REWA regularization + contrastive loss
        if self.active_sheaf is not None and self.active_sheaf.initialized and labels is not None:
            sheaf = self.active_sheaf
            frozen_encoder = self.sheaf_frozen_encoders.get(sheaf.name)

            if frozen_encoder is not None:
                with torch.no_grad():
                    frozen_outputs = frozen_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                for i in range(self.num_layers):
                    if i < self.layer_focus_start or i not in sheaf.pca_components:
                        continue

                    U = sheaf.pca_components[i]
                    curr_h = outputs.hidden_states[i+1][:, 0, :]
                    target_h = frozen_outputs.hidden_states[i+1][:, 0, :]

                    curr_proj = curr_h @ U
                    target_proj = target_h @ U

                    layer_loss = F.mse_loss(curr_proj, target_proj)
                    lambda_l = sheaf.layer_lambdas[i].to(device)
                    rewa_loss = rewa_loss + lambda_l * layer_loss

            # Contrastive centroid attraction
            hidden_dict = {i: outputs.hidden_states[i+1] for i in range(self.num_layers)}
            contrast_loss = self.contrastive_centroid_loss(hidden_dict, sheaf, device)

        # Combine all losses
        if loss is not None:
            if self.active_sheaf is not None and self.active_sheaf.initialized:
                loss = loss + rewa_loss + self.active_sheaf.lambda_contrast * contrast_loss
            else:
                loss = loss + rewa_loss

        return {
            'loss': loss,
            'logits': logits,
            'rewa_loss': rewa_loss,
            'contrast_loss': contrast_loss,
        }

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes

    def refine_task_in_sheaf(
        self,
        sheaf: AdversarialSheaf,
        task_name: str,
        train_loader: DataLoader,
        device: str,
        lr: float
    ):
        """Refine a previous task with contrastive centroid attraction."""
        log(f"      Refining {task_name} in {sheaf.name}...")

        current_head_w = self.classifier.weight.detach().clone()
        current_head_b = self.classifier.bias.detach().clone()
        current_classes = self.num_classes

        task_head = sheaf.task_heads[task_name]
        num_classes = task_head['weight'].shape[0]
        self.update_classifier(num_classes)
        self.classifier.weight.data = task_head['weight'].to(device)
        self.classifier.bias.data = task_head['bias'].to(device)

        self.active_sheaf = sheaf

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr * self.refine_lr_ratio,
            weight_decay=0.01
        )

        self.train()
        for epoch in range(self.refine_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask, labels)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

        sheaf.task_heads[task_name] = {
            'weight': self.classifier.weight.detach().cpu().clone(),
            'bias': self.classifier.bias.detach().cpu().clone()
        }

        self.update_classifier(current_classes)
        self.classifier.weight.data = current_head_w
        self.classifier.bias.data = current_head_b


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, desc="Training"):
    model.train()
    total_loss = 0
    total_rewa = 0
    total_contrast = 0
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
        total_rewa += outputs['rewa_loss'].item()
        total_contrast += outputs['contrast_loss'].item()

        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{correct/total:.1%}',
            'ctr': f'{outputs["contrast_loss"].item():.3f}'
        })

    n = len(dataloader)
    return total_loss / n, correct / total, total_rewa / n, total_contrast / n


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_multisheaf_maximal_benchmark(
    tasks: List[Tuple],
    device: str,
    epochs_per_task: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    """Run benchmark with Multi-Sheaf + Centroid Attraction."""

    log(f"\n{'='*70}")
    log(f"MULTI-SHEAF + CENTROID ATTRACTION BENCHMARK (v2)")
    log(f"{'='*70}")
    log("\nCombines: Sheaf structure + Centroid attraction contrastive loss")
    log("Key: Cosine similarity to pull toward sheaf task centroids\n")

    results = {
        'model': 'MultiSheaf-Maximal-v2',
        'task_order': [t[3] for t in tasks],
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
        'per_sheaf_fi': {},
        'sheaf_metrics': {},
    }

    model = None
    completed_tasks = []

    # Reset sheaf state
    for sheaf in ADVERSARIAL_SHEAVES.values():
        sheaf.pca_components = {}
        sheaf.layer_lambdas = None
        sheaf.tasks_completed = 0
        sheaf.task_heads = {}
        sheaf.task_train_datasets = {}
        sheaf.task_centroids = {}
        sheaf.initialized = False

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        sheaf = get_task_sheaf(task_name)
        sheaf_name = sheaf.name if sheaf else "Unknown"

        log(f"\n{'─'*60}")
        log(f"Task {task_idx + 1}/{len(tasks)}: {task_name} ({num_classes} classes)")
        log(f"Sheaf: {sheaf_name} ({sheaf.invariant_type if sheaf else 'N/A'})")
        log(f"{'─'*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        if sheaf:
            sheaf.task_train_datasets[task_name] = train_ds

        if model is None:
            log(f"  Initializing MultiSheafMaximalREWA (v2)...")
            model = MultiSheafMaximalREWA(
                num_classes=num_classes,
                layer_focus_start=7,
                refine_epochs=1,
                refine_lr_ratio=0.3,
            ).to(device)
        else:
            model.update_classifier(num_classes)

        model.active_sheaf = sheaf
        model.current_task_name = task_name

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        for epoch in range(epochs_per_task):
            desc = f"T{task_idx+1} E{epoch+1}"
            loss, acc, rewa, contrast = train_epoch(model, train_loader, optimizer, scheduler, device, desc=desc)
            log(f"  Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.1%}, REWA={rewa:.4f}, Contrast={contrast:.4f}")

        if sheaf:
            sheaf.task_heads[task_name] = {
                'weight': model.classifier.weight.detach().cpu().clone(),
                'bias': model.classifier.bias.detach().cpu().clone()
            }

        if sheaf:
            if not sheaf.initialized:
                model.initialize_sheaf(sheaf, train_loader, device, task_name)
            else:
                model.anneal_sheaf_lambdas(sheaf)
                log(f"  {sheaf.name} lambdas annealed (task {sheaf.tasks_completed})")
                model.expand_sheaf_subspace(sheaf, train_loader, device, task_name)

                log(f"  Refining {sheaf.name} sheaf tasks:")
                for prev_loader, prev_classes, prev_name, prev_sheaf in completed_tasks:
                    if prev_sheaf and prev_sheaf.name == sheaf.name:
                        prev_train_loader = DataLoader(
                            sheaf.task_train_datasets[prev_name],
                            batch_size=batch_size,
                            shuffle=True
                        )
                        model.refine_task_in_sheaf(
                            sheaf, prev_name, prev_train_loader, device, lr
                        )

        completed_tasks.append((test_loader, num_classes, task_name, sheaf))

        # Evaluate all
        log(f"  Evaluating all {len(completed_tasks)} tasks...")
        current_head_w = model.classifier.weight.detach().clone()
        current_head_b = model.classifier.bias.detach().clone()
        current_classes = model.num_classes

        for prev_loader, prev_classes, prev_name, prev_sheaf in completed_tasks:
            model.update_classifier(prev_classes)
            if prev_sheaf:
                saved_head = prev_sheaf.task_heads[prev_name]
            else:
                continue
            model.classifier.weight.data = saved_head['weight'].to(device)
            model.classifier.bias.data = saved_head['bias'].to(device)

            acc = evaluate(model, prev_loader, device)
            results['task_accuracies'][prev_name].append(acc)
            log(f"    {prev_name}: {acc:.1%}")

        model.update_classifier(current_classes)
        model.classifier.weight.data = current_head_w
        model.classifier.bias.data = current_head_b

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    log(f"\n{'='*70}")
    log(f"FINAL METRICS")
    log(f"{'='*70}")

    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final
    results['WTA'] = min([accs[-1] for accs in results['task_accuracies'].values()])

    forgettings = []
    for task_name, accs in results['task_accuracies'].items():
        if len(accs) > 1:
            max_acc = max(accs[:-1])
            final_acc = accs[-1]
            forgetting = max_acc - final_acc
            results['per_task_forgetting'][task_name] = forgetting
            forgettings.append(forgetting)

    results['FI_global'] = np.mean(forgettings) if forgettings else 0.0

    final_values = [accs[-1] for accs in results['task_accuracies'].values()]
    results['PV'] = np.var(final_values)

    # Per-Sheaf FI
    log(f"\n{'─'*60}")
    log(f"PER-SHEAF FORGETTING INDEX (FI)")
    log(f"{'─'*60}")

    for sheaf_key, sheaf in ADVERSARIAL_SHEAVES.items():
        sheaf_forgettings = []
        sheaf_final_accs = []

        for task_name in sheaf.task_names:
            if task_name in results['task_accuracies']:
                accs = results['task_accuracies'][task_name]
                sheaf_final_accs.append(accs[-1])

                if len(accs) > 1:
                    max_acc = max(accs[:-1])
                    final_acc = accs[-1]
                    forgetting = max_acc - final_acc
                    sheaf_forgettings.append(forgetting)

        sheaf_fi = np.mean(sheaf_forgettings) if sheaf_forgettings else 0.0
        sheaf_avg = np.mean(sheaf_final_accs) if sheaf_final_accs else 0.0

        results['per_sheaf_fi'][sheaf.name] = sheaf_fi
        results['sheaf_metrics'][sheaf.name] = {
            'fi': sheaf_fi,
            'avg_accuracy': sheaf_avg,
            'tasks': sheaf.task_names,
            'invariant': sheaf.invariant_type,
        }

        if sheaf_final_accs:
            fi_status = "NEGATIVE" if sheaf_fi <= 0 else "positive"
            log(f"  {sheaf.name} ({sheaf.invariant_type}):")
            log(f"    FI = {sheaf_fi:+.2%} [{fi_status}]")
            log(f"    Avg Accuracy = {sheaf_avg:.1%}")
            log(f"    Tasks: {[t for t in sheaf.task_names if t in results['task_accuracies']]}")

    # Print detailed task summary
    log(f"\n{'─'*60}")
    log(f"{'Task':<12} {'Sheaf':<12} {'Final Acc':<12} {'Forgetting':<12}")
    log(f"{'─'*60}")

    for task_name in results['task_order']:
        if task_name not in results['final_accuracies']:
            continue
        sheaf = get_task_sheaf(task_name)
        sheaf_name = sheaf.name if sheaf else "N/A"
        acc = results['final_accuracies'][task_name]
        forg = results['per_task_forgetting'].get(task_name, 0.0)
        forg_str = f"{forg:+.1%}" if task_name in results['per_task_forgetting'] else "N/A"
        log(f"{task_name:<12} {sheaf_name:<12} {acc:<12.1%} {forg_str:<12}")

    log(f"{'─'*60}")
    log(f"\nAggregate Metrics:")
    log(f"  Average Final Accuracy: {avg_final:.1%}")
    log(f"  Worst-Task Accuracy:    {results['WTA']:.1%}")
    log(f"  Global FI:              {results['FI_global']:+.2%}")
    log(f"  Performance Variance:   {results['PV']:.4f}")

    # Success check
    log(f"\n{'='*70}")
    if results['FI_global'] < -0.05:
        log(f"MAJOR SUCCESS: Global FI < -5% ({results['FI_global']:+.2%})")
    elif results['FI_global'] < -0.03:
        log(f"SUCCESS: Global FI < -3% ({results['FI_global']:+.2%})")
    elif results['FI_global'] < 0:
        log(f"FORWARD TRANSFER: Global FI is negative ({results['FI_global']:+.2%})")
    else:
        log(f"Forgetting: Global FI is positive ({results['FI_global']:+.2%})")
    log(f"{'='*70}")

    return results, model


# ============================================================================
# MAIN
# ============================================================================

def main():
    log("="*70)
    log("MULTI-SHEAF + CENTROID ATTRACTION (v2)")
    log("="*70)
    log("\nFull 10-task GLUE benchmark")
    log("Using centroid attraction from forward transfer experiments\n")

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"Device: {device}")

    n_train = 1500
    n_test = 300

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Full 10-task GLUE benchmark (grouped by sheaf)
    log("\nLoading tasks organized by semantic sheaf...")
    tasks = [
        # Sentiment Sheaf (polarity)
        load_sst2(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
        # Paraphrase Sheaf (symmetry)
        load_qqp(tokenizer, n_train, n_test),
        load_mrpc(tokenizer, n_train, n_test),
        # Entailment Sheaf (directionality)
        load_rte(tokenizer, n_train, n_test),
        load_qnli(tokenizer, n_train, n_test),
        load_mnli(tokenizer, n_train, n_test),
        # Syntax Sheaf (grammar)
        load_cola(tokenizer, n_train, n_test),
        # Topic Sheaf (clustering)
        load_ag_news(tokenizer, n_train, n_test),
    ]

    results, model = run_multisheaf_maximal_benchmark(
        tasks=tasks,
        device=device,
        epochs_per_task=2,
        batch_size=16,
        lr=2e-5
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multisheaf_maximal_v2_results_{timestamp}.json"

    json_results = {
        'model': results['model'],
        'num_tasks': len(tasks),
        'task_order': results['task_order'],
        'task_accuracies': {k: [float(x) for x in v] for k, v in results['task_accuracies'].items()},
        'final_accuracies': {k: float(v) for k, v in results['final_accuracies'].items()},
        'per_task_forgetting': {k: float(v) for k, v in results['per_task_forgetting'].items()},
        'per_sheaf_fi': {k: float(v) for k, v in results['per_sheaf_fi'].items()},
        'sheaf_metrics': {
            k: {
                'fi': float(v['fi']),
                'avg_accuracy': float(v['avg_accuracy']),
                'tasks': v['tasks'],
                'invariant': v['invariant'],
            }
            for k, v in results['sheaf_metrics'].items()
        },
        'average_final_accuracy': float(results['average_final_accuracy']),
        'WTA': float(results['WTA']),
        'FI_global': float(results['FI_global']),
        'PV': float(results['PV']),
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
