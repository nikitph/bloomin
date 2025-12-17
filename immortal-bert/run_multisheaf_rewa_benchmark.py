#!/usr/bin/env python3
"""
Multi-Sheaf REWA for GLUE Benchmark

Implements the theory-consistent approach:
- Negative FI is LOCAL, not global
- Tasks are grouped into semantic sheaves by geometric compatibility
- Each sheaf has its own subspace, lambda schedule, and refinement loop
- FI is reported per sheaf, not globally

Sheaf structure:
- Sentiment Sheaf: SST2, IMDB, Yelp (polarity separation invariant)
- Paraphrase Sheaf: QQP, MRPC (symmetry & equivalence invariant)
- Entailment Sheaf: RTE, QNLI, MNLI (directionality & transitivity invariant)
- Syntax Sheaf: CoLA (grammatical constraints invariant)
- Topic Sheaf: AG_News (global lexical clustering invariant)

Key theorem: There exists no single semantic manifold on which all heterogeneous
NLP tasks induce mutually curvature-reducing flows.
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
class SemanticSheaf:
    """A semantic regime with compatible geometric invariants."""
    name: str
    task_names: List[str]
    invariant_type: str  # "polarity", "symmetry", "directionality", "grammar", "clustering"

    # Sheaf-local state
    subspace_dim: int = 64
    lambda_max: float = 10.0
    anneal_rate: float = 0.85
    min_lambda: float = 1.0

    # Populated during training
    pca_components: Dict[int, torch.Tensor] = field(default_factory=dict)
    frozen_encoder_state: Optional[Dict] = None
    layer_lambdas: Optional[torch.Tensor] = None
    tasks_completed: int = 0
    task_heads: Dict = field(default_factory=dict)
    task_train_datasets: Dict = field(default_factory=dict)
    task_accuracies: Dict = field(default_factory=lambda: defaultdict(list))
    initialized: bool = False


# Define the sheaves based on geometric compatibility
SEMANTIC_SHEAVES = {
    "sentiment": SemanticSheaf(
        name="Sentiment",
        task_names=["SST2", "IMDB", "Yelp"],
        invariant_type="polarity",
        lambda_max=10.0,
        anneal_rate=0.9,  # Slower anneal for compatible tasks
    ),
    "paraphrase": SemanticSheaf(
        name="Paraphrase",
        task_names=["QQP", "MRPC"],
        invariant_type="symmetry",
        lambda_max=10.0,
        anneal_rate=0.88,
    ),
    "entailment": SemanticSheaf(
        name="Entailment",
        task_names=["RTE", "QNLI", "MNLI"],
        invariant_type="directionality",
        lambda_max=8.0,  # Lower lambda - harder to preserve
        anneal_rate=0.85,
    ),
    "syntax": SemanticSheaf(
        name="Syntax",
        task_names=["CoLA"],
        invariant_type="grammar",
        lambda_max=5.0,  # Standalone, minimal coupling
        anneal_rate=0.9,
    ),
    "topic": SemanticSheaf(
        name="Topic",
        task_names=["AG_News"],
        invariant_type="clustering",
        lambda_max=5.0,  # Standalone
        anneal_rate=0.9,
    ),
}


def get_task_sheaf(task_name: str) -> Optional[SemanticSheaf]:
    """Find which sheaf a task belongs to."""
    for sheaf in SEMANTIC_SHEAVES.values():
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


# Task loaders (ordered by sheaf)
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
# MULTI-SHEAF REWA MODEL
# ============================================================================

class MultiSheafREWA(nn.Module):
    """
    Multi-Sheaf REWA: Negative FI is local, not global.

    Key principles:
    1. Each semantic sheaf has its own subspace
    2. REWA regularization only applies within sheaves
    3. Cross-sheaf tasks have weak or no coupling
    4. FI is computed per sheaf
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'bert-base-uncased',
        layer_focus_start: int = 7,
        refine_epochs: int = 1,
        refine_lr_ratio: float = 0.3,
        cross_sheaf_lambda: float = 0.5,  # Weak cross-sheaf coupling
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
        self.cross_sheaf_lambda = cross_sheaf_lambda

        # Sheaf-local frozen encoders (one per sheaf)
        self.sheaf_frozen_encoders: Dict[str, BertModel] = {}

        # Current active sheaf
        self.active_sheaf: Optional[SemanticSheaf] = None

    def initialize_sheaf(self, sheaf: SemanticSheaf, dataloader: DataLoader, device: str):
        """Initialize a sheaf's subspace from its first task."""
        log(f"  Initializing {sheaf.name} sheaf (dim={sheaf.subspace_dim})...")
        self.eval()

        # Compute lambda schedule for this sheaf
        sheaf.layer_lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            ratio = (i + 1) / self.num_layers
            sheaf.layer_lambdas[i] = sheaf.lambda_max * (ratio ** 2)

        # Collect activations
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

        sheaf.initialized = True
        log(f"    {sheaf.name} sheaf initialized with {len(sheaf.pca_components)} layer subspaces")

    def expand_sheaf_subspace(self, sheaf: SemanticSheaf, dataloader: DataLoader, device: str):
        """Expand sheaf's subspace to include new task's directions."""
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

            # Re-orthogonalize
            Q, R = torch.linalg.qr(U_combined.cpu())
            sheaf.pca_components[i] = Q[:, :sheaf.subspace_dim].to(device)

        # Update frozen encoder for this sheaf
        self.sheaf_frozen_encoders[sheaf.name].load_state_dict(self.encoder.state_dict())

    def anneal_sheaf_lambdas(self, sheaf: SemanticSheaf):
        """Decay lambdas for a specific sheaf."""
        for i in range(self.num_layers):
            sheaf.layer_lambdas[i] = max(
                sheaf.layer_lambdas[i] * sheaf.anneal_rate,
                sheaf.min_lambda * ((i + 1) / self.num_layers) ** 2
            )
        sheaf.tasks_completed += 1

    def refine_task_in_sheaf(
        self,
        sheaf: SemanticSheaf,
        task_name: str,
        train_loader: DataLoader,
        device: str,
        lr: float
    ):
        """Refine a previous task within its sheaf."""
        log(f"      Refining {task_name} in {sheaf.name}...")

        # Save current state
        current_head_w = self.classifier.weight.detach().clone()
        current_head_b = self.classifier.bias.detach().clone()
        current_classes = self.num_classes

        # Load old task head
        task_head = sheaf.task_heads[task_name]
        num_classes = task_head['weight'].shape[0]
        self.update_classifier(num_classes)
        self.classifier.weight.data = task_head['weight'].to(device)
        self.classifier.bias.data = task_head['bias'].to(device)

        # Set active sheaf for regularization
        self.active_sheaf = sheaf

        # Light refinement
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

        # Save refined head
        sheaf.task_heads[task_name] = {
            'weight': self.classifier.weight.detach().cpu().clone(),
            'bias': self.classifier.bias.detach().cpu().clone()
        }

        # Restore current head
        self.update_classifier(current_classes)
        self.classifier.weight.data = current_head_w
        self.classifier.bias.data = current_head_b

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

        # Apply sheaf-local REWA regularization
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

        if loss is not None:
            loss = loss + rewa_loss

        return {'loss': loss, 'logits': logits, 'rewa_loss': rewa_loss}

    def update_classifier(self, num_classes: int):
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, desc="Training"):
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
# BENCHMARK RUNNER
# ============================================================================

def run_multisheaf_benchmark(
    tasks: List[Tuple],
    device: str,
    epochs_per_task: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    """Run GLUE benchmark with Multi-Sheaf REWA."""

    log(f"\n{'='*70}")
    log(f"MULTI-SHEAF REWA BENCHMARK")
    log(f"{'='*70}")
    log("\nTheory: Negative FI is LOCAL, not global.")
    log("Each semantic sheaf maintains its own subspace and FI.\n")

    # Print sheaf structure
    log("Sheaf Structure:")
    for key, sheaf in SEMANTIC_SHEAVES.items():
        log(f"  {sheaf.name}: {sheaf.task_names} ({sheaf.invariant_type})")

    results = {
        'model': 'Multi-Sheaf-REWA',
        'task_order': [t[3] for t in tasks],
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
        'per_sheaf_fi': {},
        'sheaf_metrics': {},
    }

    model = None
    completed_tasks = []  # (test_loader, num_classes, task_name, sheaf)

    # Reset sheaf state
    for sheaf in SEMANTIC_SHEAVES.values():
        sheaf.pca_components = {}
        sheaf.frozen_encoder_state = None
        sheaf.layer_lambdas = None
        sheaf.tasks_completed = 0
        sheaf.task_heads = {}
        sheaf.task_train_datasets = {}
        sheaf.task_accuracies = defaultdict(list)
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

        # Store for refinement
        if sheaf:
            sheaf.task_train_datasets[task_name] = train_ds

        # Initialize model
        if model is None:
            log(f"  Initializing MultiSheafREWA...")
            model = MultiSheafREWA(
                num_classes=num_classes,
                layer_focus_start=7,
                refine_epochs=1,
                refine_lr_ratio=0.3,
            ).to(device)
        else:
            model.update_classifier(num_classes)

        # Set active sheaf
        model.active_sheaf = sheaf

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        # Train
        for epoch in range(epochs_per_task):
            desc = f"T{task_idx+1} E{epoch+1}"
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device, desc=desc)
            log(f"  Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.1%}")

        # Save task head
        if sheaf:
            sheaf.task_heads[task_name] = {
                'weight': model.classifier.weight.detach().cpu().clone(),
                'bias': model.classifier.bias.detach().cpu().clone()
            }

        # Initialize or expand sheaf
        if sheaf:
            if not sheaf.initialized:
                model.initialize_sheaf(sheaf, train_loader, device)
            else:
                # Anneal lambdas
                model.anneal_sheaf_lambdas(sheaf)
                log(f"  {sheaf.name} lambdas annealed (task {sheaf.tasks_completed})")

                # Expand subspace
                model.expand_sheaf_subspace(sheaf, train_loader, device)

                # Refine ONLY tasks within the same sheaf
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

        # Evaluate all completed tasks
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

            # Track per-sheaf
            if prev_sheaf:
                prev_sheaf.task_accuracies[prev_name].append(acc)

            log(f"    {prev_name} ({prev_sheaf.name if prev_sheaf else 'N/A'}): {acc:.1%}")

        model.update_classifier(current_classes)
        model.classifier.weight.data = current_head_w
        model.classifier.bias.data = current_head_b

    # ========================================================================
    # COMPUTE METRICS (Per-Sheaf and Global)
    # ========================================================================

    log(f"\n{'='*70}")
    log(f"FINAL METRICS")
    log(f"{'='*70}")

    # Final accuracies
    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    # Global metrics
    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final
    results['WTA'] = min([accs[-1] for accs in results['task_accuracies'].values()])

    # Global forgetting
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

    # Per-Sheaf FI (THE KEY METRIC)
    log(f"\n{'─'*60}")
    log(f"PER-SHEAF FORGETTING INDEX (FI)")
    log(f"{'─'*60}")

    for sheaf_key, sheaf in SEMANTIC_SHEAVES.items():
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

        fi_status = "NEGATIVE" if sheaf_fi <= 0 else "positive"
        log(f"  {sheaf.name} ({sheaf.invariant_type}):")
        log(f"    FI = {sheaf_fi:+.2%} [{fi_status}]")
        log(f"    Avg Accuracy = {sheaf_avg:.1%}")
        log(f"    Tasks: {sheaf.task_names}")

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

    # Theory validation
    log(f"\n{'='*70}")
    log(f"THEORY VALIDATION")
    log(f"{'='*70}")

    negative_sheaves = [s for s, fi in results['per_sheaf_fi'].items() if fi <= 0]
    positive_sheaves = [s for s, fi in results['per_sheaf_fi'].items() if fi > 0]

    log(f"\nSheaves with NEGATIVE FI (forward transfer):")
    for s in negative_sheaves:
        log(f"  {s}: FI = {results['per_sheaf_fi'][s]:+.2%}")

    if positive_sheaves:
        log(f"\nSheaves with positive FI (some forgetting):")
        for s in positive_sheaves:
            log(f"  {s}: FI = {results['per_sheaf_fi'][s]:+.2%}")

    log(f"\nTheoretical prediction:")
    log(f"  - FI can be negative WITHIN semantically coherent sheaves")
    log(f"  - FI across incompatible sheaves is NOT expected to be negative")
    log(f"  - This is a LAW, not a limitation")
    log(f"{'='*70}")

    return results, model


# ============================================================================
# MAIN
# ============================================================================

def main():
    log("="*70)
    log("MULTI-SHEAF REWA: Theory-Consistent Continual Learning")
    log("="*70)
    log("\nKey insight: Negative FI is LOCAL, not global.")
    log("Tasks are grouped by semantic compatibility into sheaves.")
    log("FI is measured per sheaf, not globally.\n")

    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"Device: {device}")

    # Parameters
    n_train = 1500
    n_test = 300

    log("\nLoading tasks organized by semantic sheaf...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load tasks in sheaf-grouped order (within-sheaf curriculum)
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

    # Run benchmark
    results, model = run_multisheaf_benchmark(
        tasks=tasks,
        device=device,
        epochs_per_task=2,
        batch_size=16,
        lr=2e-5
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multisheaf_rewa_results_{timestamp}.json"

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
