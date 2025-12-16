#!/usr/bin/env python3
"""
GLUE Benchmark with Curriculum+Expand+Refine Approach

Uses the winning strategy from forward transfer experiments:
- Curriculum ordering: Easy tasks first (sentiment → topic)
- Subspace expansion: Grow shared subspace with each task
- Refinement: Actively improve old tasks in expanded subspace

Target: Negative FI (forward transfer) across 10 tasks
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
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.rewa_bert import evaluate


def log(msg):
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


class PairTextDataset(Dataset):
    """Dataset for sentence pair tasks (MNLI, QNLI, QQP, RTE, MRPC)."""
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


# Task loaders
def load_sst2(tokenizer, n_train=1500, n_test=300):
    """SST-2: Stanford Sentiment Treebank (binary sentiment)."""
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
    """IMDB: Movie reviews (binary sentiment)."""
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
    """Yelp: Business reviews (binary sentiment)."""
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


def load_cola(tokenizer, n_train=1500, n_test=300):
    """CoLA: Corpus of Linguistic Acceptability."""
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


def load_mrpc(tokenizer, n_train=1500, n_test=300):
    """MRPC: Microsoft Research Paraphrase Corpus."""
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


def load_qqp(tokenizer, n_train=1500, n_test=300):
    """QQP: Quora Question Pairs."""
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


def load_rte(tokenizer, n_train=1500, n_test=300):
    """RTE: Recognizing Textual Entailment."""
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
    """QNLI: Question NLI."""
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
    """MNLI: Multi-Genre NLI (3-class)."""
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


def load_ag_news(tokenizer, n_train=1500, n_test=300):
    """AG News: Topic classification (4-class)."""
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
# CURRICULUM ORDERING
# ============================================================================

# Task difficulty ordering (easy to hard)
# Rationale:
# - Binary sentiment: Easiest (SST2, IMDB, Yelp)
# - Binary single-sentence: Easy (CoLA)
# - Binary pair classification: Medium (MRPC, QQP, RTE, QNLI)
# - Multi-class NLI: Hard (MNLI)
# - Multi-class topic: Hardest (AG_News - completely different domain)

TASK_DIFFICULTY = {
    "SST2": 1,      # Easy: short binary sentiment
    "IMDB": 2,      # Easy: longer binary sentiment
    "Yelp": 3,      # Easy: review binary sentiment
    "CoLA": 4,      # Medium: grammaticality judgment
    "MRPC": 5,      # Medium: paraphrase detection
    "QQP": 6,       # Medium: question paraphrase
    "RTE": 7,       # Medium-Hard: textual entailment (small dataset)
    "QNLI": 8,      # Hard: question-answer entailment
    "MNLI": 9,      # Hard: 3-class NLI
    "AG_News": 10,  # Hardest: 4-class topic (domain shift)
}


def get_curriculum_order(tasks: List[Tuple]) -> List[Tuple]:
    """Order tasks from easy to hard."""
    task_dict = {t[3]: t for t in tasks}
    ordered_names = sorted(
        [t[3] for t in tasks],
        key=lambda x: TASK_DIFFICULTY.get(x, 100)
    )
    return [task_dict[name] for name in ordered_names]


# ============================================================================
# EXPANSION-REFINEMENT REWA MODEL
# ============================================================================

class ExpansionRefinementREWA(nn.Module):
    """
    Subspace-REWA with expansion and refinement phases.

    After each new task:
    1. Expand shared subspace to include new task's directions
    2. Refine old tasks in the expanded subspace
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
        """Expand subspace to include new task's important directions."""
        log(f"  Expanding subspace...")
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

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            U_new = Vh[:self.subspace_dim // 2, :].T.to(device)

            U_old = self.pca_components[i]
            U_combined = torch.cat([U_old, U_new], dim=1)

            # Re-orthogonalize (CPU for MPS compatibility)
            Q, R = torch.linalg.qr(U_combined.cpu())
            self.pca_components[i] = Q[:, :self.subspace_dim].to(device)

        # Update frozen encoder
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())

    def refine_task(
        self,
        task_name: str,
        train_loader: DataLoader,
        task_head: Dict,
        device: str,
        lr: float
    ):
        """Refine a previous task in the expanded subspace."""
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

def run_curriculum_expand_refine_benchmark(
    tasks: List[Tuple],
    device: str,
    epochs_per_task: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    """Run 10-task GLUE benchmark with Curriculum+Expand+Refine."""

    # Apply curriculum ordering
    tasks = get_curriculum_order(tasks)

    log(f"\n{'='*70}")
    log(f"GLUE BENCHMARK: Curriculum + Expand + Refine")
    log(f"{'='*70}")
    log(f"\nTask order (easy → hard):")
    for i, (_, _, nc, name) in enumerate(tasks):
        log(f"  {i+1}. {name} ({nc} classes)")

    results = {
        'model': 'Curriculum+Expand+Refine',
        'task_order': [t[3] for t in tasks],
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
        'lambda_history': [],
    }

    model = None
    completed_tasks = []  # (test_loader, num_classes, task_name)
    task_heads = {}
    task_train_datasets = {}  # Store train datasets for refinement

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        log(f"\n{'─'*60}")
        log(f"Task {task_idx + 1}/{len(tasks)}: {task_name} ({num_classes} classes)")
        log(f"{'─'*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Store for refinement
        task_train_datasets[task_name] = train_ds

        # Initialize or update model
        if model is None:
            log(f"  Initializing ExpansionRefinementREWA...")
            model = ExpansionRefinementREWA(
                num_classes=num_classes,
                lambda_max=10.0,
                subspace_dim=64,
                layer_focus_start=7,
                anneal_rate=0.85,
                min_lambda=1.0,
                refine_epochs=1,
                refine_lr_ratio=0.3,
            ).to(device)
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
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device, desc=desc)
            log(f"  Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.1%}")

        # Save task head
        task_heads[task_name] = {
            'weight': model.classifier.weight.detach().cpu().clone(),
            'bias': model.classifier.bias.detach().cpu().clone()
        }

        # Setup subspace on first task
        if task_idx == 0:
            model.compute_subspaces(train_loader, device)
        else:
            # Anneal lambdas
            model.anneal_lambdas()
            log(f"  Lambda annealed (task {model.tasks_completed}). L11: {model.layer_lambdas[11]:.2f}")

            # Expand subspace
            model.expand_subspace(train_loader, device)

            # Refinement phase
            log(f"  Refinement phase:")
            for prev_loader, prev_classes, prev_name in completed_tasks:
                # Create train loader for refinement
                prev_train_loader = DataLoader(
                    task_train_datasets[prev_name],
                    batch_size=batch_size,
                    shuffle=True
                )
                task_heads[prev_name] = model.refine_task(
                    prev_name, prev_train_loader, task_heads[prev_name], device, lr
                )

        completed_tasks.append((test_loader, num_classes, task_name))

        # Evaluate all completed tasks
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
    log(f"\n{'='*70}")
    log(f"FINAL METRICS")
    log(f"{'='*70}")

    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final

    task_accs = results['task_accuracies']
    results['WTA'] = min([accs[-1] for accs in task_accs.values()])

    # Compute forgetting
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

    # Lambda history
    results['lambda_history'] = [lh.tolist() for lh in model.lambda_history]

    # Print summary
    log(f"\n{'─'*60}")
    log(f"{'Task':<12} {'Final Acc':<12} {'Forgetting':<12}")
    log(f"{'─'*60}")
    for task_name in results['task_order']:
        acc = results['final_accuracies'][task_name]
        forg = results['per_task_forgetting'].get(task_name, 0.0)
        forg_str = f"{forg:+.1%}" if task_name in results['per_task_forgetting'] else "N/A"
        log(f"{task_name:<12} {acc:<12.1%} {forg_str:<12}")

    log(f"{'─'*60}")
    log(f"\nAggregate Metrics:")
    log(f"  Average Final Accuracy: {avg_final:.1%}")
    log(f"  Worst-Task Accuracy:    {results['WTA']:.1%}")
    log(f"  Forgetting Index (FI):  {results['FI']:+.2%}")
    log(f"  Performance Variance:   {results['PV']:.4f}")

    # Critical check
    log(f"\n{'='*70}")
    if results['FI'] <= 0:
        log(f"✓ FI IS NEGATIVE ({results['FI']:+.2%}): Forward transfer confirmed!")
        log(f"  → Nature MI target is REALISTIC for 10-task benchmark")
    else:
        log(f"✗ FI is positive ({results['FI']:+.2%}): Some forgetting occurred")
        log(f"  → May need hyperparameter tuning for 10 tasks")
    log(f"{'='*70}")

    return results, model


# ============================================================================
# MAIN
# ============================================================================

def main():
    log("="*70)
    log("GLUE BENCHMARK: Curriculum + Expand + Refine")
    log("="*70)
    log("\nThis approach achieved FI = -4.11% on 4 tasks.")
    log("Testing: Does it scale to 10 diverse GLUE tasks?\n")

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

    log("\nLoading 10 GLUE benchmark tasks...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load all tasks (will be reordered by curriculum)
    tasks = [
        load_sst2(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
        load_cola(tokenizer, n_train, n_test),
        load_mrpc(tokenizer, n_train, n_test),
        load_qqp(tokenizer, n_train, n_test),
        load_rte(tokenizer, n_train, n_test),
        load_qnli(tokenizer, n_train, n_test),
        load_mnli(tokenizer, n_train, n_test),
        load_ag_news(tokenizer, n_train, n_test),
    ]

    # Run benchmark
    results, model = run_curriculum_expand_refine_benchmark(
        tasks=tasks,
        device=device,
        epochs_per_task=2,
        batch_size=16,
        lr=2e-5
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"curriculum_expand_refine_results_{timestamp}.json"

    json_results = {
        'model': results['model'],
        'num_tasks': len(tasks),
        'task_order': results['task_order'],
        'task_accuracies': {k: [float(x) for x in v] for k, v in results['task_accuracies'].items()},
        'final_accuracies': {k: float(v) for k, v in results['final_accuracies'].items()},
        'per_task_forgetting': {k: float(v) for k, v in results['per_task_forgetting'].items()},
        'average_final_accuracy': float(results['average_final_accuracy']),
        'WTA': float(results['WTA']),
        'FI': float(results['FI']),
        'PV': float(results['PV']),
        'lambda_history': results['lambda_history'],
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
