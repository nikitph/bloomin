#!/usr/bin/env python3
"""
Optimal Order GLUE Benchmark with MaximalForwardTransferREWA

NO SHEAFS - just the simple PCA-subspace approach from run_forward_transfer_experiments.py
with optimal task ordering to maximize negative FI.

Task order (easy → hard, grouped by similarity):
1. SST2      - Simple binary sentiment
2. IMDB      - Binary sentiment, longer
3. Yelp      - Binary sentiment, reviews
4. MRPC      - Paraphrase (small)
5. QQP       - Paraphrase (large)
6. RTE       - Entailment (2-class)
7. QNLI      - Question entailment
8. MNLI      - Multi-genre entailment (3-class)
9. CoLA      - Linguistic acceptability
10. AG_News  - Topic (4-class, most different)
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
from typing import Dict, List
from tqdm import tqdm

from src.rewa_bert import evaluate


def log(msg):
    print(msg, flush=True)


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
# MaximalForwardTransferREWA (from run_forward_transfer_experiments.py)
# NO SHEAFS - just PCA subspaces
# ============================================================================

class MaximalForwardTransferREWA(nn.Module):
    """
    PCA-subspace REWA with:
    - Subspace expansion
    - Task refinement
    - Contrastive centroid alignment
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
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.refine_epochs = refine_epochs
        self.refine_lr_ratio = refine_lr_ratio

        self.pca_components: Dict[int, torch.Tensor] = {}
        self.subspace_identified = False

        self.layer_lambdas = self._compute_initial_lambda_schedule()
        self.lambda_history: List[torch.Tensor] = [self.layer_lambdas.clone()]

        self.frozen_encoder = None
        self.tasks_completed = 0

        # Task centroids for contrastive
        self.task_centroids: Dict[str, Dict[int, torch.Tensor]] = {}

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

    def compute_subspaces(self, dataloader: DataLoader, device: str):
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

    def expand_subspace(self, dataloader: DataLoader, device: str):
        log(f"  Expanding subspace...")
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
            if i < self.layer_focus_start:
                continue

            X = torch.cat(layer_activations[i], dim=0)
            mu = X.mean(dim=0)
            X_centered = X - mu

            _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            U_new = Vh[:self.subspace_dim // 2, :].T.to(device)

            U_old = self.pca_components[i]
            U_combined = torch.cat([U_old, U_new], dim=1)

            Q, R = torch.linalg.qr(U_combined.cpu())
            self.pca_components[i] = Q[:, :self.subspace_dim].to(device)

        self.frozen_encoder.load_state_dict(self.encoder.state_dict())

    def compute_task_centroid(self, task_name: str, dataloader: DataLoader, device: str):
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
                    proj = h @ U
                    layer_projections[i].append(proj.cpu())

        self.task_centroids[task_name] = {}
        for i in layer_projections:
            if layer_projections[i]:
                all_proj = torch.cat(layer_projections[i], dim=0)
                self.task_centroids[task_name][i] = all_proj.mean(dim=0).to(device)

    def contrastive_loss(self, hidden_states: Dict[int, torch.Tensor], device: str) -> torch.Tensor:
        if not self.task_centroids:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        for i in range(self.num_layers):
            if i < self.layer_focus_start or i not in self.pca_components:
                continue

            U = self.pca_components[i]
            h = hidden_states[i][:, 0, :]
            h_proj = F.normalize(h @ U, dim=1)

            for task_name, centroids in self.task_centroids.items():
                if i in centroids:
                    centroid = F.normalize(centroids[i].unsqueeze(0), dim=1)
                    sim = (h_proj @ centroid.T).squeeze() / self.temperature
                    loss = loss - sim.mean()

        return loss

    def refine_task(
        self,
        task_name: str,
        task_loader: DataLoader,
        task_head: Dict,
        device: str,
        lr: float
    ):
        log(f"    Refining {task_name}...")

        current_head_w = self.classifier.weight.detach().clone()
        current_head_b = self.classifier.bias.detach().clone()
        current_classes = self.num_classes

        num_classes = task_head['weight'].shape[0]
        self.update_classifier(num_classes)
        self.classifier.weight.data = task_head['weight'].to(device)
        self.classifier.bias.data = task_head['bias'].to(device)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr * self.refine_lr_ratio,
            weight_decay=0.01
        )

        self.train()

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

        task_head['weight'] = self.classifier.weight.detach().cpu().clone()
        task_head['bias'] = self.classifier.bias.detach().cpu().clone()

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


def run_benchmark(tasks, device, epochs_per_task=2, batch_size=16, lr=2e-5):
    """Run benchmark with MaximalForwardTransferREWA."""
    log(f"\n{'='*70}")
    log(f"OPTIMAL ORDER GLUE BENCHMARK: MaximalForwardTransferREWA")
    log(f"{'='*70}")

    results = {
        'model': 'MaximalForwardTransferREWA',
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
        'lambda_history': [],
    }

    model = None
    completed_tasks = []
    task_heads = {}
    task_loaders = {}  # Store for refinement

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        log(f"\n{'─'*60}")
        log(f"Task {task_idx + 1}/{len(tasks)}: {task_name} ({num_classes} classes)")
        log(f"{'─'*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        task_loaders[task_name] = train_loader

        if model is None:
            log(f"  Initializing MaximalForwardTransferREWA...")
            model = MaximalForwardTransferREWA(
                num_classes=num_classes,
                lambda_max=10.0,
                subspace_dim=64,
                layer_focus_start=7,
                anneal_rate=0.85,
                min_lambda=1.0,
                lambda_contrast=0.3,
                temperature=0.1,
                refine_epochs=1,
                refine_lr_ratio=0.3,
            ).to(device)
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
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device, desc=desc)
            log(f"  Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.1%}")

        # Save task head
        task_heads[task_name] = {
            'weight': model.classifier.weight.detach().cpu().clone(),
            'bias': model.classifier.bias.detach().cpu().clone()
        }

        # First task: setup subspace
        if task_idx == 0:
            model.compute_subspaces(train_loader, device)
            model.compute_task_centroid(task_name, train_loader, device)
        else:
            # Anneal lambdas
            model.anneal_lambdas()
            log(f"  Lambda annealed. L11: {model.layer_lambdas[11]:.2f}")

            # Expand subspace
            model.expand_subspace(train_loader, device)

            # Compute centroid
            model.compute_task_centroid(task_name, train_loader, device)

            # Refinement phase
            log(f"  Refinement phase:")
            for prev_name in list(task_heads.keys())[:-1]:  # All except current
                if prev_name in task_loaders:
                    task_heads[prev_name] = model.refine_task(
                        prev_name, task_loaders[prev_name], task_heads[prev_name], device, lr
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
    log(f"\n{'='*70}")
    log(f"FINAL METRICS")
    log(f"{'='*70}")

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

    results['lambda_history'] = [lh.tolist() for lh in model.lambda_history]

    # Print summary
    log(f"\n{'─'*50}")
    log(f"{'Task':<12} {'Final Acc':<12} {'Forgetting':<12}")
    log(f"{'─'*50}")
    for task_name in results['final_accuracies']:
        acc = results['final_accuracies'][task_name]
        forg = results['per_task_forgetting'].get(task_name, 0.0)
        forg_str = f"{forg:+.1%}" if task_name in results['per_task_forgetting'] else "N/A"
        log(f"{task_name:<12} {acc:<12.1%} {forg_str:<12}")

    log(f"{'─'*50}")
    log(f"\nAggregate Metrics:")
    log(f"  Average Final Accuracy: {avg_final:.1%}")
    log(f"  Worst-Task Accuracy:    {results['WTA']:.1%}")
    log(f"  Forgetting Index (FI):  {results['FI']:+.2%}")
    log(f"  Performance Variance:   {results['PV']:.4f}")

    log(f"\n{'='*70}")
    if results['FI'] < -0.05:
        log(f"MAJOR SUCCESS: FI < -5% ({results['FI']:+.2%})")
    elif results['FI'] < -0.03:
        log(f"SUCCESS: FI < -3% ({results['FI']:+.2%})")
    elif results['FI'] <= 0:
        log(f"NEGATIVE FI ACHIEVED ({results['FI']:+.2%}): Forward transfer confirmed!")
    else:
        log(f"FI is positive ({results['FI']:+.2%}): Some forgetting occurred")
    log(f"{'='*70}")

    return results


def main():
    log("="*70)
    log("OPTIMAL ORDER GLUE BENCHMARK")
    log("MaximalForwardTransferREWA (NO SHEAFS)")
    log("="*70)

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"Device: {device}")

    n_train = 1500
    n_test = 300

    log("\nLoading tasks in OPTIMAL order (easy -> hard, grouped by similarity)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Optimal order: sentiment -> paraphrase -> entailment -> grammar -> topic
    tasks = [
        load_sst2(tokenizer, n_train, n_test),      # 1. Simple binary sentiment
        load_imdb(tokenizer, n_train, n_test),      # 2. Binary sentiment, longer
        load_yelp(tokenizer, n_train, n_test),      # 3. Binary sentiment, reviews
        load_mrpc(tokenizer, n_train, n_test),      # 4. Paraphrase (small)
        load_qqp(tokenizer, n_train, n_test),       # 5. Paraphrase (large)
        load_rte(tokenizer, n_train, n_test),       # 6. Entailment (2-class)
        load_qnli(tokenizer, n_train, n_test),      # 7. Question entailment
        load_mnli(tokenizer, n_train, n_test),      # 8. Multi-genre entailment (3-class)
        load_cola(tokenizer, n_train, n_test),      # 9. Linguistic acceptability
        load_ag_news(tokenizer, n_train, n_test),   # 10. Topic (4-class, most different)
    ]

    log(f"\nTask sequence ({len(tasks)} tasks):")
    for i, (_, _, nc, name) in enumerate(tasks):
        log(f"  {i+1}. {name} ({nc} classes)")

    results = run_benchmark(
        tasks=tasks,
        device=device,
        epochs_per_task=2,
        batch_size=16,
        lr=2e-5
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimal_order_results_{timestamp}.json"

    json_results = {
        'model': results['model'],
        'num_tasks': len(tasks),
        'task_order': [t[3] for t in tasks],
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
