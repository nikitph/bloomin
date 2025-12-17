#!/usr/bin/env python3
"""
Maximal Combined REWA GLUE Benchmark

Runs only the Maximal Combined approach (v1+v2+v3) on 10-task GLUE benchmark:
- Task Discriminator (domain adaptation style)
- Witness Adversary (diverse witness usage)
- Contrastive Adversarial (task clustering)

Hypothesis: Combined adversarial training should achieve FI of -3% to -8%
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
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.rewa_bert import evaluate
from src.adversarial_rewa import MaximalAdversarialREWA


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


def run_maximal_combined_benchmark(
    tasks,
    device,
    epochs_per_task=2,
    batch_size=16,
    lr=2e-5
):
    """Run 10-task GLUE benchmark with Maximal Combined Adversarial REWA."""
    log(f"\n{'='*70}")
    log(f"MAXIMAL COMBINED ADVERSARIAL REWA BENCHMARK")
    log(f"Combines: Task Discriminator + Witness Adversary + Contrastive")
    log(f"{'='*70}")

    results = {
        'model': 'Maximal-Combined-Adversarial-REWA',
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
        'lambda_history': [],
    }

    model = None
    completed_tasks = []
    task_heads = {}

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        log(f"\n{'─'*60}")
        log(f"Task {task_idx + 1}/{len(tasks)}: {task_name} ({num_classes} classes)")
        log(f"{'─'*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        if model is None:
            log(f"  Initializing Maximal Combined Adversarial REWA...")
            model = MaximalAdversarialREWA(
                num_classes=num_classes,
                lambda_max=10.0,
                lambda_adv=0.3,
                lambda_witness=0.3,
                lambda_contrast=0.3,
                subspace_dim=64,
                layer_focus_start=7,
                anneal_rate=0.85,
                min_lambda=1.0
            ).to(device)
        else:
            model.update_classifier(num_classes)

        # Set task ID
        if hasattr(model, 'set_task_id'):
            model.set_task_id(task_idx)

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
            log(f"  Computing subspaces...")
            model.compute_subspaces(train_loader, device)
        else:
            # Anneal lambdas
            if hasattr(model, 'anneal_lambdas'):
                model.anneal_lambdas()
                log(f"  Lambda annealed. L11: {model.layer_lambdas[11]:.2f}")

            # Update task-specific info
            if hasattr(model, 'update_task_info'):
                model.update_task_info(train_loader, device)

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

    # Compute forgetting for each task
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
    if hasattr(model, 'lambda_history'):
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

    # Critical check
    log(f"\n{'='*70}")
    if results['FI'] < -0.03:
        log(f"SUCCESS: FI < -3% ({results['FI']:+.2%})")
        log(f"  -> Strong forward transfer confirmed!")
        log(f"  -> Nature MI target is HIGHLY REALISTIC")
    elif results['FI'] <= 0:
        log(f"PARTIAL SUCCESS: FI is negative ({results['FI']:+.2%})")
        log(f"  -> Forward transfer detected")
        log(f"  -> Better than no-adversarial baseline")
    else:
        log(f"FI is positive ({results['FI']:+.2%})")
        log(f"  -> Some forgetting occurred")
        log(f"  -> Consider hyperparameter tuning")
    log(f"{'='*70}")

    return results, model


# ============================================================================
# MAIN
# ============================================================================

def main():
    log("="*70)
    log("MAXIMAL COMBINED ADVERSARIAL REWA - GLUE BENCHMARK")
    log("="*70)
    log("\nRunning combined adversarial approach (v1+v2+v3) on 10 tasks")
    log("Components: Task Discriminator + Witness Adversary + Contrastive\n")

    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"Device: {device}")

    # Load all 10 tasks
    n_train = 1500
    n_test = 300

    log("\nLoading 10 GLUE benchmark tasks...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Same task order as curriculum_expand_refine for comparison
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

    log(f"\nTask sequence ({len(tasks)} tasks):")
    for i, (_, _, nc, name) in enumerate(tasks):
        log(f"  {i+1}. {name} ({nc} classes)")

    # Run benchmark
    results, model = run_maximal_combined_benchmark(
        tasks=tasks,
        device=device,
        epochs_per_task=2,
        batch_size=16,
        lr=2e-5
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"maximal_combined_results_{timestamp}.json"

    # Add task order to results
    task_order = [name for _, _, _, name in tasks]

    json_results = {
        'model': results['model'],
        'num_tasks': len(tasks),
        'task_order': task_order,
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
