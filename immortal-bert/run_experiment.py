#!/usr/bin/env python3
"""
Immortal BERT: Continual Domain Adaptation Without Forgetting

Experiment:
- Train DistilBERT sequentially on multiple text classification tasks
- Compare: Baseline (catastrophic forgetting) vs EWC vs REWA-C

Tasks (in order):
1. AG News (4-class news classification)
2. IMDB (2-class sentiment)
3. SST-2 (2-class sentiment, different distribution)
4. Yelp (2-class sentiment, reviews)

Metrics:
- Accuracy on each task after training on all tasks
- Forward transfer (how well does prior learning help?)
- Backward transfer (how much is forgotten?)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

from src.rewa_bert import REWABert, BaselineBert, EWCBert, train_epoch, evaluate


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
    """Load AG News dataset (4-class news classification)."""
    print("Loading AG News...")
    dataset = load_dataset('ag_news')

    texts_train = dataset['train']['text'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['test']['text'][:n_test]
    labels_test = dataset['test']['label'][:n_test]

    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)

    return train_ds, test_ds, 4, "AG_News"


def load_imdb(tokenizer, n_train=2000, n_test=500):
    """Load IMDB dataset (2-class sentiment)."""
    print("Loading IMDB...")
    dataset = load_dataset('imdb')

    texts_train = dataset['train']['text'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['test']['text'][:n_test]
    labels_test = dataset['test']['label'][:n_test]

    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)

    return train_ds, test_ds, 2, "IMDB"


def load_sst2(tokenizer, n_train=2000, n_test=500):
    """Load SST-2 dataset (2-class sentiment)."""
    print("Loading SST-2...")
    dataset = load_dataset('glue', 'sst2')

    texts_train = dataset['train']['sentence'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['validation']['sentence'][:n_test]
    labels_test = dataset['validation']['label'][:n_test]

    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)

    return train_ds, test_ds, 2, "SST2"


def load_yelp(tokenizer, n_train=2000, n_test=500):
    """Load Yelp dataset (2-class sentiment)."""
    print("Loading Yelp...")
    dataset = load_dataset('yelp_polarity')

    texts_train = dataset['train']['text'][:n_train]
    labels_train = dataset['train']['label'][:n_train]
    texts_test = dataset['test']['text'][:n_test]
    labels_test = dataset['test']['label'][:n_test]

    train_ds = TextDataset(texts_train, labels_train, tokenizer)
    test_ds = TextDataset(texts_test, labels_test, tokenizer)

    return train_ds, test_ds, 2, "Yelp"


def run_continual_learning(
    model_class,
    model_kwargs,
    tasks,
    device,
    epochs_per_task=3,
    batch_size=16,
    lr=2e-5,
    model_name="Model"
):
    """
    Run continual learning experiment.

    Args:
        model_class: Model class to instantiate
        model_kwargs: Kwargs for model initialization
        tasks: List of (train_loader, test_loader, num_classes, task_name)
        device: Device to train on
        epochs_per_task: Epochs per task
        batch_size: Batch size
        lr: Learning rate
        model_name: Name for logging

    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Running: {model_name}")
    print(f"{'='*60}")

    results = {
        'model': model_name,
        'task_accuracies': defaultdict(list),  # task_name -> [acc_after_task1, acc_after_task2, ...]
        'final_accuracies': {},
    }

    model = None
    completed_tasks = []

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        print(f"\n--- Task {task_idx + 1}: {task_name} ({num_classes} classes) ---")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Initialize or update model
        if model is None:
            model = model_class(num_classes=num_classes, **model_kwargs).to(device)
        else:
            model.update_classifier(num_classes)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        # Train
        for epoch in range(epochs_per_task):
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"  Epoch {epoch + 1}/{epochs_per_task}: Loss={loss:.4f}, Acc={acc:.1%}")

        # Store geometry for REWA model
        if hasattr(model, 'store_task_geometry'):
            model.store_task_geometry(train_loader, task_name)

        # Compute Fisher for EWC model
        if hasattr(model, 'compute_fisher'):
            model.compute_fisher(train_loader)

        completed_tasks.append((test_loader, num_classes, task_name))

        # Evaluate on ALL completed tasks
        print(f"\n  Evaluating on all tasks:")
        for prev_loader, prev_classes, prev_name in completed_tasks:
            # Temporarily update classifier if needed
            curr_classes = model.num_classes
            model.update_classifier(prev_classes)

            acc = evaluate(model, prev_loader, device)
            results['task_accuracies'][prev_name].append(acc)
            print(f"    {prev_name}: {acc:.1%}")

            # Restore classifier
            model.update_classifier(curr_classes)

    # Final accuracies
    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final
    print(f"\n  Average final accuracy: {avg_final:.1%}")

    return results


def main():
    print("="*70)
    print("IMMORTAL BERT: Continual Domain Adaptation Without Forgetting")
    print("="*70)

    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"\nDevice: {device}")

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load datasets (smaller subsets for faster experimentation)
    n_train = 1500  # Samples per task
    n_test = 300

    tasks = [
        load_ag_news(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_sst2(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
    ]

    # Experiment parameters
    epochs_per_task = 2
    batch_size = 16
    lr = 2e-5

    all_results = []

    # 1. Baseline (no protection)
    baseline_results = run_continual_learning(
        model_class=BaselineBert,
        model_kwargs={},
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Baseline (No Protection)"
    )
    all_results.append(baseline_results)

    # 2. EWC
    ewc_results = run_continual_learning(
        model_class=EWCBert,
        model_kwargs={'ewc_lambda': 1000.0},
        tasks=tasks,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="EWC (λ=1000)"
    )
    all_results.append(ewc_results)

    # 3. REWA-C (ours) - test multiple λ values
    for lambda_geom in [0.1, 1.0, 10.0]:
        rewa_results = run_continual_learning(
            model_class=REWABert,
            model_kwargs={'lambda_geom': lambda_geom},
            tasks=tasks,
            device=device,
            epochs_per_task=epochs_per_task,
            batch_size=batch_size,
            lr=lr,
            model_name=f"REWA-C (λ={lambda_geom})"
        )
        all_results.append(rewa_results)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nFinal accuracies by task:")
    task_names = list(tasks[0][3] for tasks in [tasks])
    task_names = [t[3] for t in tasks]

    header = f"{'Method':<30}"
    for task_name in task_names:
        header += f" {task_name:>10}"
    header += f" {'Average':>10}"
    print(header)
    print("-" * len(header))

    for result in all_results:
        row = f"{result['model']:<30}"
        for task_name in task_names:
            acc = result['final_accuracies'].get(task_name, 0)
            row += f" {acc:>10.1%}"
        row += f" {result['average_final_accuracy']:>10.1%}"
        print(row)

    # Forgetting analysis
    print("\n\nForgetting Analysis (accuracy drop from peak):")
    for result in all_results:
        forgetting = []
        for task_name, accs in result['task_accuracies'].items():
            if len(accs) > 1:
                peak = max(accs[:-1]) if len(accs) > 1 else accs[0]
                final = accs[-1]
                forgetting.append(peak - final)

        avg_forgetting = np.mean(forgetting) if forgetting else 0
        print(f"  {result['model']}: {avg_forgetting:.1%} average forgetting")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{timestamp}.json"

    # Convert to JSON-serializable format
    json_results = []
    for r in all_results:
        jr = {
            'model': r['model'],
            'task_accuracies': {k: [float(x) for x in v] for k, v in r['task_accuracies'].items()},
            'final_accuracies': {k: float(v) for k, v in r['final_accuracies'].items()},
            'average_final_accuracy': float(r['average_final_accuracy'])
        }
        json_results.append(jr)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
