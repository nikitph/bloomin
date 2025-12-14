#!/usr/bin/env python3
"""
Immortal BERT: Continual Domain Adaptation Without Forgetting

Experiment:
- Train DistilBERT/BERT sequentially on multiple text classification tasks
- Compare: Baseline vs EWC vs REWA-C vs Subspace-REWA (New)

Tasks (in order):
1. AG News (4-class news classification)
2. IMDB (2-class sentiment)
3. SST-2 (2-class sentiment, different distribution)
4. Yelp (2-class sentiment, reviews)

Metrics:
- Accuracy on each task after training on all tasks
- Worst-Task Accuracy (WTA)
- Forgetting Index (FI)
- Performance Variance (PV)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from typing import Dict, List

from src.rewa_bert import REWABert, BaselineBert, EWCBert, train_epoch, evaluate
from src.subspace_rewa import SubspaceREWABert


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


# --- METRICS CALCULATIONS ---

def calculate_wta(task_accuracies: Dict[str, List[float]]) -> float:
    """Worst-Task Accuracy (final)."""
    if not task_accuracies:
        return 0.0
    final_accs = [accs[-1] for accs in task_accuracies.values()]
    return min(final_accs)

def calculate_fi(task_accuracies: Dict[str, List[float]]) -> float:
    """Forgetting Index."""
    forgettings = []
    # Skip the last task because it hasn't been "forgotten" yet?
    # FI formula: Avg over T-1 tasks of (Max_k<=t Acc_k - Acc_T)
    # i.e. Peak accuracy minus Final accuracy
    for task_name, accs in task_accuracies.items():
        if len(accs) > 1:
            peak = max(accs[:-1])
            final = accs[-1]
            forgettings.append(peak - final)
        else:
            # If only trained on one task, no forgetting
            pass
            
    if not forgettings:
        return 0.0
    return np.mean(forgettings)

def calculate_pv(task_accuracies: Dict[str, List[float]]) -> float:
    """Performance Variance (of final accuracies)."""
    final_accs = [accs[-1] for accs in task_accuracies.values()]
    return np.var(final_accs)


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
    print(f"\n{'='*60}")
    print(f"Running: {model_name}")
    print(f"{'='*60}")

    results = {
        'model': model_name,
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
    }

    model = None
    completed_tasks = []
    # Store classifier heads: task_name -> dict(weight, bias)
    task_heads = {}

    for task_idx, (train_ds, test_ds, num_classes, task_name) in enumerate(tasks):
        print(f"\n--- Task {task_idx + 1}: {task_name} ({num_classes} classes) ---")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Initialize or update model
        if model is None:
            model = model_class(num_classes=num_classes, **model_kwargs).to(device)
        else:
            model.update_classifier(num_classes)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        # Train
        for epoch in range(epochs_per_task):
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"  Epoch {epoch + 1}/{epochs_per_task}: Loss={loss:.4f}, Acc={acc:.1%}")

        # Save the trained head for this task
        task_heads[task_name] = {
            'weight': model.classifier.weight.detach().cpu().clone(),
            'bias': model.classifier.bias.detach().cpu().clone()
        }

        # --- SUBSPACE REWA SPECIFIC LIFECYCLE ---
        if isinstance(model, SubspaceREWABert):
             # Identify subspaces after Task 1 (AG News)
             if task_idx == 0:
                 print("  Task 1 complete. Identifying shared subspaces to preserve...")
                 model.compute_subspaces(train_loader, device)

        # Store geometry for classic REWA
        if hasattr(model, 'store_task_geometry'):
            model.store_task_geometry(train_loader, task_name)

        # Compute Fisher for EWC
        if hasattr(model, 'compute_fisher'):
            model.compute_fisher(train_loader)

        completed_tasks.append((test_loader, num_classes, task_name))

        # Evaluate on ALL completed tasks
        print(f"\n  Evaluating on all tasks:")
        # Save current head (just in case training continues or logic changes)
        current_head_w = model.classifier.weight.detach().clone()
        current_head_b = model.classifier.bias.detach().clone()
        current_classes = model.num_classes

        for prev_loader, prev_classes, prev_name in completed_tasks:
            # 1. Resize classifier to match task
            model.update_classifier(prev_classes)
            
            # 2. Load the SAVED head weights
            saved_head = task_heads[prev_name]
            model.classifier.weight.data = saved_head['weight'].to(device)
            model.classifier.bias.data = saved_head['bias'].to(device)

            acc = evaluate(model, prev_loader, device)
            results['task_accuracies'][prev_name].append(acc)
            print(f"    {prev_name}: {acc:.1%}")

        # Restore current task head (though we will likely reset it next loop anyway)
        model.update_classifier(current_classes)
        model.classifier.weight.data = current_head_w
        model.classifier.bias.data = current_head_b


    # Final accuracies
    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))
    results['average_final_accuracy'] = avg_final
    
    # New metrics
    task_accs = results['task_accuracies']
    results['WTA'] = min([accs[-1] for accs in task_accs.values()])
    
    forgettings = []
    for accs in task_accs.values():
        if len(accs) > 1:
            forgettings.append(max(accs[:-1]) - accs[-1])
    results['FI'] = np.mean(forgettings) if forgettings else 0.0
    
    final_values = [accs[-1] for accs in task_accs.values()]
    results['PV'] = np.var(final_values)

    print(f"\n  Average final accuracy: {avg_final:.1%}")
    print(f"  Worst-Task Accuracy:    {results['WTA']:.1%}")
    print(f"  Forgetting Index:       {results['FI']:.1%}")
    print(f"  Performance Variance:   {results['PV']:.4f}")

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

    # Load datasets
    n_train = 1500
    n_test = 300
    
    # Note: Using BertTokenizer for SubspaceREWA (uses bert-base) 
    # and DistilBertTokenizer for others (uses distilbert).
    # For fair comparison, we should try to keep inputs similar, 
    # but the models expect different tokenizers.
    
    # To save time, we load only one tokenizer for the main experiment, 
    # but since we compare DistilBert vs Bert, we need both.
    
    print("\nPreparing tasks for DistilBERT models...")
    tokenizer_distil = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tasks_distil = [
        load_ag_news(tokenizer_distil, n_train, n_test),
        load_imdb(tokenizer_distil, n_train, n_test),
        load_sst2(tokenizer_distil, n_train, n_test),
        load_yelp(tokenizer_distil, n_train, n_test),
    ]
    
    print("\nPreparing tasks for BERT models...")
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    tasks_bert = [
        load_ag_news(tokenizer_bert, n_train, n_test),
        load_imdb(tokenizer_bert, n_train, n_test),
        load_sst2(tokenizer_bert, n_train, n_test),
        load_yelp(tokenizer_bert, n_train, n_test),
    ]

    # Experiment parameters
    epochs_per_task = 2
    batch_size = 16
    lr = 2e-5

    all_results = []

    # 1. Baseline
    baseline_results = run_continual_learning(
        model_class=BaselineBert,
        model_kwargs={},
        tasks=tasks_distil,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Baseline (DistilBERT)"
    )
    all_results.append(baseline_results)

    # 2. EWC
    ewc_results = run_continual_learning(
        model_class=EWCBert,
        model_kwargs={'ewc_lambda': 1000.0},
        tasks=tasks_distil,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="EWC (λ=1000)"
    )
    all_results.append(ewc_results)

    # 3. Subspace-REWA (New)
    print("\nRunning Subspace-REWA (BERT-Base, Layer-wise λ)...")
    subspace_results = run_continual_learning(
        model_class=SubspaceREWABert,
        model_kwargs={
            'lambda_max': 10.0,
            'subspace_dim': 64,
            'layer_focus_start': 7 # Start from layer 7 (0-indexed means layer 8)
        },
        tasks=tasks_bert,
        device=device,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        lr=lr,
        model_name="Subspace-REWA (λ=10, L≥8)"
    )
    all_results.append(subspace_results)

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    header = f"{'Method':<30} {'Avg Acc':<10} {'WTA':<10} {'FI':<10} {'PV':<10}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(f"{r['model']:<30} {r['average_final_accuracy']:<10.1%} "
              f"{r['WTA']:<10.1%} {r['FI']:<10.1%} {r['PV']:<10.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_subspace_{timestamp}.json"
    
    # Make serializable
    json_results = []
    for r in all_results:
        jr = r.copy()
        jr['task_accuracies'] = {k: [float(x) for x in v] for k,v in r['task_accuracies'].items()}
        jr['final_accuracies'] = {k: float(v) for k,v in r['final_accuracies'].items()}
        jr['average_final_accuracy'] = float(r['average_final_accuracy'])
        jr['WTA'] = float(r['WTA'])
        jr['FI'] = float(r['FI'])
        jr['PV'] = float(r['PV'])
        json_results.append(jr)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
