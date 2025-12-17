#!/usr/bin/env python3
"""
4-Task Sheaf Test: SST2, IMDB, Yelp, AG_News only.

Testing if Sentiment sheaf gets negative FI with fewer tasks.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

from run_multisheaf_rewa_benchmark import (
    load_sst2, load_imdb, load_yelp, load_ag_news,
    MultiSheafREWA, train_epoch, get_task_sheaf,
    SEMANTIC_SHEAVES, log
)
from transformers import get_linear_schedule_with_warmup
from src.rewa_bert import evaluate


def run_4task_test(device, epochs_per_task=2, batch_size=16, lr=2e-5):
    """Run 4-task benchmark: SST2, IMDB, Yelp, AG_News."""

    log(f"\n{'='*70}")
    log(f"4-TASK SHEAF TEST: Sentiment + Topic Only")
    log(f"{'='*70}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    n_train = 1500
    n_test = 300

    tasks = [
        load_sst2(tokenizer, n_train, n_test),
        load_imdb(tokenizer, n_train, n_test),
        load_yelp(tokenizer, n_train, n_test),
        load_ag_news(tokenizer, n_train, n_test),
    ]

    log(f"\nTask order: {[t[3] for t in tasks]}")

    results = {
        'model': '4-Task-MultiSheaf-Test',
        'task_order': [t[3] for t in tasks],
        'task_accuracies': defaultdict(list),
        'final_accuracies': {},
        'per_task_forgetting': {},
    }

    model = None
    completed_tasks = []

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
        log(f"Sheaf: {sheaf_name}")
        log(f"{'─'*60}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        if sheaf:
            sheaf.task_train_datasets[task_name] = train_ds

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

        model.active_sheaf = sheaf

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs_per_task
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        for epoch in range(epochs_per_task):
            desc = f"T{task_idx+1} E{epoch+1}"
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device, desc=desc)
            log(f"  Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.1%}")

        if sheaf:
            sheaf.task_heads[task_name] = {
                'weight': model.classifier.weight.detach().cpu().clone(),
                'bias': model.classifier.bias.detach().cpu().clone()
            }

        if sheaf:
            if not sheaf.initialized:
                model.initialize_sheaf(sheaf, train_loader, device)
            else:
                model.anneal_sheaf_lambdas(sheaf)
                log(f"  {sheaf.name} lambdas annealed (task {sheaf.tasks_completed})")
                model.expand_sheaf_subspace(sheaf, train_loader, device)

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

    # Compute metrics
    log(f"\n{'='*70}")
    log(f"FINAL METRICS")
    log(f"{'='*70}")

    for task_name, accs in results['task_accuracies'].items():
        results['final_accuracies'][task_name] = accs[-1]

    avg_final = np.mean(list(results['final_accuracies'].values()))

    forgettings = []
    for task_name, accs in results['task_accuracies'].items():
        if len(accs) > 1:
            max_acc = max(accs[:-1])
            final_acc = accs[-1]
            forgetting = max_acc - final_acc
            results['per_task_forgetting'][task_name] = forgetting
            forgettings.append(forgetting)

    fi_global = np.mean(forgettings) if forgettings else 0.0

    # Sentiment sheaf FI
    sentiment_forgettings = []
    for task_name in ["SST2", "IMDB", "Yelp"]:
        if task_name in results['per_task_forgetting']:
            sentiment_forgettings.append(results['per_task_forgetting'][task_name])
    sentiment_fi = np.mean(sentiment_forgettings) if sentiment_forgettings else 0.0

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
    log(f"  Global FI:              {fi_global:+.2%}")
    log(f"  Sentiment Sheaf FI:     {sentiment_fi:+.2%}")
    log(f"{'='*70}")

    return results


def main():
    log("="*70)
    log("4-TASK SHEAF TEST")
    log("="*70)
    log("\nTesting: SST2 -> IMDB -> Yelp -> AG_News")
    log("Question: Does sentiment sheaf get negative FI with only 4 tasks?\n")

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log(f"Device: {device}")

    results = run_4task_test(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"4task_sheaf_test_{timestamp}.json"

    json_results = {
        'model': results['model'],
        'task_order': results['task_order'],
        'task_accuracies': {k: [float(x) for x in v] for k, v in results['task_accuracies'].items()},
        'final_accuracies': {k: float(v) for k, v in results['final_accuracies'].items()},
        'per_task_forgetting': {k: float(v) for k, v in results['per_task_forgetting'].items()},
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    log(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
