#!/usr/bin/env python3
"""
Experiment: Do Ambiguous Classes Move More?

Tests REWA Term I: -2R_{ij}

Hypothesis: Classes with high inter-class confusion (negative curvature regions)
should experience larger centroid displacement during training, while clean/distinct
classes act as anchors.

Prediction: ||Δμ_c|| ∝ local curvature / ambiguity of class c
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import SimpleMLP


class AmbiguityTracker:
    """Track class ambiguity and centroid displacement during training."""

    def __init__(self, model, device='cpu', num_classes=10):
        self.model = model
        self.device = device
        self.num_classes = num_classes

        # Storage for tracking
        self.initial_centroids = None
        self.centroid_history = []  # List of centroid snapshots
        self.ambiguity_scores = None  # Per-class ambiguity
        self.confusion_matrix = None

    def compute_embeddings_by_class(self, dataloader):
        """Get embeddings organized by class."""
        self.model.eval()
        embeddings_by_class = defaultdict(list)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)

                # Get penultimate layer embeddings using model's method
                embeddings = self.model.get_embeddings(batch_x)

                for emb, label in zip(embeddings.cpu(), batch_y):
                    embeddings_by_class[label.item()].append(emb)

        # Stack embeddings
        for c in embeddings_by_class:
            embeddings_by_class[c] = torch.stack(embeddings_by_class[c])

        return embeddings_by_class

    def compute_centroids(self, embeddings_by_class):
        """Compute class centroids from embeddings."""
        centroids = {}
        for c, embs in embeddings_by_class.items():
            centroids[c] = embs.mean(dim=0)
        return centroids

    def compute_ambiguity_scores(self, dataloader):
        """
        Compute per-class ambiguity using multiple measures:
        1. Confusion rate: How often is this class confused with others?
        2. Embedding overlap: How much do embeddings overlap with other classes?
        3. Local curvature proxy: Distance to nearest class centroid
        """
        embeddings_by_class = self.compute_embeddings_by_class(dataloader)
        centroids = self.compute_centroids(embeddings_by_class)

        # Store initial centroids
        self.initial_centroids = {c: cent.clone() for c, cent in centroids.items()}

        ambiguity = {}

        for c in range(self.num_classes):
            if c not in embeddings_by_class:
                ambiguity[c] = 0.0
                continue

            embs = embeddings_by_class[c]
            own_centroid = centroids[c]

            # Measure 1: Average distance to own centroid (intra-class spread)
            intra_spread = torch.norm(embs - own_centroid.unsqueeze(0), dim=1).mean().item()

            # Measure 2: Distance to nearest other centroid
            min_inter_dist = float('inf')
            for other_c, other_cent in centroids.items():
                if other_c != c:
                    dist = torch.norm(own_centroid - other_cent).item()
                    min_inter_dist = min(min_inter_dist, dist)

            # Measure 3: Compute confusion - what fraction of samples are closer to wrong centroid?
            confusion_count = 0
            for emb in embs:
                own_dist = torch.norm(emb - own_centroid).item()
                for other_c, other_cent in centroids.items():
                    if other_c != c:
                        if torch.norm(emb - other_cent).item() < own_dist:
                            confusion_count += 1
                            break
            confusion_rate = confusion_count / len(embs)

            # Combined ambiguity score:
            # High spread + low inter-class distance + high confusion = high ambiguity
            # Ambiguity ∝ (spread / min_inter_dist) * (1 + confusion_rate)
            if min_inter_dist > 0:
                ambiguity[c] = (intra_spread / min_inter_dist) * (1 + confusion_rate)
            else:
                ambiguity[c] = intra_spread * (1 + confusion_rate)

        self.ambiguity_scores = ambiguity
        return ambiguity

    def compute_confusion_matrix(self, dataloader):
        """Compute the actual classification confusion matrix."""
        self.model.eval()
        confusion = torch.zeros(self.num_classes, self.num_classes)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1)

                for true, pred in zip(batch_y, preds):
                    confusion[true.item(), pred.item()] += 1

        # Normalize by row (true class)
        row_sums = confusion.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.confusion_matrix = confusion / row_sums

        return self.confusion_matrix

    def snapshot_centroids(self, dataloader):
        """Take a snapshot of current centroids."""
        embeddings_by_class = self.compute_embeddings_by_class(dataloader)
        centroids = self.compute_centroids(embeddings_by_class)
        self.centroid_history.append({c: cent.clone() for c, cent in centroids.items()})
        return centroids

    def compute_displacements(self):
        """Compute total displacement per class from initial to final."""
        if self.initial_centroids is None or len(self.centroid_history) == 0:
            return None

        final_centroids = self.centroid_history[-1]
        displacements = {}

        for c in self.initial_centroids:
            if c in final_centroids:
                disp = torch.norm(final_centroids[c] - self.initial_centroids[c]).item()
                displacements[c] = disp

        return displacements


def train_with_tracking(model, train_loader, test_loader, tracker, epochs=10, lr=0.001,
                        snapshot_every=1, device='cpu'):
    """Train model while tracking centroid movement."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initial snapshot
    tracker.snapshot_centroids(train_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Snapshot centroids
        if (epoch + 1) % snapshot_every == 0:
            tracker.snapshot_centroids(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

    return model


def analyze_correlation(ambiguity_scores, displacements):
    """Analyze correlation between ambiguity and displacement."""
    classes = sorted(set(ambiguity_scores.keys()) & set(displacements.keys()))

    amb_vals = [ambiguity_scores[c] for c in classes]
    disp_vals = [displacements[c] for c in classes]

    # Pearson correlation
    correlation, p_value = stats.pearsonr(amb_vals, disp_vals)

    # Spearman rank correlation
    spearman_corr, spearman_p = stats.spearmanr(amb_vals, disp_vals)

    return {
        'pearson_r': correlation,
        'pearson_p': p_value,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'ambiguity': dict(zip(classes, amb_vals)),
        'displacement': dict(zip(classes, disp_vals))
    }


def plot_results(results, dataset_name, save_path):
    """Create visualization of ambiguity vs displacement."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    classes = sorted(results['ambiguity'].keys())
    amb = [results['ambiguity'][c] for c in classes]
    disp = [results['displacement'][c] for c in classes]

    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(amb, disp, s=100, alpha=0.7)
    for c, a, d in zip(classes, amb, disp):
        ax1.annotate(str(c), (a, d), fontsize=12, ha='center', va='bottom')

    # Fit line
    z = np.polyfit(amb, disp, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(amb), max(amb), 100)
    ax1.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'r={results["pearson_r"]:.3f}')

    ax1.set_xlabel('Initial Ambiguity Score', fontsize=12)
    ax1.set_ylabel('Centroid Displacement', fontsize=12)
    ax1.set_title(f'{dataset_name}: Ambiguity vs Displacement', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart of ambiguity
    ax2 = axes[1]
    bars = ax2.bar(classes, amb, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Ambiguity Score', fontsize=12)
    ax2.set_title('Initial Class Ambiguity', fontsize=14)
    ax2.set_xticks(classes)

    # Plot 3: Bar chart of displacement
    ax3 = axes[2]
    bars = ax3.bar(classes, disp, color='coral', alpha=0.7)
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_ylabel('Centroid Displacement', fontsize=12)
    ax3.set_title('Centroid Movement During Training', fontsize=14)
    ax3.set_xticks(classes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {save_path}")


def run_experiment(dataset_name, dataset_class, data_root='./data', subset_size=5000,
                   epochs=10, device='cpu'):
    """Run the ambiguity-displacement experiment on a dataset."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {dataset_name}")
    print('='*60)

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = dataset_class(data_root, train=True, download=True, transform=transform)
    test_data = dataset_class(data_root, train=False, download=True, transform=transform)

    if subset_size:
        train_data = Subset(train_data, range(min(subset_size, len(train_data))))
        test_data = Subset(test_data, range(min(subset_size // 5, len(test_data))))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Initialize model and tracker
    model = SimpleMLP()
    tracker = AmbiguityTracker(model, device=device)

    # Step 1: Train briefly to get initial representation
    print("\nPhase 1: Initial training (2 epochs) to establish representation...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Step 2: Measure initial ambiguity
    print("\nPhase 2: Computing initial class ambiguity...")
    ambiguity = tracker.compute_ambiguity_scores(train_loader)
    confusion = tracker.compute_confusion_matrix(test_loader)

    print("\nInitial Ambiguity Scores:")
    for c in sorted(ambiguity.keys()):
        print(f"  Class {c}: {ambiguity[c]:.4f}")

    # Identify most/least ambiguous
    sorted_by_amb = sorted(ambiguity.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMost ambiguous: Class {sorted_by_amb[0][0]} (score: {sorted_by_amb[0][1]:.4f})")
    print(f"Least ambiguous: Class {sorted_by_amb[-1][0]} (score: {sorted_by_amb[-1][1]:.4f})")

    # Step 3: Continue training and track centroids
    print(f"\nPhase 3: Training for {epochs} more epochs while tracking centroids...")
    train_with_tracking(model, train_loader, test_loader, tracker,
                       epochs=epochs, device=device, snapshot_every=2)

    # Step 4: Compute displacements
    print("\nPhase 4: Computing centroid displacements...")
    displacements = tracker.compute_displacements()

    print("\nCentroid Displacements:")
    for c in sorted(displacements.keys()):
        print(f"  Class {c}: {displacements[c]:.4f}")

    # Step 5: Analyze correlation
    print("\nPhase 5: Analyzing correlation...")
    results = analyze_correlation(ambiguity, displacements)

    print(f"\n{'='*40}")
    print("CORRELATION RESULTS")
    print('='*40)
    print(f"Pearson r:  {results['pearson_r']:.4f} (p={results['pearson_p']:.4f})")
    print(f"Spearman r: {results['spearman_r']:.4f} (p={results['spearman_p']:.4f})")

    if results['pearson_r'] > 0.5 and results['pearson_p'] < 0.1:
        print("\n*** STRONG POSITIVE CORRELATION ***")
        print("Ambiguous classes DO move more!")
        print("This supports REWA Term I: -2R_ij")
    elif results['pearson_r'] > 0.3:
        print("\n* Moderate positive correlation detected *")
    elif results['pearson_r'] < -0.3:
        print("\n! Negative correlation - unexpected !")
    else:
        print("\n~ Weak correlation ~")

    # Plot results
    os.makedirs('./results', exist_ok=True)
    plot_path = f'./results/ambiguity_displacement_{dataset_name.lower()}.png'
    plot_results(results, dataset_name, plot_path)

    return results, tracker


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    all_results = {}

    # Run on MNIST
    mnist_results, mnist_tracker = run_experiment(
        'MNIST', datasets.MNIST,
        subset_size=5000, epochs=10, device=device
    )
    all_results['mnist'] = mnist_results

    # Run on FashionMNIST (likely more ambiguous classes)
    fashion_results, fashion_tracker = run_experiment(
        'FashionMNIST', datasets.FashionMNIST,
        subset_size=5000, epochs=10, device=device
    )
    all_results['fashion'] = fashion_results

    # Run on KMNIST
    kmnist_results, kmnist_tracker = run_experiment(
        'KMNIST', datasets.KMNIST,
        subset_size=5000, epochs=10, device=device
    )
    all_results['kmnist'] = kmnist_results

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: AMBIGUITY-DISPLACEMENT CORRELATION")
    print("="*80)
    print("\nREWA Prediction: ||Δμ_c|| ∝ class ambiguity (curvature)")
    print("-"*80)

    print(f"\n{'Dataset':<15} {'Pearson r':<12} {'p-value':<12} {'Spearman r':<12} {'Verdict':<20}")
    print("-"*80)

    support_count = 0
    for name, results in all_results.items():
        r = results['pearson_r']
        p = results['pearson_p']
        sr = results['spearman_r']

        if r > 0.5 and p < 0.1:
            verdict = "STRONG SUPPORT"
            support_count += 1
        elif r > 0.3 and p < 0.2:
            verdict = "Moderate support"
            support_count += 0.5
        elif r > 0:
            verdict = "Weak positive"
        else:
            verdict = "No support"

        print(f"{name:<15} {r:<12.4f} {p:<12.4f} {sr:<12.4f} {verdict:<20}")

    print("\n" + "="*80)
    if support_count >= 2:
        print("*** REWA HYPOTHESIS SUPPORTED ***")
        print("Ambiguous classes (negative curvature regions) show greater displacement!")
        print("This is direct evidence of Ricci smoothing in neural network representation space.")
    elif support_count >= 1:
        print("* Partial support for REWA hypothesis *")
        print("Some datasets show the expected ambiguity-displacement correlation.")
    else:
        print("Insufficient evidence for REWA hypothesis in this experiment.")
    print("="*80)

    # Save summary
    import json
    with open('./results/ambiguity_displacement_summary.json', 'w') as f:
        # Convert to serializable format
        serializable = {}
        for name, results in all_results.items():
            serializable[name] = {
                'pearson_r': float(results['pearson_r']),
                'pearson_p': float(results['pearson_p']),
                'spearman_r': float(results['spearman_r']),
                'spearman_p': float(results['spearman_p']),
                'ambiguity': {str(k): float(v) for k, v in results['ambiguity'].items()},
                'displacement': {str(k): float(v) for k, v in results['displacement'].items()}
            }
        json.dump(serializable, f, indent=2)
    print("\nResults saved to ./results/ambiguity_displacement_summary.json")


if __name__ == "__main__":
    main()
