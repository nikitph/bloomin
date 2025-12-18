"""
Proof of Concept Validation: Operator-Guided Training

This experiment validates the core claim:
    "Operator-guided training achieves seed-independent convergence
     by learning the transport operator, not just following gradients."

What we test:
1. Multiple seeds with standard Adam → different local minima (varying loss)
2. Multiple seeds with operator guidance → same functional class (consistent loss)
3. Spectral gap as convergence indicator → phase transitions detected
4. Representation dynamics → successful escapes identified

The key metric is NOT final loss, but CONSISTENCY across seeds.
If operator guidance works, different seeds should reach equivalent solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os

from models import MultiWellMLP, DeepNonConvexMLP, create_representation_extractor
from operator_optimizer import OperatorGuidedTrainer, StandardTrainer


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_multimodal_dataset(
    n_samples: int = 2000,
    input_dim: int = 20,
    n_clusters: int = 5,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a classification dataset with multiple modes.

    This creates a challenging dataset where:
    - Multiple clusters per class
    - Non-linearly separable boundaries
    - Noise to prevent trivial solutions

    This is designed to have multiple local minima in the loss landscape.
    """
    np.random.seed(seed)

    X_list = []
    y_list = []

    n_per_cluster = n_samples // n_clusters

    for cluster_id in range(n_clusters):
        # Random cluster center
        center = np.random.randn(input_dim) * 2

        # Generate points around center
        X_cluster = center + np.random.randn(n_per_cluster, input_dim) * noise

        # Binary label based on cluster (creates non-linear boundary)
        label = cluster_id % 2

        X_list.append(X_cluster)
        y_list.append(np.full(n_per_cluster, label))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    return torch.from_numpy(X), torch.from_numpy(y)


def create_regression_dataset(
    n_samples: int = 2000,
    input_dim: int = 20,
    output_dim: int = 5,
    complexity: int = 3,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a regression dataset with complex target function.

    The target is a composition of non-linear functions,
    creating multiple equivalent solutions.
    """
    np.random.seed(seed)

    X = np.random.randn(n_samples, input_dim).astype(np.float32)

    # Complex target: composition of transforms
    y = X[:, :output_dim].copy()

    for _ in range(complexity):
        # Non-linear transform
        y = np.sin(y) + 0.5 * np.tanh(y @ np.random.randn(output_dim, output_dim).astype(np.float32))

    # Add noise
    y += np.random.randn(*y.shape).astype(np.float32) * noise

    return torch.from_numpy(X), torch.from_numpy(y)


def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    epochs: int = 100,
    lr: float = 1e-3
) -> Dict:
    """Train with standard Adam optimizer."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = StandardTrainer(model, optimizer, device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            result = trainer.train_step(batch_x, batch_y, loss_fn)
            epoch_loss += result['loss']
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                val_loss += loss_fn(output, batch_y).item()
                n_val += 1
        val_losses.append(val_loss / n_val)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }


def train_operator_guided(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    hidden_dim: int,
    device: str,
    epochs: int = 100,
    lr: float = 1e-3
) -> Dict:
    """Train with operator-guided optimizer."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = OperatorGuidedTrainer(
        model=model,
        optimizer=optimizer,
        hidden_dim=hidden_dim,
        device=device,
        update_frequency=10
    )

    rep_extractor = create_representation_extractor('mlp')

    train_losses = []
    val_losses = []
    spectral_gaps = []
    phases = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        epoch_gaps = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            result = trainer.train_step(
                batch_x, batch_y, loss_fn, rep_extractor
            )
            epoch_loss += result['loss']
            n_batches += 1

            if 'spectral_gap' in result:
                epoch_gaps.append(result['spectral_gap'])

        train_losses.append(epoch_loss / n_batches)
        spectral_gaps.append(np.mean(epoch_gaps) if epoch_gaps else 0)
        phases.append(result.get('phase', 'UNKNOWN'))

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                val_loss += loss_fn(output, batch_y).item()
                n_val += 1
        val_losses.append(val_loss / n_val)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'spectral_gaps': spectral_gaps,
        'phases': phases,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'history': trainer.get_history()
    }


def run_seed_comparison(
    seeds: List[int],
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    hidden_dim: int,
    device: str,
    epochs: int = 100,
    lr: float = 1e-3
) -> Dict:
    """
    Run comparison across multiple seeds.

    The key validation: operator guidance should show less variance
    in final loss across seeds.
    """
    standard_results = []
    operator_results = []

    print("\n" + "="*60)
    print("SEED COMPARISON EXPERIMENT")
    print("="*60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Standard training
        print("  Training with standard Adam...")
        set_seed(seed)
        model_std = model_factory()
        std_result = train_standard(
            model_std, train_loader, val_loader,
            loss_fn, device, epochs, lr
        )
        standard_results.append(std_result)
        print(f"    Final val loss: {std_result['final_val_loss']:.4f}")

        # Operator-guided training
        print("  Training with operator guidance...")
        set_seed(seed)
        model_op = model_factory()
        op_result = train_operator_guided(
            model_op, train_loader, val_loader,
            loss_fn, hidden_dim, device, epochs, lr
        )
        operator_results.append(op_result)
        print(f"    Final val loss: {op_result['final_val_loss']:.4f}")
        print(f"    Final spectral gap: {op_result['spectral_gaps'][-1]:.4f}")
        print(f"    Final phase: {op_result['phases'][-1]}")

    # Compute statistics
    std_final_losses = [r['final_val_loss'] for r in standard_results]
    op_final_losses = [r['final_val_loss'] for r in operator_results]

    return {
        'standard': {
            'results': standard_results,
            'mean_loss': np.mean(std_final_losses),
            'std_loss': np.std(std_final_losses),
            'min_loss': np.min(std_final_losses),
            'max_loss': np.max(std_final_losses)
        },
        'operator': {
            'results': operator_results,
            'mean_loss': np.mean(op_final_losses),
            'std_loss': np.std(op_final_losses),
            'min_loss': np.min(op_final_losses),
            'max_loss': np.max(op_final_losses)
        }
    }


def compute_representation_similarity(
    models: List[nn.Module],
    test_data: torch.Tensor,
    device: str
) -> np.ndarray:
    """
    Compute pairwise representation similarity between models.

    If operator guidance works, models trained with different seeds
    should produce similar representations (same function class).
    """
    representations = []

    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(test_data.to(device))
            rep = model.get_representations()
            representations.append(rep.cpu().numpy())

    n_models = len(representations)
    similarity = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            # Centered kernel alignment (CKA) approximation
            rep_i = representations[i]
            rep_j = representations[j]

            # Normalize
            rep_i = rep_i - rep_i.mean(axis=0)
            rep_j = rep_j - rep_j.mean(axis=0)

            # Compute similarity
            cov_ij = np.trace(rep_i @ rep_j.T)
            cov_ii = np.sqrt(np.trace(rep_i @ rep_i.T))
            cov_jj = np.sqrt(np.trace(rep_j @ rep_j.T))

            if cov_ii > 0 and cov_jj > 0:
                similarity[i, j] = cov_ij / (cov_ii * cov_jj)
            else:
                similarity[i, j] = 0

    return similarity


def plot_results(comparison_results: Dict, output_dir: str):
    """Generate visualization of results."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss variance comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training curves
    ax = axes[0]
    for i, result in enumerate(comparison_results['standard']['results']):
        ax.plot(result['val_losses'], alpha=0.5, color='blue',
                label='Standard' if i == 0 else None)
    for i, result in enumerate(comparison_results['operator']['results']):
        ax.plot(result['val_losses'], alpha=0.5, color='red',
                label='Operator' if i == 0 else None)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Curves Across Seeds')
    ax.legend()

    # Final loss distribution
    ax = axes[1]
    std_losses = [r['final_val_loss'] for r in comparison_results['standard']['results']]
    op_losses = [r['final_val_loss'] for r in comparison_results['operator']['results']]

    x = [0, 1]
    ax.boxplot([std_losses, op_losses], positions=x)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard\nAdam', 'Operator\nGuided'])
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Loss Variance Across Seeds')

    # Add variance annotations
    std_var = np.std(std_losses)
    op_var = np.std(op_losses)
    ax.text(0, max(std_losses), f'σ={std_var:.4f}', ha='center')
    ax.text(1, max(op_losses), f'σ={op_var:.4f}', ha='center')

    # Spectral gap evolution
    ax = axes[2]
    for result in comparison_results['operator']['results']:
        if 'spectral_gaps' in result:
            ax.plot(result['spectral_gaps'], alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Gap')
    ax.set_title('Spectral Gap Evolution')
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Phase 2→3')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Phase 3→4')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    """Run the POC validation experiment."""
    print("="*60)
    print("OPERATOR-GUIDED TRAINING: PROOF OF CONCEPT VALIDATION")
    print("="*60)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Dataset parameters
    n_samples = 2000
    input_dim = 20
    n_classes = 2

    # Model parameters
    hidden_dim = 64

    # Training parameters
    epochs = 50
    batch_size = 64
    lr = 1e-3

    # Seeds to test
    seeds = [42, 123, 456, 789, 1000]

    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Seeds: {seeds}")

    # Create dataset
    print("\nCreating dataset...")
    X, y = create_multimodal_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        n_clusters=10,  # More clusters = more complex boundary
        noise=0.3,
        seed=42
    )

    # Split
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Model factory
    def model_factory():
        return DeepNonConvexMLP(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim * 2, hidden_dim),
            output_dim=n_classes,
            use_skip=True,
            dropout=0.2
        )

    # Run comparison
    results = run_seed_comparison(
        seeds=seeds,
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        hidden_dim=hidden_dim,
        device=device,
        epochs=epochs,
        lr=lr
    )

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nStandard Adam:")
    print(f"  Mean loss: {results['standard']['mean_loss']:.4f}")
    print(f"  Std loss:  {results['standard']['std_loss']:.4f}")
    print(f"  Range:     [{results['standard']['min_loss']:.4f}, {results['standard']['max_loss']:.4f}]")

    print("\nOperator-Guided:")
    print(f"  Mean loss: {results['operator']['mean_loss']:.4f}")
    print(f"  Std loss:  {results['operator']['std_loss']:.4f}")
    print(f"  Range:     [{results['operator']['min_loss']:.4f}, {results['operator']['max_loss']:.4f}]")

    # Key metric: variance reduction
    variance_reduction = (
        (results['standard']['std_loss'] - results['operator']['std_loss']) /
        results['standard']['std_loss'] * 100
        if results['standard']['std_loss'] > 0 else 0
    )

    print(f"\n*** Variance Reduction: {variance_reduction:.1f}% ***")

    if variance_reduction > 0:
        print("✓ Operator guidance REDUCED variance across seeds")
        print("  → This supports seed-independent convergence")
    else:
        print("✗ Operator guidance did not reduce variance")
        print("  → Need to tune parameters or longer training")

    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    save_results = {
        'config': {
            'n_samples': n_samples,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'seeds': seeds
        },
        'standard': {
            'mean_loss': float(results['standard']['mean_loss']),
            'std_loss': float(results['standard']['std_loss']),
            'losses': [float(r['final_val_loss']) for r in results['standard']['results']]
        },
        'operator': {
            'mean_loss': float(results['operator']['mean_loss']),
            'std_loss': float(results['operator']['std_loss']),
            'losses': [float(r['final_val_loss']) for r in results['operator']['results']]
        },
        'variance_reduction_pct': float(variance_reduction),
        'timestamp': datetime.now().isoformat()
    }

    with open(f'{output_dir}/poc_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Generate plots
    plot_results(results, output_dir)

    print(f"\nResults saved to {output_dir}/")
    print("\n" + "="*60)
    print("POC VALIDATION COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
