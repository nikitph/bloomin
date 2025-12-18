"""
Proof of Concept Validation V2: Harder Problem

The previous dataset was too easy - both methods converged perfectly.
This version uses a harder problem:
1. XOR-like structure (non-linearly separable)
2. Higher noise
3. More complex decision boundary
4. Track representation dynamics more carefully
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

from models import DeepNonConvexMLP, create_representation_extractor
from representation_buffer import RepresentationBuffer
from spectral_regulator import SpectralRegulator, TrainingPhase


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_xor_spiral_dataset(
    n_samples: int = 3000,
    noise: float = 0.5,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create XOR-like spiral dataset that's genuinely hard.

    This creates a non-linearly separable problem with multiple
    valid decision boundaries - perfect for testing whether
    different seeds find different local minima.
    """
    np.random.seed(seed)

    n_per_class = n_samples // 2

    # Spiral 1 (class 0)
    theta1 = np.linspace(0, 4*np.pi, n_per_class) + np.random.randn(n_per_class) * noise
    r1 = theta1 / (4*np.pi) * 2 + np.random.randn(n_per_class) * noise * 0.3
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # Spiral 2 (class 1) - offset by pi
    theta2 = np.linspace(0, 4*np.pi, n_per_class) + np.pi + np.random.randn(n_per_class) * noise
    r2 = theta2 / (4*np.pi) * 2 + np.random.randn(n_per_class) * noise * 0.3
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)

    # Combine and add extra dimensions
    X0 = np.stack([x1, y1], axis=1)
    X1 = np.stack([x2, y2], axis=1)

    # Add extra noise dimensions
    extra_dims = 18
    X0_extra = np.random.randn(n_per_class, extra_dims) * 0.5
    X1_extra = np.random.randn(n_per_class, extra_dims) * 0.5

    X0 = np.hstack([X0, X0_extra])
    X1 = np.hstack([X1, X1_extra])

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    return torch.from_numpy(X), torch.from_numpy(y)


def create_checkerboard_dataset(
    n_samples: int = 3000,
    grid_size: int = 4,
    noise: float = 0.3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create checkerboard pattern - classic hard classification.

    Multiple equivalent solutions exist, making this ideal for
    testing seed-independent convergence.
    """
    np.random.seed(seed)

    X = np.random.rand(n_samples, 2) * grid_size
    X += np.random.randn(n_samples, 2) * noise

    # Checkerboard label
    y = ((X[:, 0].astype(int) + X[:, 1].astype(int)) % 2).astype(np.int64)

    # Add extra dimensions
    extra_dims = 18
    X_extra = np.random.randn(n_samples, extra_dims) * 0.5
    X = np.hstack([X, X_extra]).astype(np.float32)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    return torch.from_numpy(X), torch.from_numpy(y)


class OperatorGuidedTrainerV2:
    """
    Improved operator-guided trainer with proper representation tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        hidden_dim: int,
        device: str = 'cpu',
        update_frequency: int = 5
    ):
        self.model = model
        self.base_optimizer = optimizer
        self.hidden_dim = hidden_dim
        self.device = device
        self.update_frequency = update_frequency

        # Representation tracking
        self.rep_buffer = RepresentationBuffer(
            hidden_dim=hidden_dim,
            buffer_size=500,
            device=device
        )

        # Spectral regulation
        self.spectral_regulator = SpectralRegulator(
            hidden_dim=hidden_dim,
            device=device
        )

        self.step_count = 0
        self.history = {
            'loss': [],
            'spectral_gap': [],
            'phase': [],
            'escape_magnitude': [],
            'reynolds': []
        }

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        loss_fn
    ) -> Dict:
        self.model.train()
        self.base_optimizer.zero_grad()
        self.step_count += 1

        # Forward pass
        output = self.model(batch_x)
        representations = self.model.get_representations()

        # Compute loss
        loss = loss_fn(output, batch_y)

        # Store representations BEFORE backward (captures current state)
        if representations is not None:
            self.rep_buffer.store(
                representations.detach(),
                loss.item(),
                self.step_count
            )

        # Backward pass
        loss.backward()

        # Periodically update operator state and apply drift
        spectral_gap = 0.0
        phase = 'DIFFUSION_DOMINATED'
        escape_mag = 0.0
        reynolds = 0.0

        if self.step_count % self.update_frequency == 0 and self.step_count > 20:
            # Update escape momentum
            self.rep_buffer.update_escape_momentum(ema_decay=0.9, window=30)

            # Get spectral state
            cov = self.rep_buffer.compute_representation_covariance(window=50)
            state = self.spectral_regulator.update(
                cov,
                escape_magnitude=self.rep_buffer.escape_magnitude
            )

            spectral_gap = state.spectral_gap
            phase = state.phase.name
            escape_mag = self.rep_buffer.escape_magnitude
            reynolds = state.reynolds_number

            # Apply representation drift to gradients
            self._apply_drift_to_gradients(state.reynolds_number)

        # Optimizer step
        self.base_optimizer.step()

        # Record history
        self.history['loss'].append(loss.item())
        self.history['spectral_gap'].append(spectral_gap)
        self.history['phase'].append(phase)
        self.history['escape_magnitude'].append(escape_mag)
        self.history['reynolds'].append(reynolds)

        return {
            'loss': loss.item(),
            'spectral_gap': spectral_gap,
            'phase': phase,
            'escape_magnitude': escape_mag,
            'reynolds': reynolds
        }

    def _apply_drift_to_gradients(self, reynolds: float):
        """
        Apply escape drift by modifying gradients.

        The drift is in representation space, so we add it to the
        last layer's gradient (which directly affects representations).
        """
        if reynolds < 0.01 or self.rep_buffer.escape_magnitude < 0.01:
            return

        drift = self.rep_buffer.get_escape_drift(scale=reynolds * 0.1)

        # Find the last hidden layer and modify its gradient
        for name, param in reversed(list(self.model.named_parameters())):
            if param.grad is not None and 'output' not in name.lower():
                # Add drift to gradient (scaled appropriately)
                # Handle different gradient shapes
                if len(param.grad.shape) == 1 and param.grad.shape[0] == self.hidden_dim:
                    # Bias term
                    param.grad.add_(drift * 0.01)
                    break
                elif len(param.grad.shape) == 2 and param.grad.shape[0] == self.hidden_dim:
                    # Weight matrix (hidden_dim, in_dim)
                    drift_expanded = drift.unsqueeze(1).expand_as(param.grad)
                    param.grad.add_(drift_expanded * 0.01)
                    break
                elif len(param.grad.shape) == 2 and param.grad.shape[1] == self.hidden_dim:
                    # Weight matrix (out_dim, hidden_dim)
                    drift_expanded = drift.unsqueeze(0).expand_as(param.grad)
                    param.grad.add_(drift_expanded * 0.01)
                    break


class StandardTrainerV2:
    """Standard trainer for fair comparison."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {'loss': []}

    def train_step(self, batch_x, batch_y, loss_fn) -> Dict:
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(batch_x)
        loss = loss_fn(output, batch_y)

        loss.backward()
        self.optimizer.step()

        self.history['loss'].append(loss.item())

        return {'loss': loss.item()}


def train_and_evaluate(
    trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    device: str,
    epochs: int,
    is_operator: bool = False
) -> Dict:
    """Train model and return results."""
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Training
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
        trainer.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = trainer.model(batch_x)
                val_loss += loss_fn(output, batch_y).item()

                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct / total)

        # Print progress
        if (epoch + 1) % 10 == 0:
            extra = ""
            if is_operator and hasattr(trainer, 'history'):
                gap = trainer.history['spectral_gap'][-1] if trainer.history['spectral_gap'] else 0
                phase = trainer.history['phase'][-1] if trainer.history['phase'] else 'N/A'
                extra = f" | gap: {gap:.3f} | phase: {phase}"
            print(f"    Epoch {epoch+1}: loss={train_losses[-1]:.4f}, val_acc={val_accs[-1]:.3f}{extra}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_loss': val_losses[-1],
        'final_acc': val_accs[-1],
        'history': trainer.history if hasattr(trainer, 'history') else {}
    }


def compute_model_similarity(models: List[nn.Module], test_x: torch.Tensor, device: str) -> float:
    """
    Compute average pairwise similarity of model representations.

    Higher similarity = more seed-independent convergence.
    """
    representations = []

    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(test_x.to(device))
            rep = model.get_representations().cpu().numpy()
            # Normalize
            rep = rep / (np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8)
            representations.append(rep)

    n = len(representations)
    similarities = []

    for i in range(n):
        for j in range(i+1, n):
            # Compute CKA-like similarity
            sim = np.abs(np.mean(representations[i] * representations[j]))
            similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def run_experiment(dataset_name: str = 'spiral'):
    """Run the full POC experiment."""
    print("="*70)
    print("OPERATOR-GUIDED TRAINING: POC VALIDATION V2")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Configuration
    input_dim = 20
    hidden_dim = 128
    n_classes = 2
    epochs = 100
    batch_size = 32
    lr = 5e-4
    seeds = [42, 123, 456, 789, 1000, 2023, 3141, 9999]

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Seeds: {len(seeds)}")

    # Create dataset
    print("\nCreating dataset...")
    if dataset_name == 'spiral':
        X, y = create_xor_spiral_dataset(n_samples=3000, noise=0.5, seed=42)
    else:
        X, y = create_checkerboard_dataset(n_samples=3000, grid_size=4, noise=0.3, seed=42)

    # Split
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()

    # Model factory
    def make_model():
        return DeepNonConvexMLP(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim*2, hidden_dim),
            output_dim=n_classes,
            use_skip=True,
            dropout=0.3
        )

    # Run experiments
    print("\n" + "="*70)
    print("RUNNING SEED COMPARISON")
    print("="*70)

    std_results = []
    op_results = []
    std_models = []
    op_models = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Standard training
        print("  Standard Adam:")
        set_seed(seed)
        model_std = make_model().to(device)
        optimizer_std = torch.optim.Adam(model_std.parameters(), lr=lr)
        trainer_std = StandardTrainerV2(model_std, optimizer_std, device)

        result_std = train_and_evaluate(
            trainer_std, train_loader, val_loader,
            loss_fn, device, epochs, is_operator=False
        )
        std_results.append(result_std)
        std_models.append(model_std)
        print(f"    Final: loss={result_std['final_loss']:.4f}, acc={result_std['final_acc']:.3f}")

        # Operator-guided training
        print("  Operator-Guided:")
        set_seed(seed)
        model_op = make_model().to(device)
        optimizer_op = torch.optim.Adam(model_op.parameters(), lr=lr)
        trainer_op = OperatorGuidedTrainerV2(model_op, optimizer_op, hidden_dim, device)

        result_op = train_and_evaluate(
            trainer_op, train_loader, val_loader,
            loss_fn, device, epochs, is_operator=True
        )
        op_results.append(result_op)
        op_models.append(model_op)
        print(f"    Final: loss={result_op['final_loss']:.4f}, acc={result_op['final_acc']:.3f}")

    # Analyze results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    std_losses = [r['final_loss'] for r in std_results]
    std_accs = [r['final_acc'] for r in std_results]
    op_losses = [r['final_loss'] for r in op_results]
    op_accs = [r['final_acc'] for r in op_results]

    print("\nStandard Adam:")
    print(f"  Loss: {np.mean(std_losses):.4f} ± {np.std(std_losses):.4f}")
    print(f"  Acc:  {np.mean(std_accs):.3f} ± {np.std(std_accs):.3f}")
    print(f"  Range: [{np.min(std_accs):.3f}, {np.max(std_accs):.3f}]")

    print("\nOperator-Guided:")
    print(f"  Loss: {np.mean(op_losses):.4f} ± {np.std(op_losses):.4f}")
    print(f"  Acc:  {np.mean(op_accs):.3f} ± {np.std(op_accs):.3f}")
    print(f"  Range: [{np.min(op_accs):.3f}, {np.max(op_accs):.3f}]")

    # Key metrics
    loss_var_reduction = (np.std(std_losses) - np.std(op_losses)) / np.std(std_losses) * 100 if np.std(std_losses) > 0 else 0
    acc_var_reduction = (np.std(std_accs) - np.std(op_accs)) / np.std(std_accs) * 100 if np.std(std_accs) > 0 else 0

    print("\n" + "-"*50)
    print("KEY METRICS:")
    print(f"  Loss variance reduction: {loss_var_reduction:.1f}%")
    print(f"  Accuracy variance reduction: {acc_var_reduction:.1f}%")

    # Compute representation similarity
    test_x = X_val[:100]
    std_sim = compute_model_similarity(std_models, test_x, device)
    op_sim = compute_model_similarity(op_models, test_x, device)

    print(f"\n  Representation similarity (higher = more consistent):")
    print(f"    Standard: {std_sim:.4f}")
    print(f"    Operator: {op_sim:.4f}")

    if op_sim > std_sim:
        print("    ✓ Operator guidance produces MORE consistent representations")
    else:
        print("    ✗ No improvement in representation consistency")

    print("-"*50)

    # Verdict
    print("\nVERDICT:")
    if acc_var_reduction > 10 or op_sim > std_sim * 1.1:
        print("  ✓ POC VALIDATED: Operator guidance shows seed-independent benefits")
    elif np.mean(op_accs) > np.mean(std_accs):
        print("  ~ PARTIAL: Better mean accuracy but not clearly more consistent")
    else:
        print("  ✗ POC NOT VALIDATED: Need parameter tuning")

    # Generate plots
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy distribution
    ax = axes[0, 0]
    ax.boxplot([std_accs, op_accs], labels=['Standard', 'Operator'])
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'Accuracy Distribution Across {len(seeds)} Seeds')
    ax.axhline(y=np.mean(std_accs), color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(op_accs), color='orange', linestyle='--', alpha=0.5)

    # Plot 2: Training curves
    ax = axes[0, 1]
    for r in std_results:
        ax.plot(r['train_losses'], alpha=0.3, color='blue')
    for r in op_results:
        ax.plot(r['train_losses'], alpha=0.3, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Curves (Blue=Standard, Orange=Operator)')

    # Plot 3: Spectral gaps
    ax = axes[1, 0]
    for r in op_results:
        if 'spectral_gap' in r['history']:
            gaps = r['history']['spectral_gap']
            # Smooth for visualization
            if len(gaps) > 10:
                window = 10
                smoothed = [np.mean(gaps[max(0,i-window):i+1]) for i in range(len(gaps))]
                ax.plot(smoothed, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Spectral Gap')
    ax.set_title('Spectral Gap Evolution')
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Phase threshold')

    # Plot 4: Final accuracy scatter
    ax = axes[1, 1]
    x = range(len(seeds))
    ax.scatter(x, std_accs, label='Standard', marker='o', s=100)
    ax.scatter(x, op_accs, label='Operator', marker='^', s=100)
    ax.set_xlabel('Seed Index')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Per-Seed Final Accuracy')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/poc_v2_results.png', dpi=150)
    plt.close()

    # Save results
    results = {
        'dataset': dataset_name,
        'seeds': seeds,
        'standard': {
            'mean_acc': float(np.mean(std_accs)),
            'std_acc': float(np.std(std_accs)),
            'accs': [float(a) for a in std_accs]
        },
        'operator': {
            'mean_acc': float(np.mean(op_accs)),
            'std_acc': float(np.std(op_accs)),
            'accs': [float(a) for a in op_accs]
        },
        'acc_variance_reduction_pct': float(acc_var_reduction),
        'representation_similarity': {
            'standard': float(std_sim),
            'operator': float(op_sim)
        }
    }

    with open(f'{output_dir}/poc_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("="*70)

    return results


if __name__ == "__main__":
    run_experiment('spiral')
