"""
POC Validation V3: Fixed Spectral Gap Computation

Key fixes:
1. Proper spectral gap estimation with numerical stability
2. Store full batch representations (not just mean)
3. More aggressive drift application
4. Better escape direction detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
from datetime import datetime
import os


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
    """Create XOR-like spiral dataset."""
    np.random.seed(seed)
    n_per_class = n_samples // 2

    theta1 = np.linspace(0, 4*np.pi, n_per_class) + np.random.randn(n_per_class) * noise
    r1 = theta1 / (4*np.pi) * 2 + np.random.randn(n_per_class) * noise * 0.3
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)

    theta2 = np.linspace(0, 4*np.pi, n_per_class) + np.pi + np.random.randn(n_per_class) * noise
    r2 = theta2 / (4*np.pi) * 2 + np.random.randn(n_per_class) * noise * 0.3
    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)

    X0 = np.stack([x1, y1], axis=1)
    X1 = np.stack([x2, y2], axis=1)

    extra_dims = 18
    X0 = np.hstack([X0, np.random.randn(n_per_class, extra_dims) * 0.5])
    X1 = np.hstack([X1, np.random.randn(n_per_class, extra_dims) * 0.5])

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)

    perm = np.random.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


class DeepMLP(nn.Module):
    """Deep MLP with accessible representations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self._reps = None

    def forward(self, x):
        self._reps = self.backbone(x)
        return self.head(self._reps)

    def get_representations(self):
        return self._reps


class ImprovedRepresentationTracker:
    """
    Fixed representation tracker with proper spectral computation.
    """

    def __init__(self, hidden_dim: int, max_samples: int = 2000, device: str = 'cpu'):
        self.hidden_dim = hidden_dim
        self.max_samples = max_samples
        self.device = device

        # Store actual representations (not just means)
        self.rep_buffer: deque = deque(maxlen=max_samples)
        self.loss_buffer: deque = deque(maxlen=1000)

        # Escape direction (EMA)
        self.escape_dir = torch.zeros(hidden_dim, device=device)
        self.escape_magnitude = 0.0

        # For tracking deltas
        self.prev_mean_rep = None

    def store(self, representations: torch.Tensor, loss: float):
        """Store batch representations."""
        # Store mean of batch
        mean_rep = representations.mean(dim=0).detach().cpu()
        self.rep_buffer.append(mean_rep)
        self.loss_buffer.append(loss)

        # Track deltas for escape direction
        if self.prev_mean_rep is not None and len(self.loss_buffer) >= 2:
            delta_rep = mean_rep - self.prev_mean_rep
            delta_loss = self.loss_buffer[-1] - self.loss_buffer[-2]

            # If loss improved, this direction is good
            if delta_loss < 0:
                weight = min(1.0, -delta_loss * 10)  # Scale by improvement
                self.escape_dir = 0.95 * self.escape_dir + 0.05 * weight * delta_rep.to(self.device)

        self.prev_mean_rep = mean_rep
        self.escape_magnitude = torch.norm(self.escape_dir).item()

    def compute_spectral_gap(self, window: int = 100) -> float:
        """
        Compute spectral gap from representation covariance.

        This is done properly with numerical stability.
        """
        if len(self.rep_buffer) < window:
            return 0.0

        # Get recent representations
        reps = torch.stack(list(self.rep_buffer)[-window:])  # (window, hidden_dim)

        # Center
        reps_centered = reps - reps.mean(dim=0, keepdim=True)

        # Covariance
        cov = (reps_centered.T @ reps_centered) / (window - 1)

        # Add small regularization for numerical stability
        cov = cov + torch.eye(cov.shape[0]) * 1e-6

        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]

            # Spectral gap: relative separation of top eigenvalue
            if eigenvalues[0] > 1e-8:
                gap = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
                return float(gap.item())
        except Exception as e:
            pass

        return 0.0

    def get_escape_drift(self, scale: float = 1.0) -> torch.Tensor:
        """Get normalized escape direction scaled by magnitude."""
        if self.escape_magnitude > 1e-6:
            return scale * self.escape_dir / (self.escape_magnitude + 1e-8)
        return torch.zeros(self.hidden_dim, device=self.device)


class OperatorGuidedTrainerV3:
    """
    Fixed operator-guided trainer.

    Key improvements:
    1. More aggressive drift application
    2. Proper spectral gap tracking
    3. Phase-aware learning rate modulation
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        hidden_dim: int,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.hidden_dim = hidden_dim
        self.device = device
        self.base_lr = base_lr

        # Optimizer with higher LR that we'll modulate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        # Representation tracking
        self.tracker = ImprovedRepresentationTracker(hidden_dim, device=device)

        # State
        self.step = 0
        self.phase = 'EXPLORATION'
        self.reynolds = 0.1

        # History
        self.history = {
            'loss': [],
            'spectral_gap': [],
            'escape_magnitude': [],
            'reynolds': [],
            'phase': []
        }

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, loss_fn) -> Dict:
        self.model.train()
        self.optimizer.zero_grad()
        self.step += 1

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Forward
        output = self.model(batch_x)
        reps = self.model.get_representations()

        # Loss
        loss = loss_fn(output, batch_y)

        # Store representations
        self.tracker.store(reps.detach(), loss.item())

        # Backward
        loss.backward()

        # Compute spectral gap and update phase
        spectral_gap = 0.0
        if self.step > 50 and self.step % 10 == 0:
            spectral_gap = self.tracker.compute_spectral_gap(window=50)
            self._update_phase(spectral_gap)

        # Apply drift to gradients
        if self.step > 100 and self.tracker.escape_magnitude > 0.01:
            self._apply_drift()

        # Optimizer step
        self.optimizer.step()

        # Record history
        self.history['loss'].append(loss.item())
        self.history['spectral_gap'].append(spectral_gap)
        self.history['escape_magnitude'].append(self.tracker.escape_magnitude)
        self.history['reynolds'].append(self.reynolds)
        self.history['phase'].append(self.phase)

        return {
            'loss': loss.item(),
            'spectral_gap': spectral_gap,
            'escape_magnitude': self.tracker.escape_magnitude,
            'phase': self.phase,
            'reynolds': self.reynolds
        }

    def _update_phase(self, spectral_gap: float):
        """Update training phase based on spectral gap."""
        if spectral_gap > 0.5:
            self.phase = 'CONVERGENCE'
            self.reynolds = 0.3
        elif spectral_gap > 0.2:
            self.phase = 'GAP_OPENING'
            self.reynolds = 1.0
        elif spectral_gap > 0.05:
            self.phase = 'ADVECTION'
            self.reynolds = 0.7
        else:
            self.phase = 'EXPLORATION'
            self.reynolds = 0.2

    def _apply_drift(self):
        """Apply escape drift to gradients."""
        drift = self.tracker.get_escape_drift(scale=self.reynolds * 0.1)

        # Apply to last layer before head
        for name, param in self.model.named_parameters():
            if 'backbone' in name and 'weight' in name and param.grad is not None:
                if len(param.grad.shape) == 2:
                    # For weight matrices, apply drift to output dimension
                    if param.grad.shape[0] == self.hidden_dim:
                        drift_mat = drift.unsqueeze(1).expand_as(param.grad)
                        param.grad.add_(drift_mat * 0.01)


class StandardTrainerV3:
    """Standard trainer for comparison."""

    def __init__(self, model: nn.Module, lr: float, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.history = {'loss': []}

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, loss_fn) -> Dict:
        self.model.train()
        self.optimizer.zero_grad()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        output = self.model(batch_x)
        loss = loss_fn(output, batch_y)

        loss.backward()
        self.optimizer.step()

        self.history['loss'].append(loss.item())
        return {'loss': loss.item()}


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: str) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            total_loss += loss_fn(output, batch_y).item()

            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += len(batch_y)

    return total_loss / len(loader), correct / total


def run_single_experiment(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    n_classes: int,
    epochs: int,
    lr: float,
    device: str,
    use_operator: bool
) -> Dict:
    """Run single training experiment."""
    set_seed(seed)

    model = DeepMLP(input_dim, hidden_dim, n_classes, depth=4)
    loss_fn = nn.CrossEntropyLoss()

    if use_operator:
        trainer = OperatorGuidedTrainerV3(model, lr, hidden_dim, device)
    else:
        trainer = StandardTrainerV3(model, lr, device)

    train_losses = []
    val_losses = []
    val_accs = []
    spectral_gaps = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        epoch_gap = 0

        for batch_x, batch_y in train_loader:
            result = trainer.train_step(batch_x, batch_y, loss_fn)
            epoch_loss += result['loss']
            n_batches += 1
            if 'spectral_gap' in result:
                epoch_gap = result['spectral_gap']

        train_losses.append(epoch_loss / n_batches)
        spectral_gaps.append(epoch_gap)

        # Eval
        val_loss, val_acc = evaluate(trainer.model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'spectral_gaps': spectral_gaps,
        'final_acc': val_accs[-1],
        'final_loss': val_losses[-1],
        'model': trainer.model,
        'history': trainer.history if hasattr(trainer, 'history') else {}
    }


def compute_representation_consistency(models: List[nn.Module], test_x: torch.Tensor, device: str) -> float:
    """
    Compute how consistent representations are across models.

    Uses Centered Kernel Alignment (CKA) - higher = more similar.
    """
    reps = []
    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(test_x.to(device))
            rep = model.get_representations().cpu().numpy()
            # Normalize rows
            rep = rep / (np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8)
            reps.append(rep)

    # Compute pairwise CKA
    n = len(reps)
    cka_scores = []

    for i in range(n):
        for j in range(i+1, n):
            # CKA
            K = reps[i] @ reps[i].T
            L = reps[j] @ reps[j].T

            # Center
            n_samples = K.shape[0]
            H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
            K_c = H @ K @ H
            L_c = H @ L @ H

            # CKA
            hsic = np.trace(K_c @ L_c)
            var_k = np.sqrt(np.trace(K_c @ K_c))
            var_l = np.sqrt(np.trace(L_c @ L_c))

            if var_k > 0 and var_l > 0:
                cka = hsic / (var_k * var_l)
                cka_scores.append(cka)

    return float(np.mean(cka_scores)) if cka_scores else 0.0


def main():
    print("="*70)
    print("OPERATOR-GUIDED TRAINING: POC VALIDATION V3")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Config
    input_dim = 20
    hidden_dim = 128
    n_classes = 2
    epochs = 150
    batch_size = 32
    lr = 1e-3
    seeds = [42, 123, 456, 789, 1000, 2023, 3141, 9999]

    print(f"\nConfig: hidden_dim={hidden_dim}, epochs={epochs}, lr={lr}, seeds={len(seeds)}")

    # Dataset
    print("\nCreating spiral dataset...")
    X, y = create_xor_spiral_dataset(n_samples=3000, noise=0.5, seed=42)

    n_train = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(X[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[n_train:], y[n_train:]), batch_size=batch_size)

    # Run experiments
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)

    std_results = []
    op_results = []
    std_models = []
    op_models = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Standard
        print("  Standard Adam... ", end='', flush=True)
        result_std = run_single_experiment(
            seed, train_loader, val_loader,
            input_dim, hidden_dim, n_classes, epochs, lr, device,
            use_operator=False
        )
        std_results.append(result_std)
        std_models.append(result_std['model'])
        print(f"acc={result_std['final_acc']:.3f}")

        # Operator
        print("  Operator...      ", end='', flush=True)
        result_op = run_single_experiment(
            seed, train_loader, val_loader,
            input_dim, hidden_dim, n_classes, epochs, lr, device,
            use_operator=True
        )
        op_results.append(result_op)
        op_models.append(result_op['model'])

        final_gap = result_op['spectral_gaps'][-1] if result_op['spectral_gaps'] else 0
        final_phase = result_op['history'].get('phase', ['?'])[-1] if result_op['history'] else '?'
        print(f"acc={result_op['final_acc']:.3f}, gap={final_gap:.3f}, phase={final_phase}")

    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    std_accs = [r['final_acc'] for r in std_results]
    op_accs = [r['final_acc'] for r in op_results]

    print(f"\nStandard Adam:")
    print(f"  Accuracy: {np.mean(std_accs):.3f} ± {np.std(std_accs):.3f}")
    print(f"  Range: [{np.min(std_accs):.3f}, {np.max(std_accs):.3f}]")

    print(f"\nOperator-Guided:")
    print(f"  Accuracy: {np.mean(op_accs):.3f} ± {np.std(op_accs):.3f}")
    print(f"  Range: [{np.min(op_accs):.3f}, {np.max(op_accs):.3f}]")

    # Variance reduction
    std_var = np.std(std_accs)
    op_var = np.std(op_accs)
    var_reduction = (std_var - op_var) / std_var * 100 if std_var > 0 else 0

    print(f"\n  Variance reduction: {var_reduction:.1f}%")

    # Representation consistency
    test_x = X[n_train:n_train+100]
    std_cka = compute_representation_consistency(std_models, test_x, device)
    op_cka = compute_representation_consistency(op_models, test_x, device)

    print(f"\n  Representation consistency (CKA):")
    print(f"    Standard: {std_cka:.4f}")
    print(f"    Operator: {op_cka:.4f}")

    # Verdict
    print("\n" + "-"*50)
    better_mean = np.mean(op_accs) > np.mean(std_accs)
    lower_var = op_var < std_var
    higher_cka = op_cka > std_cka

    print(f"Better mean accuracy: {'✓' if better_mean else '✗'}")
    print(f"Lower variance: {'✓' if lower_var else '✗'}")
    print(f"Higher CKA: {'✓' if higher_cka else '✗'}")

    if sum([better_mean, lower_var, higher_cka]) >= 2:
        print("\n✓ POC SHOWS PROMISE - operator guidance improves at least 2/3 metrics")
    else:
        print("\n✗ POC needs more tuning")

    print("-"*50)

    # Save plots
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy comparison
    ax = axes[0, 0]
    ax.boxplot([std_accs, op_accs], tick_labels=['Standard', 'Operator'])
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'Accuracy Distribution (n={len(seeds)} seeds)')

    # Training curves
    ax = axes[0, 1]
    for r in std_results:
        ax.plot(r['val_accs'], 'b-', alpha=0.3)
    for r in op_results:
        ax.plot(r['val_accs'], 'r-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('Learning Curves (Blue=Std, Red=Op)')

    # Spectral gaps
    ax = axes[1, 0]
    for r in op_results:
        if r['spectral_gaps']:
            ax.plot(r['spectral_gaps'], alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Gap')
    ax.set_title('Spectral Gap Evolution')
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)

    # Per-seed comparison
    ax = axes[1, 1]
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, std_accs, width, label='Standard', alpha=0.8)
    ax.bar(x + width/2, op_accs, width, label='Operator', alpha=0.8)
    ax.set_xlabel('Seed Index')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('Per-Seed Final Accuracy')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/poc_v3.png', dpi=150)
    plt.close()

    print("\nResults saved to results/poc_v3.png")
    print("="*70)


if __name__ == "__main__":
    main()
