"""
POC Final: Theoretically Grounded Operator-Guided Training

The core insight from the theory:
- Standard SGD follows LOCAL gradient information
- We augment with GLOBAL information from successful escapes
- The key is that "u" (velocity field) comes from REPRESENTATION DYNAMICS,
  not parameter dynamics

This implementation:
1. Tracks representation trajectories (not just snapshots)
2. Identifies "escape events" - moments when loss dropped significantly
3. Builds a momentum field from the representation directions that led to escapes
4. Injects this as a bias toward productive directions

The test: does this produce more CONSISTENT results across seeds?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import os


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_challenging_dataset(n_samples: int = 4000, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a challenging classification problem with multiple local minima.

    This is a "two moons" variant with extra noise dimensions.
    """
    np.random.seed(seed)

    n_per_class = n_samples // 2

    # Two moons
    theta = np.linspace(0, np.pi, n_per_class)
    x1 = np.cos(theta) + np.random.randn(n_per_class) * 0.2
    y1 = np.sin(theta) + np.random.randn(n_per_class) * 0.2

    theta = np.linspace(0, np.pi, n_per_class)
    x2 = 1 - np.cos(theta) + np.random.randn(n_per_class) * 0.2
    y2 = 0.5 - np.sin(theta) + np.random.randn(n_per_class) * 0.2

    X1 = np.stack([x1, y1], axis=1)
    X2 = np.stack([x2, y2], axis=1)

    # Add noise dimensions that CREATE local minima
    noise_dims = 18
    X1_noise = np.random.randn(n_per_class, noise_dims) * 0.5
    X2_noise = np.random.randn(n_per_class, noise_dims) * 0.5

    # Add correlations that can fool the network
    X1_noise[:, 0] = X1[:, 0] * 0.3 + np.random.randn(n_per_class) * 0.3
    X2_noise[:, 0] = X2[:, 0] * 0.3 + np.random.randn(n_per_class) * 0.3

    X1 = np.hstack([X1, X1_noise])
    X2 = np.hstack([X2, X2_noise])

    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)

    perm = np.random.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


class MLP(nn.Module):
    """Simple MLP with accessible hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3):
        super().__init__()

        self.hidden_dim = hidden_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]

        for _ in range(depth - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self._hidden = None

    def forward(self, x):
        self._hidden = self.features(x)
        return self.classifier(self._hidden)

    def get_hidden(self):
        return self._hidden


@dataclass
class EscapeEvent:
    """Records a successful escape from a local minimum."""
    step: int
    rep_before: torch.Tensor  # Representation before escape
    rep_after: torch.Tensor   # Representation after escape
    loss_drop: float          # How much loss decreased
    direction: torch.Tensor   # Normalized direction of escape


class EscapeTracker:
    """
    Tracks escape events and builds momentum field.

    An "escape" is defined as:
    - Loss dropping by more than threshold
    - Over a window of steps

    The escape direction is the average representation change during the escape.
    """

    def __init__(self, hidden_dim: int, device: str, window: int = 5, threshold: float = 0.01):
        self.hidden_dim = hidden_dim
        self.device = device
        self.window = window
        self.threshold = threshold

        # History
        self.rep_history: deque = deque(maxlen=window * 2)
        self.loss_history: deque = deque(maxlen=window * 2)

        # Escape events
        self.escape_events: List[EscapeEvent] = []
        self.max_events = 100

        # Momentum field (EMA of successful directions)
        self.momentum = torch.zeros(hidden_dim, device=device)
        self.momentum_strength = 0.0

    def record(self, representations: torch.Tensor, loss: float, step: int):
        """Record current state and check for escapes."""
        mean_rep = representations.mean(dim=0).detach()

        self.rep_history.append(mean_rep.cpu())
        self.loss_history.append(loss)

        # Check for escape after we have enough history
        if len(self.loss_history) >= self.window * 2:
            self._check_for_escape(step)

    def _check_for_escape(self, step: int):
        """Check if an escape event occurred."""
        losses = list(self.loss_history)
        reps = list(self.rep_history)

        # Compare average loss in first half vs second half
        first_half_loss = np.mean(losses[:self.window])
        second_half_loss = np.mean(losses[self.window:])

        loss_drop = first_half_loss - second_half_loss

        # Significant drop = escape
        if loss_drop > self.threshold:
            # Compute direction
            rep_before = torch.stack(reps[:self.window]).mean(dim=0)
            rep_after = torch.stack(reps[self.window:]).mean(dim=0)

            direction = rep_after - rep_before
            norm = torch.norm(direction)

            if norm > 1e-6:
                direction = direction / norm

                event = EscapeEvent(
                    step=step,
                    rep_before=rep_before,
                    rep_after=rep_after,
                    loss_drop=loss_drop,
                    direction=direction
                )

                self.escape_events.append(event)
                if len(self.escape_events) > self.max_events:
                    self.escape_events.pop(0)

                # Update momentum with this direction
                weight = min(1.0, loss_drop * 10)  # Weight by improvement magnitude
                self.momentum = 0.9 * self.momentum + 0.1 * weight * direction.to(self.device)
                self.momentum_strength = torch.norm(self.momentum).item()

    def get_momentum(self) -> torch.Tensor:
        """Get current momentum field (normalized)."""
        if self.momentum_strength > 1e-6:
            return self.momentum / self.momentum_strength
        return torch.zeros(self.hidden_dim, device=self.device)

    def compute_spectral_gap(self) -> float:
        """Compute spectral gap from representation history."""
        if len(self.rep_history) < 20:
            return 0.0

        reps = torch.stack(list(self.rep_history)[-50:])
        reps = reps - reps.mean(dim=0, keepdim=True)

        cov = (reps.T @ reps) / (len(reps) - 1)
        cov = cov + torch.eye(cov.shape[0]) * 1e-6

        try:
            eigs = torch.linalg.eigvalsh(cov)
            eigs = torch.sort(eigs, descending=True)[0]
            if eigs[0] > 1e-8:
                return float((eigs[0] - eigs[1]) / eigs[0])
        except:
            pass

        return 0.0


class OperatorGuidedTrainer:
    """
    Trainer that uses escape dynamics to guide optimization.

    The key mechanism:
    1. Track representation history
    2. Detect "escape events" (significant loss drops)
    3. Record the representation direction during escapes
    4. Build momentum toward productive directions
    5. Apply this as additional gradient during training
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        hidden_dim: int,
        device: str,
        momentum_scale: float = 0.5  # How strongly to apply momentum
    ):
        self.model = model.to(device)
        self.device = device
        self.hidden_dim = hidden_dim
        self.momentum_scale = momentum_scale

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.tracker = EscapeTracker(hidden_dim, device)

        self.step = 0
        self.phase = 'exploration'

        self.history = {
            'loss': [],
            'spectral_gap': [],
            'momentum_strength': [],
            'n_escapes': [],
            'phase': []
        }

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, loss_fn) -> Dict:
        self.model.train()
        self.optimizer.zero_grad()
        self.step += 1

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Forward
        logits = self.model(batch_x)
        hidden = self.model.get_hidden()

        # Loss
        loss = loss_fn(logits, batch_y)

        # Record for escape tracking
        self.tracker.record(hidden.detach(), loss.item(), self.step)

        # Backward
        loss.backward()

        # Apply momentum-based drift to gradients
        if self.step > 100 and self.tracker.momentum_strength > 0.01:
            self._apply_momentum_drift()

        # Update phase based on spectral gap
        spectral_gap = 0.0
        if self.step % 20 == 0:
            spectral_gap = self.tracker.compute_spectral_gap()
            self._update_phase(spectral_gap)

        self.optimizer.step()

        # Record history
        self.history['loss'].append(loss.item())
        self.history['spectral_gap'].append(spectral_gap)
        self.history['momentum_strength'].append(self.tracker.momentum_strength)
        self.history['n_escapes'].append(len(self.tracker.escape_events))
        self.history['phase'].append(self.phase)

        return {
            'loss': loss.item(),
            'spectral_gap': spectral_gap,
            'momentum': self.tracker.momentum_strength,
            'n_escapes': len(self.tracker.escape_events),
            'phase': self.phase
        }

    def _apply_momentum_drift(self):
        """Apply escape momentum as gradient modification."""
        momentum = self.tracker.get_momentum()

        # Scale by phase (less drift in convergence phase)
        phase_scale = {'exploration': 1.0, 'advection': 0.8, 'convergence': 0.3}
        scale = self.momentum_scale * phase_scale.get(self.phase, 0.5)

        # Apply to classifier layer (most direct effect)
        for name, param in self.model.named_parameters():
            if 'classifier' in name and 'weight' in name and param.grad is not None:
                if param.grad.shape[1] == self.hidden_dim:
                    # Weight is (out_dim, hidden_dim)
                    drift = momentum.unsqueeze(0).expand(param.grad.shape[0], -1)
                    param.grad.add_(drift * scale * 0.1)

    def _update_phase(self, spectral_gap: float):
        """Update training phase."""
        if spectral_gap > 0.4:
            self.phase = 'convergence'
        elif spectral_gap > 0.15:
            self.phase = 'advection'
        else:
            self.phase = 'exploration'


class StandardTrainer:
    """Standard trainer for comparison."""

    def __init__(self, model: nn.Module, lr: float, device: str):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.history = {'loss': []}

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor, loss_fn) -> Dict:
        self.model.train()
        self.optimizer.zero_grad()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        logits = self.model(batch_x)
        loss = loss_fn(logits, batch_y)

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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

    return total_loss / len(loader), correct / total


def get_representation_similarity(models: List[nn.Module], test_x: torch.Tensor, device: str) -> float:
    """Compute average representation similarity across models (CKA)."""
    reps = []
    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(test_x.to(device))
            h = model.get_hidden().cpu().numpy()
            h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
            reps.append(h)

    scores = []
    for i in range(len(reps)):
        for j in range(i+1, len(reps)):
            K = reps[i] @ reps[i].T
            L = reps[j] @ reps[j].T
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            Kc, Lc = H @ K @ H, H @ L @ H
            hsic = np.trace(Kc @ Lc)
            vk = np.sqrt(np.trace(Kc @ Kc))
            vl = np.sqrt(np.trace(Lc @ Lc))
            if vk > 0 and vl > 0:
                scores.append(hsic / (vk * vl))

    return float(np.mean(scores)) if scores else 0


def run_experiment():
    print("="*70)
    print("OPERATOR-GUIDED TRAINING: FINAL POC")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Config
    input_dim = 20
    hidden_dim = 64
    n_classes = 2
    epochs = 200
    batch_size = 64
    lr = 5e-4
    seeds = [42, 123, 456, 789, 1000, 2023, 3141, 9999, 5555, 7777]

    print(f"\nConfiguration:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  epochs: {epochs}")
    print(f"  lr: {lr}")
    print(f"  n_seeds: {len(seeds)}")

    # Data
    print("\nCreating dataset...")
    X, y = create_challenging_dataset(n_samples=4000, seed=42)
    n_train = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(X[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[n_train:], y[n_train:]), batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()

    # Run
    print("\n" + "="*70)
    std_results, op_results = [], []
    std_models, op_models = [], []

    for seed in seeds:
        print(f"\nSeed {seed}:")

        # Standard
        set_seed(seed)
        model_std = MLP(input_dim, hidden_dim, n_classes, depth=4)
        trainer_std = StandardTrainer(model_std, lr, device)

        for _ in range(epochs):
            for x, y in train_loader:
                trainer_std.train_step(x, y, loss_fn)

        _, acc_std = evaluate(model_std, val_loader, loss_fn, device)
        std_results.append(acc_std)
        std_models.append(model_std)
        print(f"  Standard: {acc_std:.3f}")

        # Operator-guided
        set_seed(seed)
        model_op = MLP(input_dim, hidden_dim, n_classes, depth=4)
        trainer_op = OperatorGuidedTrainer(model_op, lr, hidden_dim, device, momentum_scale=0.5)

        for _ in range(epochs):
            for x, y in train_loader:
                trainer_op.train_step(x, y, loss_fn)

        _, acc_op = evaluate(model_op, val_loader, loss_fn, device)
        op_results.append(acc_op)
        op_models.append(model_op)

        n_esc = len(trainer_op.tracker.escape_events)
        final_gap = trainer_op.tracker.compute_spectral_gap()
        print(f"  Operator: {acc_op:.3f} (escapes: {n_esc}, gap: {final_gap:.3f})")

    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    std_mean, std_std = np.mean(std_results), np.std(std_results)
    op_mean, op_std = np.mean(op_results), np.std(op_results)

    print(f"\nStandard Adam:")
    print(f"  {std_mean:.3f} ± {std_std:.3f}  [{min(std_results):.3f}, {max(std_results):.3f}]")

    print(f"\nOperator-Guided:")
    print(f"  {op_mean:.3f} ± {op_std:.3f}  [{min(op_results):.3f}, {max(op_results):.3f}]")

    # Metrics
    var_reduction = (std_std - op_std) / std_std * 100 if std_std > 0 else 0
    mean_improvement = (op_mean - std_mean) / std_mean * 100

    test_x = X[n_train:n_train+200]
    std_cka = get_representation_similarity(std_models, test_x, device)
    op_cka = get_representation_similarity(op_models, test_x, device)

    print(f"\nVariance reduction: {var_reduction:.1f}%")
    print(f"Mean improvement: {mean_improvement:.1f}%")
    print(f"CKA (Standard): {std_cka:.4f}")
    print(f"CKA (Operator): {op_cka:.4f}")

    # Verdict
    print("\n" + "-"*50)
    wins = 0
    if op_mean > std_mean:
        print("✓ Higher mean accuracy")
        wins += 1
    else:
        print("✗ Lower mean accuracy")

    if op_std < std_std:
        print("✓ Lower variance (more consistent)")
        wins += 1
    else:
        print("✗ Higher variance")

    if op_cka > std_cka:
        print("✓ Higher CKA (more similar representations)")
        wins += 1
    else:
        print("✗ Lower CKA")

    print(f"\nScore: {wins}/3")
    if wins >= 2:
        print("\n*** POC VALIDATED ***")
        print("Operator guidance improves optimization consistency")
    else:
        print("\n*** NEEDS MORE WORK ***")

    print("-"*50)

    # Plot
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.boxplot([std_results, op_results], tick_labels=['Standard', 'Operator'])
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy Distribution ({len(seeds)} seeds)')

    ax = axes[0, 1]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, std_results, w, label='Standard', alpha=0.8)
    ax.bar(x + w/2, op_results, w, label='Operator', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Seed Results')
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(std_results, op_results)
    ax.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Standard Acc')
    ax.set_ylabel('Operator Acc')
    ax.set_title('Pairwise Comparison')

    ax = axes[1, 1]
    diffs = np.array(op_results) - np.array(std_results)
    colors = ['g' if d > 0 else 'r' for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Seed Index')
    ax.set_ylabel('Operator - Standard')
    ax.set_title('Improvement per Seed')

    plt.tight_layout()
    plt.savefig('results/poc_final.png', dpi=150)
    plt.close()

    print("\nSaved: results/poc_final.png")
    print("="*70)


if __name__ == "__main__":
    run_experiment()
