"""
Representation Buffer: Core data structure for operator-guided training.

Tracks representation statistics across training steps to identify
"successful escape directions" - representation changes that consistently
lead to loss improvements across batches.

The key insight: we're not tracking gradients or weights, but the actual
movement in representation space that correlates with improvement.
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch.nn.functional as F


@dataclass
class RepresentationSnapshot:
    """Single snapshot of representations and associated metrics."""
    step: int
    representations: torch.Tensor  # (batch_size, hidden_dim) or aggregated
    loss: float
    batch_indices: Optional[torch.Tensor] = None


class RepresentationBuffer:
    """
    Circular buffer tracking representation dynamics over training.

    Core functionality:
    1. Store representation snapshots with associated losses
    2. Compute representation deltas between consecutive steps
    3. Identify which deltas consistently correlate with loss improvement
    4. Aggregate successful escape directions into a momentum field
    """

    def __init__(
        self,
        hidden_dim: int,
        buffer_size: int = 1000,
        aggregation: str = 'mean',  # 'mean', 'layerwise', or 'pca'
        device: str = 'cpu'
    ):
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.aggregation = aggregation
        self.device = device

        # Circular buffer of snapshots
        self.snapshots: deque = deque(maxlen=buffer_size)

        # Running statistics for normalization
        self.rep_mean = torch.zeros(hidden_dim, device=device)
        self.rep_var = torch.ones(hidden_dim, device=device)
        self.n_samples = 0

        # Successful escape directions (exponential moving average)
        self.escape_direction = torch.zeros(hidden_dim, device=device)
        self.escape_magnitude = 0.0

    def store(
        self,
        representations: torch.Tensor,
        loss: float,
        step: int,
        batch_indices: Optional[torch.Tensor] = None
    ):
        """
        Store a representation snapshot.

        Args:
            representations: Hidden states, shape (batch_size, hidden_dim)
            loss: Loss value for this batch
            step: Training step number
            batch_indices: Optional indices for tracking specific samples
        """
        # Aggregate representations (we track the mean representation per batch)
        if self.aggregation == 'mean':
            aggregated = representations.mean(dim=0).detach()
        elif self.aggregation == 'layerwise':
            # If representations are (batch, layers, dim), flatten
            aggregated = representations.mean(dim=0).view(-1).detach()
        else:
            aggregated = representations.mean(dim=0).detach()

        # Update running statistics (Welford's online algorithm)
        self.n_samples += 1
        delta = aggregated - self.rep_mean
        self.rep_mean = self.rep_mean + delta / self.n_samples
        delta2 = aggregated - self.rep_mean
        self.rep_var = self.rep_var + (delta * delta2 - self.rep_var) / self.n_samples

        # Store snapshot
        snapshot = RepresentationSnapshot(
            step=step,
            representations=aggregated.to(self.device),
            loss=loss,
            batch_indices=batch_indices
        )
        self.snapshots.append(snapshot)

    def compute_representation_deltas(
        self,
        window: int = 10
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Compute representation deltas and their associated loss changes.

        Returns list of (delta_representation, delta_loss) tuples.
        Negative delta_loss means improvement.
        """
        if len(self.snapshots) < 2:
            return []

        deltas = []
        snapshots = list(self.snapshots)

        for i in range(max(0, len(snapshots) - window), len(snapshots) - 1):
            curr = snapshots[i]
            next_snap = snapshots[i + 1]

            delta_rep = next_snap.representations - curr.representations
            delta_loss = next_snap.loss - curr.loss

            deltas.append((delta_rep, delta_loss))

        return deltas

    def successful_escape_directions(
        self,
        improvement_threshold: float = 0.0,
        window: int = 50
    ) -> torch.Tensor:
        """
        Extract directions in representation space that consistently
        correlate with loss improvement.

        This is the key insight: we're learning which directions in
        representation space lead to better solutions, not just following
        gradients.

        Args:
            improvement_threshold: Minimum loss decrease to count as improvement
            window: Number of recent steps to consider

        Returns:
            Aggregated escape direction (hidden_dim,)
        """
        deltas = self.compute_representation_deltas(window)

        if not deltas:
            return torch.zeros(self.hidden_dim, device=self.device)

        # Filter for improving steps
        successful_deltas = []
        weights = []

        for delta_rep, delta_loss in deltas:
            if delta_loss < -improvement_threshold:
                # This direction led to improvement
                successful_deltas.append(delta_rep)
                # Weight by magnitude of improvement
                weights.append(-delta_loss)

        if not successful_deltas:
            return torch.zeros(self.hidden_dim, device=self.device)

        # Stack and compute weighted average
        deltas_tensor = torch.stack(successful_deltas)
        weights_tensor = torch.tensor(weights, device=self.device)
        weights_tensor = weights_tensor / (weights_tensor.sum() + 1e-8)

        # Weighted average of successful directions
        escape_dir = (deltas_tensor * weights_tensor.unsqueeze(1)).sum(dim=0)

        # Normalize
        norm = torch.norm(escape_dir)
        if norm > 1e-8:
            escape_dir = escape_dir / norm

        return escape_dir

    def update_escape_momentum(
        self,
        ema_decay: float = 0.9,
        window: int = 50
    ):
        """
        Update the escape direction momentum field using EMA.

        This creates a smooth, stable estimate of successful directions
        that can be injected into the optimizer.
        """
        current_escape = self.successful_escape_directions(window=window)

        # EMA update
        self.escape_direction = (
            ema_decay * self.escape_direction +
            (1 - ema_decay) * current_escape
        )

        # Track magnitude (how confident we are in the direction)
        self.escape_magnitude = torch.norm(self.escape_direction).item()

    def get_escape_drift(self, scale: float = 1.0) -> torch.Tensor:
        """
        Get the current escape drift to inject into optimizer.

        Args:
            scale: Reynolds number analog - controls strength of drift

        Returns:
            Drift vector in representation space
        """
        return scale * self.escape_direction

    def compute_representation_covariance(
        self,
        window: int = 100
    ) -> torch.Tensor:
        """
        Compute covariance matrix of recent representations.

        This is used for spectral gap estimation - the eigenvalue
        structure tells us about mode separation.
        """
        if len(self.snapshots) < window:
            return torch.eye(self.hidden_dim, device=self.device)

        snapshots = list(self.snapshots)[-window:]
        reps = torch.stack([s.representations for s in snapshots])

        # Center
        reps_centered = reps - reps.mean(dim=0, keepdim=True)

        # Covariance
        cov = (reps_centered.T @ reps_centered) / (len(reps) - 1)

        return cov

    def estimate_spectral_gap(self, window: int = 100) -> float:
        """
        Estimate spectral gap of representation covariance.

        The spectral gap (λ₁ - λ₂) / λ₁ indicates how well-separated
        the dominant mode is. A large gap suggests convergence to
        a stable representation.

        This is the key observable for detecting phase transitions.
        """
        cov = self.compute_representation_covariance(window)

        # Compute eigenvalues (top-k for efficiency)
        try:
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.real
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]

            if len(eigenvalues) >= 2 and eigenvalues[0] > 1e-8:
                # Relative spectral gap
                gap = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
                return gap.item()
            else:
                return 0.0
        except:
            return 0.0

    def get_statistics(self) -> dict:
        """Get current buffer statistics for monitoring."""
        return {
            'n_snapshots': len(self.snapshots),
            'n_samples_total': self.n_samples,
            'escape_magnitude': self.escape_magnitude,
            'spectral_gap': self.estimate_spectral_gap(),
            'rep_mean_norm': torch.norm(self.rep_mean).item(),
            'rep_var_mean': self.rep_var.mean().item()
        }


class LayerwiseRepresentationBuffer:
    """
    Track representations across multiple layers.

    For transformers, this tracks each layer's hidden states separately,
    allowing us to identify which layers are contributing to successful escapes.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        buffer_size: int = 1000,
        device: str = 'cpu'
    ):
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.device = device

        # Separate buffer per layer
        self.layer_buffers = [
            RepresentationBuffer(dim, buffer_size, device=device)
            for dim in hidden_dims
        ]

        # Cross-layer escape direction (concatenated)
        self.total_dim = sum(hidden_dims)
        self.global_escape = torch.zeros(self.total_dim, device=device)

    def store(
        self,
        layer_representations: List[torch.Tensor],
        loss: float,
        step: int
    ):
        """Store representations from all layers."""
        for i, (buffer, reps) in enumerate(zip(self.layer_buffers, layer_representations)):
            buffer.store(reps, loss, step)

    def update_escape_momentum(self, ema_decay: float = 0.9, window: int = 50):
        """Update escape momentum for all layers."""
        for buffer in self.layer_buffers:
            buffer.update_escape_momentum(ema_decay, window)

        # Concatenate layer escape directions
        self.global_escape = torch.cat([
            buffer.escape_direction for buffer in self.layer_buffers
        ])

    def get_layer_escape_drifts(self, scale: float = 1.0) -> List[torch.Tensor]:
        """Get escape drift per layer."""
        return [buffer.get_escape_drift(scale) for buffer in self.layer_buffers]

    def estimate_global_spectral_gap(self) -> float:
        """Estimate spectral gap across all layers (average)."""
        gaps = [buffer.estimate_spectral_gap() for buffer in self.layer_buffers]
        return np.mean(gaps) if gaps else 0.0

    def get_layer_statistics(self) -> List[dict]:
        """Get statistics per layer."""
        return [buffer.get_statistics() for buffer in self.layer_buffers]
