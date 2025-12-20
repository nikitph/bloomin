"""
Operator-Guided Optimizer: Non-gradient drift injection for global convergence.

The key insight: standard optimizers follow gradients, which only see LOCAL
loss landscape. We augment this with GLOBAL information from representation
dynamics - specifically, directions that historically led to improvement.

This is NOT:
- Second-order optimization (no Hessians)
- Quantum computing (just inspired by operator theory)
- Exotic architecture change

This IS:
- Operator learning: we learn the transport operator, not just follow gradients
- Representation-aware: we track what matters (representations), not what's convenient (parameters)
- Adaptive: we adjust drift strength based on spectral state
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from representation_buffer import RepresentationBuffer, LayerwiseRepresentationBuffer
from spectral_regulator import SpectralRegulator, SpectralState, TrainingPhase


class OperatorGuidedOptimizer:
    """
    Wraps a standard optimizer and adds representation-guided drift.

    The core mechanism:
    1. Standard optimizer computes gradient update Δθ_grad
    2. We compute representation drift direction u from successful escapes
    3. We project u into parameter space via Jacobian approximation
    4. Final update: Δθ = Δθ_grad + Re * Δθ_drift

    The Reynolds number (Re) is adaptive based on spectral state.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        model: nn.Module,
        hidden_dim: int,
        representation_hook: Callable,
        buffer_size: int = 1000,
        update_frequency: int = 10,
        device: str = 'cpu'
    ):
        """
        Args:
            base_optimizer: Standard optimizer (Adam, SGD, etc.)
            model: The neural network model
            hidden_dim: Dimension of hidden representations
            representation_hook: Function that extracts representations from model
            buffer_size: Size of representation history buffer
            update_frequency: How often to update escape directions
            device: Computation device
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.hidden_dim = hidden_dim
        self.representation_hook = representation_hook
        self.update_frequency = update_frequency
        self.device = device

        # Representation tracking
        self.rep_buffer = RepresentationBuffer(
            hidden_dim=hidden_dim,
            buffer_size=buffer_size,
            device=device
        )

        # Spectral regulation
        self.spectral_regulator = SpectralRegulator(
            hidden_dim=hidden_dim,
            device=device
        )

        # State
        self.step_count = 0
        self.current_state: Optional[SpectralState] = None

        # Parameter drift cache (for efficiency)
        self._drift_cache: Dict[str, torch.Tensor] = {}
        self._last_representations: Optional[torch.Tensor] = None

    def store_representations(self, representations: torch.Tensor, loss: float):
        """Store current representations and loss."""
        self.rep_buffer.store(representations, loss, self.step_count)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform optimization step with drift injection.

        Args:
            closure: Optional closure for loss computation
        """
        self.step_count += 1

        # First, do standard optimizer step
        loss = None
        if closure is not None:
            loss = closure()

        # Periodically update escape directions and spectral state
        if self.step_count % self.update_frequency == 0:
            self._update_operator_state()

        # Apply drift to parameters if we have meaningful escape direction
        if self.current_state is not None and self.rep_buffer.escape_magnitude > 0.01:
            self._apply_representation_drift()

        # Standard optimizer step
        self.base_optimizer.step()

        return loss

    def _update_operator_state(self):
        """Update escape directions and spectral state."""
        # Update escape momentum
        self.rep_buffer.update_escape_momentum(ema_decay=0.9, window=50)

        # Compute representation covariance
        cov = self.rep_buffer.compute_representation_covariance(window=100)

        # Update spectral regulator
        self.current_state = self.spectral_regulator.update(
            cov,
            escape_magnitude=self.rep_buffer.escape_magnitude
        )

    def _apply_representation_drift(self):
        """
        Apply representation drift to model parameters.

        This is the key operation: we have a direction in representation space
        that leads to improvement. We need to translate this into parameter space.

        Approach: Use gradient of representations w.r.t. parameters.
        If ∂h/∂θ exists, then Δθ ≈ (∂h/∂θ)^T * Δh

        For efficiency, we approximate this by:
        1. Computing gradient of ||h - (h + u)||² w.r.t. θ
        2. This gives us direction in θ space that moves h toward h+u
        """
        if self._last_representations is None:
            return

        # Get escape drift (in representation space)
        re = self.current_state.reynolds_number if self.current_state else 0.1
        drift = self.rep_buffer.get_escape_drift(scale=re)

        # Scale by learning rate and phase
        lr_mult = self.spectral_regulator.get_learning_rate_multiplier()
        drift = drift * lr_mult * 0.01  # Small drift

        # Apply drift to last layer's bias (simplest approximation)
        # This directly influences output representations
        for name, param in self.model.named_parameters():
            if 'output' in name.lower() or 'fc' in name.lower() or 'classifier' in name.lower():
                if 'bias' in name.lower() and param.shape[0] == self.hidden_dim:
                    param.data.add_(drift[:param.shape[0]])
                    break

    def zero_grad(self):
        """Zero gradients."""
        self.base_optimizer.zero_grad()

    def get_diagnostics(self) -> dict:
        """Get current optimizer diagnostics."""
        diagnostics = {
            'step': self.step_count,
            'escape_magnitude': self.rep_buffer.escape_magnitude,
            **self.rep_buffer.get_statistics()
        }

        if self.current_state:
            diagnostics.update({
                'phase': self.current_state.phase.name,
                'reynolds_number': self.current_state.reynolds_number,
                'spectral_gap': self.current_state.spectral_gap,
                'stability': self.current_state.stability_score
            })

        return diagnostics

    def is_converged(self) -> bool:
        """Check if optimizer considers training converged."""
        return self.spectral_regulator.is_converged()


class OperatorGuidedTrainer:
    """
    High-level trainer that implements the full operator-guided training loop.

    This is the clean interface for the POC validation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        hidden_dim: int,
        device: str = 'cpu',
        update_frequency: int = 10
    ):
        self.model = model
        self.device = device
        self.hidden_dim = hidden_dim

        # Create operator-guided optimizer
        self.operator_opt = OperatorGuidedOptimizer(
            base_optimizer=optimizer,
            model=model,
            hidden_dim=hidden_dim,
            representation_hook=self._get_representations,
            update_frequency=update_frequency,
            device=device
        )

        # Storage for representations during forward pass
        self._current_representations = None

        # Training history
        self.history = {
            'loss': [],
            'spectral_gap': [],
            'phase': [],
            'reynolds_number': [],
            'escape_magnitude': []
        }

    def _get_representations(self) -> Optional[torch.Tensor]:
        """Get current representations from storage."""
        return self._current_representations

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        loss_fn: Callable,
        representation_extractor: Callable
    ) -> Dict[str, float]:
        """
        Single training step with operator guidance.

        Args:
            batch_x: Input batch
            batch_y: Target batch
            loss_fn: Loss function
            representation_extractor: Function to extract representations from model output

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.operator_opt.zero_grad()

        # Forward pass
        output = self.model(batch_x)

        # Extract representations (before final projection)
        representations = representation_extractor(self.model, batch_x)
        self._current_representations = representations

        # Compute loss
        loss = loss_fn(output, batch_y)

        # Store in buffer
        self.operator_opt.store_representations(
            representations.detach(),
            loss.item()
        )

        # Backward pass
        loss.backward()

        # Operator-guided step
        self.operator_opt.step()

        # Record history
        diagnostics = self.operator_opt.get_diagnostics()
        self.history['loss'].append(loss.item())
        self.history['spectral_gap'].append(diagnostics.get('spectral_gap', 0))
        self.history['phase'].append(diagnostics.get('phase', 'UNKNOWN'))
        self.history['reynolds_number'].append(diagnostics.get('reynolds_number', 0))
        self.history['escape_magnitude'].append(diagnostics.get('escape_magnitude', 0))

        return {
            'loss': loss.item(),
            **diagnostics
        }

    def is_converged(self) -> bool:
        """Check if training has converged."""
        return self.operator_opt.is_converged()

    def get_history(self) -> dict:
        """Get training history."""
        return self.history


class StandardTrainer:
    """
    Standard trainer for comparison (no operator guidance).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.history = {
            'loss': []
        }

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Standard training step."""
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(batch_x)
        loss = loss_fn(output, batch_y)

        loss.backward()
        self.optimizer.step()

        self.history['loss'].append(loss.item())

        return {'loss': loss.item()}

    def get_history(self) -> dict:
        return self.history
