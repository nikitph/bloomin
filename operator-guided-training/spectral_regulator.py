"""
Spectral Regulator: Adaptive control of training dynamics.

The key insight from the theory:
- Loss ↓ is a terrible signal for convergence
- Gradient norm ↓ is misleading
- Curvature is local

The ONLY global convergence signal is spectral separation of modes.

This module implements:
1. Real-time spectral gap estimation from representations
2. Phase detection (identifying which phase of training we're in)
3. Adaptive Reynolds number (Re) control based on spectral state
"""

import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque


class TrainingPhase(Enum):
    """
    Training phases based on spectral analysis.

    Phase 1: Diffusion-dominated (ψ spread across minima)
    Phase 2: Advection emergence (directional flow develops)
    Phase 3: Gap opening (eigenvalue separation)
    Phase 4: Hypocoercive collapse (convergence to invariant measure)
    """
    DIFFUSION_DOMINATED = 1
    ADVECTION_EMERGENCE = 2
    GAP_OPENING = 3
    HYPOCOERCIVE_COLLAPSE = 4


@dataclass
class SpectralState:
    """Current spectral state of training."""
    spectral_gap: float
    top_eigenvalues: torch.Tensor
    explained_variance_ratio: float
    phase: TrainingPhase
    reynolds_number: float
    stability_score: float


class SpectralRegulator:
    """
    Monitors spectral properties of representation dynamics and
    adaptively controls training parameters.

    Key observables:
    - Spectral gap: (λ₁ - λ₂) / λ₁
    - Explained variance: how much of representation variance is in top mode
    - Gap stability: how consistent the gap is over time

    Control outputs:
    - Reynolds number (Re): controls strength of advective drift
    - Phase detection: which training phase we're in
    - Convergence signal: when to stop or reduce learning rate
    """

    def __init__(
        self,
        hidden_dim: int,
        history_size: int = 100,
        phase_threshold_low: float = 0.1,
        phase_threshold_high: float = 0.5,
        device: str = 'cpu'
    ):
        self.hidden_dim = hidden_dim
        self.history_size = history_size
        self.device = device

        # Phase transition thresholds
        self.phase_threshold_low = phase_threshold_low
        self.phase_threshold_high = phase_threshold_high

        # History for stability estimation
        self.gap_history: deque = deque(maxlen=history_size)
        self.eigenvalue_history: deque = deque(maxlen=history_size)

        # Current state
        self.current_phase = TrainingPhase.DIFFUSION_DOMINATED
        self.reynolds_number = 0.0
        self.step_count = 0

        # Convergence detection
        self.stable_gap_count = 0
        self.convergence_threshold = 50  # Steps of stable gap

    def compute_spectral_properties(
        self,
        covariance: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute key spectral properties from covariance matrix.

        Returns:
            eigenvalues: Top-k eigenvalues
            spectral_gap: Relative gap between top two eigenvalues
            explained_variance: Fraction of variance in top eigenvalue
        """
        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(covariance)
            eigenvalues = eigenvalues.real
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]

            # Keep top k
            eigenvalues = eigenvalues[:min(k, len(eigenvalues))]

            # Spectral gap
            if len(eigenvalues) >= 2 and eigenvalues[0] > 1e-8:
                spectral_gap = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
            else:
                spectral_gap = 0.0

            # Explained variance ratio
            total_variance = eigenvalues.sum()
            if total_variance > 1e-8:
                explained_variance = eigenvalues[0] / total_variance
            else:
                explained_variance = 1.0

            return eigenvalues, spectral_gap.item(), explained_variance.item()

        except Exception as e:
            # Fallback for numerical issues
            return (
                torch.ones(k, device=self.device),
                0.0,
                1.0 / k
            )

    def detect_phase(self, spectral_gap: float, explained_variance: float) -> TrainingPhase:
        """
        Detect current training phase based on spectral properties.

        Phase transitions:
        1 → 2: Gap starts increasing from near-zero
        2 → 3: Gap exceeds low threshold, clear separation emerging
        3 → 4: Gap exceeds high threshold, convergence imminent
        """
        # Phase 4: Hypocoercive collapse
        if spectral_gap > self.phase_threshold_high and explained_variance > 0.7:
            return TrainingPhase.HYPOCOERCIVE_COLLAPSE

        # Phase 3: Gap opening
        if spectral_gap > self.phase_threshold_low:
            return TrainingPhase.GAP_OPENING

        # Phase 2: Advection emergence (some structure, but gap not clear)
        if explained_variance > 0.3:
            return TrainingPhase.ADVECTION_EMERGENCE

        # Phase 1: Diffusion dominated
        return TrainingPhase.DIFFUSION_DOMINATED

    def compute_reynolds_number(
        self,
        phase: TrainingPhase,
        spectral_gap: float,
        escape_magnitude: float
    ) -> float:
        """
        Compute adaptive Reynolds number based on current state.

        The Reynolds number controls the balance between:
        - Diffusion (exploration, random walk in loss landscape)
        - Advection (directed flow along successful escape directions)

        High Re → more advection → follow learned escape directions
        Low Re → more diffusion → explore broadly

        Adaptation strategy:
        - Phase 1: Low Re (need exploration)
        - Phase 2: Increasing Re (start following successful directions)
        - Phase 3: High Re (commit to found direction)
        - Phase 4: Decreasing Re (fine-tuning, less momentum)
        """
        base_re = {
            TrainingPhase.DIFFUSION_DOMINATED: 0.1,
            TrainingPhase.ADVECTION_EMERGENCE: 0.5,
            TrainingPhase.GAP_OPENING: 1.0,
            TrainingPhase.HYPOCOERCIVE_COLLAPSE: 0.3
        }

        re = base_re[phase]

        # Scale by spectral gap (more confident → stronger drift)
        re *= (1.0 + spectral_gap)

        # Scale by escape magnitude (if we have clear direction, use it)
        re *= (1.0 + escape_magnitude)

        # Clamp to reasonable range
        return np.clip(re, 0.0, 2.0)

    def update(
        self,
        covariance: torch.Tensor,
        escape_magnitude: float = 0.0
    ) -> SpectralState:
        """
        Update spectral state based on current covariance.

        Args:
            covariance: Representation covariance matrix
            escape_magnitude: Current magnitude of escape direction

        Returns:
            Current spectral state with all diagnostics
        """
        self.step_count += 1

        # Compute spectral properties
        eigenvalues, spectral_gap, explained_variance = self.compute_spectral_properties(
            covariance
        )

        # Update history
        self.gap_history.append(spectral_gap)
        self.eigenvalue_history.append(eigenvalues[0].item())

        # Detect phase
        phase = self.detect_phase(spectral_gap, explained_variance)

        # Check for phase transition
        if phase != self.current_phase:
            print(f"Phase transition: {self.current_phase.name} → {phase.name} "
                  f"(gap={spectral_gap:.3f}, var={explained_variance:.3f})")
            self.current_phase = phase

        # Compute Reynolds number
        self.reynolds_number = self.compute_reynolds_number(
            phase, spectral_gap, escape_magnitude
        )

        # Compute stability score
        stability = self._compute_stability()

        # Update convergence detection
        if stability > 0.8 and phase == TrainingPhase.HYPOCOERCIVE_COLLAPSE:
            self.stable_gap_count += 1
        else:
            self.stable_gap_count = 0

        return SpectralState(
            spectral_gap=spectral_gap,
            top_eigenvalues=eigenvalues,
            explained_variance_ratio=explained_variance,
            phase=phase,
            reynolds_number=self.reynolds_number,
            stability_score=stability
        )

    def _compute_stability(self) -> float:
        """
        Compute stability of spectral gap over recent history.

        High stability indicates we're in a stable basin.
        """
        if len(self.gap_history) < 10:
            return 0.0

        recent = list(self.gap_history)[-20:]
        mean_gap = np.mean(recent)
        std_gap = np.std(recent)

        if mean_gap < 1e-8:
            return 0.0

        # Coefficient of variation (inverted - lower CV = higher stability)
        cv = std_gap / mean_gap
        stability = 1.0 / (1.0 + cv)

        return stability

    def is_converged(self) -> bool:
        """Check if training has converged based on spectral criteria."""
        return self.stable_gap_count >= self.convergence_threshold

    def get_learning_rate_multiplier(self) -> float:
        """
        Get learning rate multiplier based on current phase.

        This implements annealing that's actually informed by
        convergence state, not just step count.
        """
        multipliers = {
            TrainingPhase.DIFFUSION_DOMINATED: 1.0,
            TrainingPhase.ADVECTION_EMERGENCE: 0.8,
            TrainingPhase.GAP_OPENING: 0.5,
            TrainingPhase.HYPOCOERCIVE_COLLAPSE: 0.2
        }
        return multipliers[self.current_phase]

    def get_diagnostics(self) -> dict:
        """Get current diagnostics for logging."""
        return {
            'phase': self.current_phase.name,
            'reynolds_number': self.reynolds_number,
            'spectral_gap': self.gap_history[-1] if self.gap_history else 0.0,
            'stability': self._compute_stability(),
            'converged': self.is_converged(),
            'step_count': self.step_count
        }


class MultiScaleSpectralRegulator:
    """
    Spectral regulator operating at multiple scales.

    Tracks spectral properties at:
    - Local scale (recent steps)
    - Medium scale (epoch level)
    - Global scale (entire training)

    This allows detecting both fast transients and slow convergence.
    """

    def __init__(
        self,
        hidden_dim: int,
        local_window: int = 50,
        medium_window: int = 500,
        device: str = 'cpu'
    ):
        self.local_regulator = SpectralRegulator(
            hidden_dim,
            history_size=local_window,
            phase_threshold_low=0.05,
            phase_threshold_high=0.3,
            device=device
        )

        self.medium_regulator = SpectralRegulator(
            hidden_dim,
            history_size=medium_window,
            phase_threshold_low=0.1,
            phase_threshold_high=0.5,
            device=device
        )

        self.hidden_dim = hidden_dim
        self.device = device

    def update(
        self,
        covariance: torch.Tensor,
        escape_magnitude: float = 0.0
    ) -> Tuple[SpectralState, SpectralState]:
        """Update both regulators and return both states."""
        local_state = self.local_regulator.update(covariance, escape_magnitude)
        medium_state = self.medium_regulator.update(covariance, escape_magnitude)

        return local_state, medium_state

    def get_combined_reynolds(self) -> float:
        """Get Reynolds number combining both scales."""
        # Use geometric mean to balance both scales
        local_re = self.local_regulator.reynolds_number
        medium_re = self.medium_regulator.reynolds_number

        return np.sqrt(local_re * medium_re)

    def is_converged(self) -> bool:
        """Converged only if both scales agree."""
        return (
            self.local_regulator.is_converged() and
            self.medium_regulator.current_phase == TrainingPhase.HYPOCOERCIVE_COLLAPSE
        )
