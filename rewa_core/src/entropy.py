"""
FR-5: Ambiguity & Entropy Detection

Estimates hull "size" / entropy to determine system state:
- VALID: hull entropy < ε (low ambiguity)
- AMBIGUOUS: hull entropy ≥ ε (high ambiguity, needs Mode B)
- IMPOSSIBLE: no hemisphere (contradictory evidence)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from .semantic_space import Witness
from .hull import SphericalHull, SphericalHullResult


class RewaState(Enum):
    """System state based on entropy analysis."""
    VALID = "valid"          # Low ambiguity, can respond
    AMBIGUOUS = "ambiguous"  # High ambiguity, needs Mode B
    IMPOSSIBLE = "impossible"  # Contradictory evidence


@dataclass
class EntropyResult:
    """Result of entropy estimation."""
    state: RewaState
    entropy: float
    angular_variance: float
    surface_coverage: float
    effective_constraints: float
    threshold_used: float
    details: Dict[str, Any]


class EntropyEstimator:
    """
    Estimates the entropy/ambiguity of the semantic hull.

    Uses multiple metrics:
    1. Angular variance of witnesses
    2. Surface coverage proxy
    3. Effective constraint count

    Key property:
    - Adding diverse evidence INCREASES entropy
    - Adding consistent evidence DECREASES entropy
    """

    def __init__(
        self,
        entropy_threshold: float = 0.3,
        min_witnesses_for_valid: int = 2
    ):
        """
        Args:
            entropy_threshold: Below this, state is VALID. Above, AMBIGUOUS.
            min_witnesses_for_valid: Minimum witnesses needed for VALID state.
        """
        self.entropy_threshold = entropy_threshold
        self.min_witnesses_for_valid = min_witnesses_for_valid
        self.hull_computer = SphericalHull()

    def estimate(
        self,
        witnesses: List[Witness],
        hemisphere_exists: bool = True
    ) -> EntropyResult:
        """
        Estimate entropy of the witness set.

        Args:
            witnesses: List of witness vectors
            hemisphere_exists: Result from HemisphereChecker

        Returns:
            EntropyResult with state and metrics
        """
        # Handle impossible case
        if not hemisphere_exists:
            return EntropyResult(
                state=RewaState.IMPOSSIBLE,
                entropy=float('inf'),
                angular_variance=float('inf'),
                surface_coverage=1.0,
                effective_constraints=0.0,
                threshold_used=self.entropy_threshold,
                details={"message": "No consistent hemisphere - contradictory evidence"}
            )

        # Handle empty/minimal witnesses
        if len(witnesses) == 0:
            return EntropyResult(
                state=RewaState.AMBIGUOUS,
                entropy=1.0,
                angular_variance=float('inf'),
                surface_coverage=1.0,
                effective_constraints=0.0,
                threshold_used=self.entropy_threshold,
                details={"message": "No witnesses - maximum ambiguity"}
            )

        if len(witnesses) == 1:
            return EntropyResult(
                state=RewaState.AMBIGUOUS,
                entropy=0.5,  # Half sphere
                angular_variance=0.0,
                surface_coverage=0.5,
                effective_constraints=1.0,
                threshold_used=self.entropy_threshold,
                details={"message": "Single witness - hemisphere of ambiguity"}
            )

        # Compute hull metrics
        hull_result = self.hull_computer.compute(witnesses)

        # Compute individual entropy components
        angular_variance = self._compute_angular_variance(witnesses)
        surface_coverage = self._compute_surface_coverage(witnesses, hull_result)
        effective_constraints = self._compute_effective_constraints(witnesses)

        # Combined entropy score
        entropy = self._compute_entropy(
            angular_variance,
            surface_coverage,
            effective_constraints,
            len(witnesses)
        )

        # Determine state
        if len(witnesses) < self.min_witnesses_for_valid:
            state = RewaState.AMBIGUOUS
        elif entropy < self.entropy_threshold:
            state = RewaState.VALID
        else:
            state = RewaState.AMBIGUOUS

        return EntropyResult(
            state=state,
            entropy=entropy,
            angular_variance=angular_variance,
            surface_coverage=surface_coverage,
            effective_constraints=effective_constraints,
            threshold_used=self.entropy_threshold,
            details={
                "num_witnesses": len(witnesses),
                "hull_angular_radius": hull_result.angular_radius,
                "hull_volume_proxy": hull_result.volume_proxy,
                "min_witnesses_for_valid": self.min_witnesses_for_valid
            }
        )

    def _compute_angular_variance(self, witnesses: List[Witness]) -> float:
        """
        Compute variance of angles between witnesses.

        High variance = witnesses pointing in different directions = high ambiguity
        Low variance = witnesses clustered together = low ambiguity
        """
        if len(witnesses) < 2:
            return 0.0

        W = np.array([w.embedding for w in witnesses])

        # Compute all pairwise angles
        dots = W @ W.T
        np.fill_diagonal(dots, 1.0)  # Self-similarity = 1
        angles = np.arccos(np.clip(dots, -1.0, 1.0))

        # Get upper triangle (unique pairs)
        n = len(witnesses)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_angles = angles[triu_indices]

        # Variance normalized by max possible angle (π)
        variance = float(np.var(pairwise_angles) / (np.pi ** 2))
        return variance

    def _compute_surface_coverage(
        self,
        witnesses: List[Witness],
        hull_result: SphericalHullResult
    ) -> float:
        """
        Estimate what fraction of the sphere the hull covers.

        Uses spherical cap approximation.
        """
        if len(witnesses) < 2:
            return 0.5  # Single witness = half sphere

        # Use hull's volume proxy
        return hull_result.volume_proxy

    def _compute_effective_constraints(self, witnesses: List[Witness]) -> float:
        """
        Compute effective number of independent constraints.

        Orthogonal witnesses provide independent constraints.
        Parallel witnesses provide redundant constraints.
        """
        if len(witnesses) < 2:
            return float(len(witnesses))

        W = np.array([w.embedding for w in witnesses])

        # Compute pairwise similarities
        dots = W @ W.T

        # Orthogonality = 1 - |similarity|
        orthogonality = 1 - np.abs(dots)
        np.fill_diagonal(orthogonality, 0)

        # Average orthogonality (0 = all parallel, 1 = all orthogonal)
        avg_orthogonality = np.mean(orthogonality[np.triu_indices(len(witnesses), k=1)])

        # Effective constraints ≈ n * orthogonality_factor
        # If all orthogonal: n constraints
        # If all parallel: 1 constraint
        effective = 1 + (len(witnesses) - 1) * avg_orthogonality
        return float(effective)

    def _compute_entropy(
        self,
        angular_variance: float,
        surface_coverage: float,
        effective_constraints: float,
        n_witnesses: int
    ) -> float:
        """
        Combine metrics into single entropy score [0, 1].

        Low entropy = low ambiguity = VALID
        High entropy = high ambiguity = AMBIGUOUS
        """
        # Surface coverage dominates
        # More constraints reduce entropy
        constraint_factor = 1.0 / (1.0 + np.log1p(effective_constraints))

        # Combine: coverage weighted by inverse constraints
        entropy = surface_coverage * constraint_factor

        # Adjust for angular variance (high variance = more entropy)
        entropy = entropy * (1 + angular_variance)

        # Ensure [0, 1]
        return float(np.clip(entropy, 0.0, 1.0))

    def entropy_change_on_add(
        self,
        witnesses: List[Witness],
        new_witness: Witness
    ) -> Dict[str, float]:
        """
        Compute how entropy changes when adding a witness.

        Used to verify conservation law:
        - Adding diverse evidence should increase entropy (temporarily)
        - Adding consistent evidence should decrease entropy
        """
        old_result = self.estimate(witnesses, hemisphere_exists=True)

        new_witnesses = witnesses + [new_witness]
        new_result = self.estimate(new_witnesses, hemisphere_exists=True)

        return {
            "old_entropy": old_result.entropy,
            "new_entropy": new_result.entropy,
            "change": new_result.entropy - old_result.entropy,
            "old_state": old_result.state.value,
            "new_state": new_result.state.value
        }

    def find_entropy_minimizing_meaning(
        self,
        witnesses: List[Witness],
        n_samples: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Find the meaning that minimizes "local entropy" (distance to witnesses).

        This is essentially the Fréchet mean.
        """
        if len(witnesses) == 0:
            raise ValueError("Cannot find meaning with no witnesses")

        # Sample from hull
        samples = self.hull_computer.sample_interior(witnesses, n_samples)

        W = np.array([w.embedding for w in witnesses])

        best_meaning = None
        best_score = float('inf')

        for sample in samples:
            # Score = sum of distances to witnesses
            dots = W @ sample
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            score = np.sum(angles ** 2)  # Sum of squared geodesic distances

            if score < best_score:
                best_score = score
                best_meaning = sample

        return best_meaning, float(best_score)
