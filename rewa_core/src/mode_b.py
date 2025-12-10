"""
FR-6: Mode B — Policy Injection

Activated when state == AMBIGUOUS.
Applies policy priors ρ(μ) to select meaning from admissible region.

μ* = argmax_{μ ∈ A(W)} ρ(μ)
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from .semantic_space import Witness
from .hull import SphericalHull
from .policy import Policy, PolicyEngine
from .entropy import RewaState


@dataclass
class ModeBResult:
    """Result of Mode B policy injection."""
    selected_meaning: np.ndarray
    policy_id: str
    policy_score: float
    selection_method: str
    candidates_evaluated: int
    in_admissible_region: bool
    details: Dict[str, Any]


class ModeBEngine:
    """
    Mode B Engine: Policy-guided meaning selection.

    When evidence is ambiguous (hull entropy high), Mode B:
    1. Samples candidate meanings from admissible region A(W)
    2. Scores each candidate with policy function ρ(μ)
    3. Selects μ* = argmax ρ(μ)

    Key guarantee: μ* is ALWAYS in A(W), meaning it's consistent with evidence.
    Policy NEVER overwrites evidence, only selects among valid interpretations.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        hull_computer: Optional[SphericalHull] = None,
        n_candidates: int = 1000,
        refinement_iterations: int = 5
    ):
        self.policy_engine = policy_engine
        self.hull_computer = hull_computer or SphericalHull()
        self.n_candidates = n_candidates
        self.refinement_iterations = refinement_iterations

    def select_meaning(
        self,
        witnesses: List[Witness],
        policy_id: str,
        state: RewaState
    ) -> ModeBResult:
        """
        Select optimal meaning given evidence and policy.

        Only activates if state == AMBIGUOUS.
        For VALID state, returns geodesic center without policy bias.
        For IMPOSSIBLE state, returns None.

        Args:
            witnesses: Evidence witnesses
            policy_id: ID of policy to apply
            state: Current system state

        Returns:
            ModeBResult with selected meaning and metadata
        """
        # Check preconditions
        if state == RewaState.IMPOSSIBLE:
            return ModeBResult(
                selected_meaning=np.zeros(witnesses[0].embedding.shape if witnesses else 1),
                policy_id=policy_id,
                policy_score=float('-inf'),
                selection_method="none",
                candidates_evaluated=0,
                in_admissible_region=False,
                details={"error": "Cannot select meaning from contradictory evidence"}
            )

        if len(witnesses) == 0:
            return ModeBResult(
                selected_meaning=np.zeros(1),
                policy_id=policy_id,
                policy_score=float('-inf'),
                selection_method="none",
                candidates_evaluated=0,
                in_admissible_region=False,
                details={"error": "No witnesses provided"}
            )

        # Get policy
        policy = self.policy_engine.get(policy_id)
        if policy is None:
            raise ValueError(f"Policy {policy_id} not found")

        # For VALID state (low ambiguity), just return geodesic center
        if state == RewaState.VALID:
            hull_result = self.hull_computer.compute(witnesses)
            center = hull_result.center

            return ModeBResult(
                selected_meaning=center,
                policy_id=policy_id,
                policy_score=policy.score(center),
                selection_method="geodesic_center",
                candidates_evaluated=1,
                in_admissible_region=True,
                details={
                    "message": "Low ambiguity - using geodesic center",
                    "hull_radius": hull_result.angular_radius
                }
            )

        # AMBIGUOUS state: Full Mode B activation
        return self._mode_b_selection(witnesses, policy)

    def _mode_b_selection(
        self,
        witnesses: List[Witness],
        policy: Policy
    ) -> ModeBResult:
        """
        Full Mode B meaning selection.

        1. Sample candidates from admissible region
        2. Score with policy
        3. Refine around best candidate
        4. Return optimal μ*
        """
        # Sample initial candidates
        candidates = self.hull_computer.sample_interior(
            witnesses,
            n_samples=self.n_candidates,
            method="combination"  # Guarantees candidates are in hull
        )

        # Score all candidates
        scores = policy.batch_score(candidates)

        # Find best
        best_idx = int(np.argmax(scores))
        best_meaning = candidates[best_idx]
        best_score = float(scores[best_idx])

        # Refine: sample around best candidate
        for iteration in range(self.refinement_iterations):
            # Sample in neighborhood of current best
            refined_candidates = self._sample_neighborhood(
                best_meaning,
                witnesses,
                n_samples=self.n_candidates // 2
            )

            if len(refined_candidates) == 0:
                break

            refined_scores = policy.batch_score(refined_candidates)

            # Update if better found
            if len(refined_scores) > 0 and np.max(refined_scores) > best_score:
                best_idx = int(np.argmax(refined_scores))
                best_meaning = refined_candidates[best_idx]
                best_score = float(refined_scores[best_idx])

        # Verify final meaning is in admissible region
        is_admissible, hull_details = self.hull_computer.contains(
            best_meaning, witnesses
        )

        return ModeBResult(
            selected_meaning=best_meaning,
            policy_id=policy.spec.id,
            policy_score=best_score,
            selection_method="mode_b_optimization",
            candidates_evaluated=self.n_candidates + (
                self.n_candidates // 2 * self.refinement_iterations
            ),
            in_admissible_region=is_admissible,
            details={
                "refinement_iterations": self.refinement_iterations,
                "policy_threshold": policy.spec.threshold,
                "meets_threshold": best_score >= policy.spec.threshold,
                "hull_check": hull_details
            }
        )

    def _sample_neighborhood(
        self,
        center: np.ndarray,
        witnesses: List[Witness],
        n_samples: int,
        angular_radius: float = 0.1
    ) -> np.ndarray:
        """
        Sample points in neighborhood of center, constrained to hull.

        Uses tangent space perturbation with geodesic projection.
        """
        d = len(center)
        samples = []

        for _ in range(n_samples * 2):  # Over-sample to account for rejections
            # Perturb in tangent space
            tangent = np.random.randn(d) * angular_radius
            tangent = tangent - np.dot(tangent, center) * center  # Project to tangent

            # Exponential map (move along geodesic)
            norm = np.linalg.norm(tangent)
            if norm < 1e-10:
                perturbed = center.copy()
            else:
                perturbed = np.cos(norm) * center + np.sin(norm) * tangent / norm

            # Check if in hull
            is_in_hull, _ = self.hull_computer.contains(perturbed, witnesses)
            if is_in_hull:
                samples.append(perturbed)

            if len(samples) >= n_samples:
                break

        return np.array(samples) if samples else np.array([]).reshape(0, d)

    def compare_policies(
        self,
        witnesses: List[Witness],
        policy_ids: List[str],
        state: RewaState
    ) -> Dict[str, ModeBResult]:
        """
        Compare meaning selection across multiple policies.

        Demonstrates FR-6 acceptance criteria:
        "Changing policy changes outcome with same evidence."
        """
        results = {}

        for policy_id in policy_ids:
            results[policy_id] = self.select_meaning(witnesses, policy_id, state)

        return results

    def verify_evidence_preservation(
        self,
        witnesses: List[Witness],
        selected_meaning: np.ndarray
    ) -> Dict[str, Any]:
        """
        Verify that policy did not overwrite evidence.

        Checks that selected meaning is still in admissible region
        defined by witnesses.
        """
        W = np.array([w.embedding for w in witnesses])
        dots = W @ selected_meaning

        return {
            "all_positive": bool(np.all(dots > 0)),
            "min_agreement": float(np.min(dots)),
            "mean_agreement": float(np.mean(dots)),
            "witness_agreements": {
                witnesses[i].id: float(dots[i])
                for i in range(len(witnesses))
            }
        }


class PolicySwapExperiment:
    """
    Experiment to verify policy swap changes behavior deterministically.

    Same evidence + different policy = different (but valid) outcome.
    """

    def __init__(self, mode_b_engine: ModeBEngine):
        self.mode_b_engine = mode_b_engine

    def run(
        self,
        witnesses: List[Witness],
        policy_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Run policy swap experiment.

        Returns:
            Analysis of how different policies affect meaning selection.
        """
        from .entropy import EntropyEstimator
        entropy_estimator = EntropyEstimator()
        entropy_result = entropy_estimator.estimate(witnesses)

        results = {
            "evidence": {
                "num_witnesses": len(witnesses),
                "witness_ids": [w.id for w in witnesses],
                "entropy": entropy_result.entropy,
                "state": entropy_result.state.value
            },
            "policy_results": {},
            "comparison": {}
        }

        meanings = []
        scores = []

        for policy_id in policy_ids:
            mode_b_result = self.mode_b_engine.select_meaning(
                witnesses, policy_id, entropy_result.state
            )

            results["policy_results"][policy_id] = {
                "score": mode_b_result.policy_score,
                "in_hull": mode_b_result.in_admissible_region,
                "method": mode_b_result.selection_method
            }

            meanings.append(mode_b_result.selected_meaning)
            scores.append(mode_b_result.policy_score)

        # Compare meanings across policies
        if len(meanings) >= 2:
            for i in range(len(policy_ids)):
                for j in range(i + 1, len(policy_ids)):
                    angle = np.arccos(np.clip(
                        np.dot(meanings[i], meanings[j]), -1.0, 1.0
                    ))
                    results["comparison"][f"{policy_ids[i]}_vs_{policy_ids[j]}"] = {
                        "angular_difference": float(angle),
                        "different_selection": angle > 0.01  # ~0.5 degrees
                    }

        # Verify all selections respect evidence
        results["evidence_preserved"] = all(
            self.mode_b_engine.verify_evidence_preservation(witnesses, m)["all_positive"]
            for m in meanings
        )

        return results
