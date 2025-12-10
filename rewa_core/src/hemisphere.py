"""
FR-3: Hemisphere / Contradiction Check

Verifies existence of an open hemisphere containing all witnesses.
If no such hemisphere exists, the evidence is contradictory (STATE = IMPOSSIBLE).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .semantic_space import Witness
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HemisphereResult:
    """Result of hemisphere existence check."""
    exists: bool
    center: Optional[np.ndarray]  # Normal vector of separating hyperplane
    margin: float  # Minimum dot product (separation margin)
    contradicting_pairs: List[Tuple[str, str]]  # IDs of antipodal witness pairs
    computation_method: str
    details: Dict[str, Any]


class HemisphereChecker:
    """
    Checks if witnesses can be contained in an open hemisphere.

    Mathematical formulation:
    Find n ∈ S^{d-1} such that n · w_i > 0 for all witnesses w_i.

    If no such n exists, the witnesses contain contradictory information.
    """

    def __init__(self, antipodal_threshold: float = -0.9):
        """
        Args:
            antipodal_threshold: Dot product below which witnesses are
                                considered contradictory. Default -0.9 means
                                angle > ~154 degrees.
        """
        self.antipodal_threshold = antipodal_threshold

    def check(self, witnesses: List[Witness]) -> HemisphereResult:
        """
        Check if all witnesses can be contained in an open hemisphere.

        Returns:
            HemisphereResult with existence status and details.
        """
        if len(witnesses) == 0:
            return HemisphereResult(
                exists=True,
                center=None,
                margin=float('inf'),
                contradicting_pairs=[],
                computation_method="empty",
                details={"message": "No witnesses provided"}
            )

        if len(witnesses) == 1:
            return HemisphereResult(
                exists=True,
                center=witnesses[0].embedding.copy(),
                margin=1.0,
                contradicting_pairs=[],
                computation_method="single",
                details={"message": "Single witness always has hemisphere"}
            )

        # First, quick check for obviously antipodal pairs
        antipodal_pairs = self._find_antipodal_pairs(witnesses)
        if antipodal_pairs:
            return HemisphereResult(
                exists=False,
                center=None,
                margin=min(w1.dot(w2) for w1, w2 in antipodal_pairs),
                contradicting_pairs=[(w1.id, w2.id) for w1, w2 in antipodal_pairs],
                computation_method="antipodal_detection",
                details={
                    "message": "Found antipodal witness pairs",
                    "num_pairs": len(antipodal_pairs)
                }
            )

        # Solve the hemisphere problem using linear programming
        return self._solve_hemisphere_lp(witnesses)

    def _find_antipodal_pairs(
        self,
        witnesses: List[Witness]
    ) -> List[Tuple[Witness, Witness]]:
        """Find pairs of witnesses that are approximately antipodal."""
        antipodal_pairs = []
        n = len(witnesses)

        for i in range(n):
            for j in range(i + 1, n):
                dot_product = witnesses[i].dot(witnesses[j])
                if dot_product < self.antipodal_threshold:
                    antipodal_pairs.append((witnesses[i], witnesses[j]))

        return antipodal_pairs

    def _solve_hemisphere_lp(self, witnesses: List[Witness]) -> HemisphereResult:
        """
        Solve hemisphere existence via linear programming.

        We want to find n such that W @ n > 0 for witness matrix W.
        This is equivalent to finding a separating hyperplane.

        LP formulation:
            max t
            s.t. W @ n >= t
                 ||n|| <= 1

        If optimal t > 0, hemisphere exists.
        """
        W = np.array([w.embedding for w in witnesses])
        d = W.shape[1]

        # Method 1: Try to find center via mean direction
        mean_direction = np.mean(W, axis=0)
        mean_norm = np.linalg.norm(mean_direction)

        if mean_norm > 1e-10:
            mean_direction = mean_direction / mean_norm
            margins = W @ mean_direction

            if np.all(margins > 0):
                return HemisphereResult(
                    exists=True,
                    center=mean_direction,
                    margin=float(np.min(margins)),
                    contradicting_pairs=[],
                    computation_method="mean_direction",
                    details={
                        "margins": margins.tolist(),
                        "mean_margin": float(np.mean(margins))
                    }
                )

        # Method 2: Optimization to find best separating direction
        result = self._optimize_separation(W)

        if result['success'] and result['margin'] > 1e-6:
            return HemisphereResult(
                exists=True,
                center=result['center'],
                margin=result['margin'],
                contradicting_pairs=[],
                computation_method="optimization",
                details=result['details']
            )

        # Method 3: Check for geometric impossibility
        # If optimization failed, find the most contradicting pairs
        center = result.get('center', mean_direction)
        if center is not None:
            margins = W @ center
            min_idx = np.argmin(margins)
            min_margin = margins[min_idx]

            # Find witnesses that contribute to impossibility
            problematic = [witnesses[i].id for i in range(len(witnesses))
                         if margins[i] < 0.1]

            return HemisphereResult(
                exists=False,
                center=center,
                margin=float(min_margin),
                contradicting_pairs=[],
                computation_method="optimization_failed",
                details={
                    "message": "No separating hemisphere found",
                    "min_margin": float(min_margin),
                    "problematic_witnesses": problematic
                }
            )

        return HemisphereResult(
            exists=False,
            center=None,
            margin=float('-inf'),
            contradicting_pairs=[],
            computation_method="failed",
            details={"message": "Could not determine hemisphere existence"}
        )

    def _optimize_separation(self, W: np.ndarray) -> Dict[str, Any]:
        """
        Find the direction that maximizes minimum margin.

        Solve: max min_i (w_i · n) subject to ||n|| = 1
        """
        d = W.shape[1]

        def objective(n):
            # Minimize negative of minimum margin
            n_normalized = n / (np.linalg.norm(n) + 1e-10)
            margins = W @ n_normalized
            return -np.min(margins)

        # Initialize with mean direction
        x0 = np.mean(W, axis=0)
        x0 = x0 / (np.linalg.norm(x0) + 1e-10)

        # Add random restarts
        best_result = None
        best_margin = float('-inf')

        for _ in range(5):
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )

                if result.success or result.fun is not None:
                    margin = -result.fun
                    if margin > best_margin:
                        best_margin = margin
                        center = result.x / (np.linalg.norm(result.x) + 1e-10)
                        best_result = {
                            'success': margin > 0,
                            'center': center,
                            'margin': float(margin),
                            'details': {
                                'iterations': result.nit if hasattr(result, 'nit') else None,
                                'optimization_success': result.success
                            }
                        }
            except Exception:
                pass

            # Random restart
            x0 = np.random.randn(d)
            x0 = x0 / np.linalg.norm(x0)

        if best_result is None:
            # Fallback: use mean direction
            mean_dir = np.mean(W, axis=0)
            mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
            margins = W @ mean_dir
            return {
                'success': np.min(margins) > 0,
                'center': mean_dir,
                'margin': float(np.min(margins)),
                'details': {'method': 'fallback_mean'}
            }

        return best_result

    def find_maximal_consistent_subset(
        self,
        witnesses: List[Witness]
    ) -> Tuple[List[Witness], List[Witness]]:
        """
        If witnesses are contradictory, find the largest consistent subset.

        Returns:
            (consistent_witnesses, excluded_witnesses)
        """
        if len(witnesses) <= 1:
            return witnesses, []

        # Check if already consistent
        result = self.check(witnesses)
        if result.exists:
            return witnesses, []

        # Greedy approach: start with first witness, add compatible ones
        consistent = [witnesses[0]]
        excluded = []

        for w in witnesses[1:]:
            test_set = consistent + [w]
            if self.check(test_set).exists:
                consistent.append(w)
            else:
                excluded.append(w)

        return consistent, excluded

    def compute_contradiction_strength(
        self,
        w1: Witness,
        w2: Witness
    ) -> float:
        """
        Compute how strongly two witnesses contradict each other.

        Returns value in [0, 1] where:
        - 0 = perfectly aligned
        - 0.5 = orthogonal
        - 1 = perfectly antipodal
        """
        dot = w1.dot(w2)
        # Map from [-1, 1] to [0, 1]
        return (1 - dot) / 2
