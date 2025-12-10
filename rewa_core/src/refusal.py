"""
FR-7: Refusal Semantics

Distinguishes refusal types with reproducible, logged decisions:
- IMPOSSIBLE: No coherent hull (contradictory evidence)
- AMBIGUOUS: Hull too large, no policy collapse
- POLICY_EXCLUDED: max ρ < τ (policy threshold not met)
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from .semantic_space import Witness
from .entropy import RewaState


class RefusalType(Enum):
    """Types of refusal with distinct semantics."""
    IMPOSSIBLE = "impossible"      # Contradictory evidence, no coherent meaning
    AMBIGUOUS = "ambiguous"        # Too many valid interpretations, need more info
    POLICY_EXCLUDED = "policy_excluded"  # Evidence valid but policy score too low
    NONE = "none"                  # Not a refusal


@dataclass
class RefusalDecision:
    """A reproducible refusal decision with full context."""
    type: RefusalType
    reason: str
    explanation: str  # Human-readable explanation
    reproducible_hash: str  # Hash for exact reproduction
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_refusal(self) -> bool:
        return self.type != RefusalType.NONE


class RefusalHandler:
    """
    Handles refusal logic with full auditability.

    Each refusal is:
    1. Reproducible - same inputs always produce same decision
    2. Logged - full context preserved
    3. Typed - distinct handling for each refusal type
    """

    def __init__(
        self,
        policy_threshold: float = 0.5,
        entropy_threshold: float = 0.3,
        require_min_witnesses: int = 2
    ):
        self.policy_threshold = policy_threshold
        self.entropy_threshold = entropy_threshold
        self.require_min_witnesses = require_min_witnesses

    def evaluate(
        self,
        witnesses: List[Witness],
        hemisphere_exists: bool,
        entropy: float,
        policy_score: Optional[float] = None,
        state: Optional[RewaState] = None
    ) -> RefusalDecision:
        """
        Evaluate whether to refuse and why.

        Priority order:
        1. IMPOSSIBLE (contradictory evidence)
        2. AMBIGUOUS (high entropy, no policy)
        3. POLICY_EXCLUDED (policy score too low)
        4. NONE (proceed)
        """
        # Generate reproducible hash
        witness_hash = self._hash_witnesses(witnesses)

        # Check for contradictory evidence
        if not hemisphere_exists or state == RewaState.IMPOSSIBLE:
            return RefusalDecision(
                type=RefusalType.IMPOSSIBLE,
                reason="Evidence contains contradictions",
                explanation=self._generate_impossible_explanation(witnesses),
                reproducible_hash=witness_hash,
                details={
                    "hemisphere_exists": hemisphere_exists,
                    "num_witnesses": len(witnesses)
                }
            )

        # Check for ambiguity without policy resolution
        if state == RewaState.AMBIGUOUS and policy_score is None:
            return RefusalDecision(
                type=RefusalType.AMBIGUOUS,
                reason="Evidence is ambiguous and no policy applied",
                explanation=self._generate_ambiguous_explanation(witnesses, entropy),
                reproducible_hash=witness_hash,
                details={
                    "entropy": entropy,
                    "entropy_threshold": self.entropy_threshold,
                    "num_witnesses": len(witnesses)
                }
            )

        # Check minimum witness requirement
        if len(witnesses) < self.require_min_witnesses:
            return RefusalDecision(
                type=RefusalType.AMBIGUOUS,
                reason=f"Insufficient evidence ({len(witnesses)} < {self.require_min_witnesses})",
                explanation=self._generate_insufficient_explanation(witnesses),
                reproducible_hash=witness_hash,
                details={
                    "num_witnesses": len(witnesses),
                    "required": self.require_min_witnesses
                }
            )

        # Check policy threshold
        if policy_score is not None and policy_score < self.policy_threshold:
            return RefusalDecision(
                type=RefusalType.POLICY_EXCLUDED,
                reason=f"Policy score {policy_score:.3f} < threshold {self.policy_threshold}",
                explanation=self._generate_policy_excluded_explanation(policy_score),
                reproducible_hash=witness_hash,
                details={
                    "policy_score": policy_score,
                    "threshold": self.policy_threshold,
                    "margin": self.policy_threshold - policy_score
                }
            )

        # No refusal
        return RefusalDecision(
            type=RefusalType.NONE,
            reason="All checks passed",
            explanation="",
            reproducible_hash=witness_hash,
            details={
                "entropy": entropy,
                "policy_score": policy_score,
                "hemisphere_exists": hemisphere_exists
            }
        )

    def _hash_witnesses(self, witnesses: List[Witness]) -> str:
        """Generate reproducible hash from witnesses."""
        import hashlib
        witness_data = "|".join(sorted([w.id for w in witnesses]))
        return hashlib.sha256(witness_data.encode()).hexdigest()[:16]

    def _generate_impossible_explanation(self, witnesses: List[Witness]) -> str:
        """Generate human-readable explanation for IMPOSSIBLE."""
        if len(witnesses) < 2:
            return "Unable to find consistent interpretation of evidence."

        return (
            f"The provided evidence ({len(witnesses)} documents) contains "
            "mutually contradictory information. No coherent meaning can be "
            "constructed that satisfies all evidence simultaneously. "
            "Please review the evidence for conflicts."
        )

    def _generate_ambiguous_explanation(
        self,
        witnesses: List[Witness],
        entropy: float
    ) -> str:
        """Generate human-readable explanation for AMBIGUOUS."""
        return (
            f"The evidence supports multiple valid interpretations "
            f"(ambiguity level: {entropy:.1%}). To provide a specific answer, "
            "please either:\n"
            "1. Provide additional constraining evidence\n"
            "2. Specify which policy/perspective to apply"
        )

    def _generate_insufficient_explanation(self, witnesses: List[Witness]) -> str:
        """Generate explanation for insufficient evidence."""
        return (
            f"Only {len(witnesses)} piece(s) of evidence provided. "
            f"At least {self.require_min_witnesses} are required for a "
            "reliable response. Please provide more context."
        )

    def _generate_policy_excluded_explanation(self, score: float) -> str:
        """Generate explanation for policy exclusion."""
        return (
            f"While the evidence is consistent, the resulting interpretation "
            f"(score: {score:.3f}) does not meet the required policy threshold "
            f"({self.policy_threshold}). The request may conflict with "
            "established policies or guidelines."
        )

    def format_refusal_response(
        self,
        decision: RefusalDecision,
        include_technical: bool = False
    ) -> str:
        """Format refusal for user-facing response."""
        if not decision.is_refusal():
            return ""

        response = f"**Unable to provide response**\n\n{decision.explanation}"

        if include_technical:
            response += f"\n\n---\n*Technical details:*\n"
            response += f"- Refusal type: {decision.type.value}\n"
            response += f"- Hash: {decision.reproducible_hash}\n"
            for key, value in decision.details.items():
                response += f"- {key}: {value}\n"

        return response


class RefusalAnalyzer:
    """
    Analyzes patterns in refusals for system improvement.
    """

    def __init__(self):
        self.history: List[RefusalDecision] = []

    def record(self, decision: RefusalDecision):
        """Record a refusal decision."""
        self.history.append(decision)

    def get_statistics(self) -> Dict[str, Any]:
        """Get refusal statistics."""
        if not self.history:
            return {"total": 0}

        type_counts = {}
        for decision in self.history:
            type_name = decision.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        total = len(self.history)
        refusals = sum(1 for d in self.history if d.is_refusal())

        return {
            "total": total,
            "refusals": refusals,
            "refusal_rate": refusals / total if total > 0 else 0,
            "by_type": type_counts,
            "by_type_pct": {
                k: v / total for k, v in type_counts.items()
            }
        }

    def find_patterns(self) -> Dict[str, Any]:
        """Analyze common patterns in refusals."""
        patterns = {
            "common_reasons": {},
            "entropy_distribution": [],
            "policy_score_distribution": []
        }

        for decision in self.history:
            # Track reasons
            reason = decision.reason
            patterns["common_reasons"][reason] = \
                patterns["common_reasons"].get(reason, 0) + 1

            # Track entropy values
            if "entropy" in decision.details:
                patterns["entropy_distribution"].append(decision.details["entropy"])

            # Track policy scores
            if "policy_score" in decision.details:
                patterns["policy_score_distribution"].append(
                    decision.details["policy_score"]
                )

        # Sort reasons by frequency
        patterns["common_reasons"] = dict(
            sorted(
                patterns["common_reasons"].items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        return patterns
