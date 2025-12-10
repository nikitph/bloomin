"""
Rewa Core - Main Orchestration

The central engine that ties together all components:
1. Embedding & Normalization (SemanticSpace)
2. Hemisphere Check (HemisphereChecker)
3. Hull Construction (SphericalHull)
4. Entropy Estimation (EntropyEstimator)
5. Mode B Policy Injection (ModeBEngine)
6. Refusal Handling (RefusalHandler)
7. Verbalization Guard (VerbalizationGuard)
8. Audit Logging (AuditLogger)

Primary Goal: Zero hallucinated approvals under ambiguous or contradictory evidence.
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .semantic_space import SemanticSpace, Witness
from .hemisphere import HemisphereChecker, HemisphereResult
from .hull import SphericalHull, SphericalHullResult
from .entropy import EntropyEstimator, EntropyResult, RewaState
from .policy import Policy, PolicyEngine, PolicySpec, RiskPosture
from .mode_b import ModeBEngine, ModeBResult
from .refusal import RefusalHandler, RefusalDecision, RefusalType
from .verbalization import VerbalizationGuard, VerbalizationResult, VerbalizationStatus
from .audit import AuditLogger, AuditEntry


@dataclass
class RewaDecision:
    """Complete decision output from Rewa Core."""
    # Core decision
    approved: bool
    selected_meaning: Optional[np.ndarray]
    state: RewaState

    # Refusal info (if not approved)
    refusal: Optional[RefusalDecision]

    # Policy info
    policy_applied: bool
    policy_id: Optional[str]
    policy_score: Optional[float]

    # Geometry info
    hemisphere_exists: bool
    entropy: float
    hull_radius: float

    # Audit
    audit_entry_id: str
    computation_time_ms: float

    # Full details
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approved": self.approved,
            "selected_meaning": self.selected_meaning.tolist() if self.selected_meaning is not None else None,
            "state": self.state.value,
            "refusal_type": self.refusal.type.value if self.refusal else None,
            "refusal_reason": self.refusal.reason if self.refusal else None,
            "policy_applied": self.policy_applied,
            "policy_id": self.policy_id,
            "policy_score": self.policy_score,
            "hemisphere_exists": self.hemisphere_exists,
            "entropy": self.entropy,
            "hull_radius": self.hull_radius,
            "audit_entry_id": self.audit_entry_id,
            "computation_time_ms": self.computation_time_ms
        }


class RewaCore:
    """
    Rewa Core v1 - Policy-Driven Semantic Validation Engine

    A middleware system that:
    1. Ingests documents (evidence) and policies (biases)
    2. Projects both into a shared semantic sphere
    3. Computes the admissible semantic region from evidence
    4. Applies explicit, logged policy bias (Mode B) when ambiguity exists
    5. Generates or refuses output without hallucination

    Primary Goal: Zero hallucinated approvals under ambiguous or contradictory evidence.
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        entropy_threshold: float = 0.3,
        policy_threshold: float = 0.5,
        drift_threshold: float = 0.2,
        min_witnesses: int = 2,
        log_dir: str = "logs",
        enable_audit: bool = True
    ):
        """
        Initialize Rewa Core.

        Args:
            model_name: Sentence transformer model for embeddings
            entropy_threshold: Above this, state is AMBIGUOUS
            policy_threshold: Minimum policy score for approval
            drift_threshold: Maximum verbalization drift allowed
            min_witnesses: Minimum witnesses for VALID state
            log_dir: Directory for audit logs
            enable_audit: Whether to enable audit logging
        """
        # Core components
        self.semantic_space = SemanticSpace(model_name)
        self.hemisphere_checker = HemisphereChecker()
        self.hull_computer = SphericalHull()
        self.entropy_estimator = EntropyEstimator(
            entropy_threshold=entropy_threshold,
            min_witnesses_for_valid=min_witnesses
        )
        self.policy_engine = PolicyEngine(self.semantic_space)
        self.mode_b_engine = ModeBEngine(
            self.policy_engine,
            self.hull_computer
        )
        self.refusal_handler = RefusalHandler(
            policy_threshold=policy_threshold,
            entropy_threshold=entropy_threshold,
            require_min_witnesses=min_witnesses
        )
        self.verbalization_guard = VerbalizationGuard(
            self.semantic_space,
            drift_threshold=drift_threshold
        )
        self.audit_logger = AuditLogger(
            log_dir=log_dir,
            enable_file_logging=enable_audit,
            enable_memory_logging=True
        )

        # Configuration
        self.entropy_threshold = entropy_threshold
        self.policy_threshold = policy_threshold

    def process(
        self,
        documents: List[str],
        policy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RewaDecision:
        """
        Process documents and produce a validated decision.

        This is the main entry point for Rewa Core.

        Args:
            documents: List of evidence documents (text)
            policy_id: Optional policy to apply for Mode B
            metadata: Optional metadata to include in audit

        Returns:
            RewaDecision with selected meaning or refusal
        """
        start_time = time.time()

        # Step 1: Create witnesses from documents
        witnesses = self.semantic_space.create_witnesses(documents, source="input")

        # Step 2: Check hemisphere (contradiction detection)
        hemisphere_result = self.hemisphere_checker.check(witnesses)

        # Step 3: Compute hull (if hemisphere exists)
        hull_result = None
        if hemisphere_result.exists:
            hull_result = self.hull_computer.compute(witnesses)

        # Step 4: Estimate entropy
        entropy_result = self.entropy_estimator.estimate(
            witnesses,
            hemisphere_exists=hemisphere_result.exists
        )

        # Step 5: Determine if Mode B needed
        mode_b_result = None
        policy_score = None

        if entropy_result.state == RewaState.AMBIGUOUS and policy_id:
            # Mode B: Apply policy to select from ambiguous region
            mode_b_result = self.mode_b_engine.select_meaning(
                witnesses, policy_id, entropy_result.state
            )
            policy_score = mode_b_result.policy_score

        elif entropy_result.state == RewaState.VALID and policy_id:
            # Not ambiguous, but still compute policy score for threshold check
            policy = self.policy_engine.get(policy_id)
            if policy and hull_result:
                policy_score = policy.score(hull_result.center)

        # Step 6: Evaluate refusal
        refusal_decision = self.refusal_handler.evaluate(
            witnesses=witnesses,
            hemisphere_exists=hemisphere_result.exists,
            entropy=entropy_result.entropy,
            policy_score=policy_score,
            state=entropy_result.state
        )

        # Step 7: Determine final outcome
        approved = not refusal_decision.is_refusal()

        # Select final meaning
        selected_meaning = None
        if approved:
            if mode_b_result:
                selected_meaning = mode_b_result.selected_meaning
            elif hull_result:
                selected_meaning = hull_result.center

        # Compute time
        computation_time_ms = (time.time() - start_time) * 1000

        # Step 8: Audit log
        audit_entry = self.audit_logger.log(
            witnesses=witnesses,
            hemisphere_result=hemisphere_result,
            hull_result=hull_result,
            entropy_result=entropy_result,
            policy_id=policy_id,
            policy_score=policy_score,
            mode_b_result=mode_b_result,
            refusal_decision=refusal_decision,
            verbalization_result=None,
            computation_time_ms=computation_time_ms,
            metadata=metadata
        )

        return RewaDecision(
            approved=approved,
            selected_meaning=selected_meaning,
            state=entropy_result.state,
            refusal=refusal_decision if refusal_decision.is_refusal() else None,
            policy_applied=mode_b_result is not None,
            policy_id=policy_id,
            policy_score=policy_score,
            hemisphere_exists=hemisphere_result.exists,
            entropy=entropy_result.entropy,
            hull_radius=hull_result.angular_radius if hull_result else 0.0,
            audit_entry_id=audit_entry.entry_id,
            computation_time_ms=computation_time_ms,
            details={
                "hemisphere": {
                    "exists": hemisphere_result.exists,
                    "margin": hemisphere_result.margin,
                    "contradicting_pairs": hemisphere_result.contradicting_pairs
                },
                "entropy": {
                    "value": entropy_result.entropy,
                    "threshold": entropy_result.threshold_used,
                    "state": entropy_result.state.value
                }
            }
        )

    def process_with_verbalization(
        self,
        documents: List[str],
        policy_id: Optional[str],
        generate_fn,  # Callable that takes meaning and returns text
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[RewaDecision, Optional[str], Optional[VerbalizationResult]]:
        """
        Process documents and verify LLM verbalization.

        Args:
            documents: Evidence documents
            policy_id: Policy to apply
            generate_fn: Function that generates text from meaning
            metadata: Optional metadata

        Returns:
            (decision, generated_text, verbalization_result)
        """
        # First, get the decision
        decision = self.process(documents, policy_id, metadata)

        if not decision.approved:
            return decision, None, None

        # Generate text
        generated_text = generate_fn(decision.selected_meaning)

        # Verify verbalization
        witnesses = self.semantic_space.create_witnesses(documents)
        verbalization_result = self.verbalization_guard.verify(
            decision.selected_meaning,
            generated_text,
            witnesses
        )

        # Update decision if verbalization failed
        if verbalization_result.status != VerbalizationStatus.VERIFIED:
            decision.approved = False
            decision.refusal = RefusalDecision(
                type=RefusalType.POLICY_EXCLUDED,
                reason="Verbalization drift exceeded threshold",
                explanation=f"Generated text drifted {np.degrees(verbalization_result.drift_distance):.1f}° from intended meaning",
                reproducible_hash="",
                details={"drift": verbalization_result.drift_distance}
            )
            return decision, None, verbalization_result

        return decision, generated_text, verbalization_result

    def register_policy(
        self,
        name: str,
        description: str,
        rules: List[str] = None,
        prototypes: List[str] = None,
        antiprototypes: List[str] = None,
        risk_posture: RiskPosture = RiskPosture.MODERATE,
        threshold: float = 0.5
    ) -> str:
        """
        Register a new policy.

        Args:
            name: Policy name
            description: Policy description
            rules: Natural language rules
            prototypes: Positive examples
            antiprototypes: Negative examples
            risk_posture: Risk tolerance level
            threshold: Minimum score for approval

        Returns:
            Policy ID
        """
        spec = PolicySpec(
            id="",  # Will be auto-generated
            name=name,
            description=description,
            rules=rules or [],
            prototypes=prototypes or [],
            antiprototypes=antiprototypes or [],
            risk_posture=risk_posture,
            threshold=threshold
        )

        policy = self.policy_engine.compile(spec)
        return policy.spec.id

    def get_audit_report(self) -> str:
        """Generate compliance report from audit log."""
        return self.audit_logger.generate_compliance_report()

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        return self.audit_logger.get_statistics()

    def verify_decision_reproducibility(
        self,
        documents: List[str],
        original_entry_id: str
    ) -> Dict[str, Any]:
        """
        Verify that a decision can be reproduced.

        Args:
            documents: Original documents
            original_entry_id: Audit entry ID to verify against

        Returns:
            Verification result
        """
        # Get original entry
        original = self.audit_logger.get_entry(original_entry_id)
        if not original:
            return {"error": "Entry not found", "reproducible": False}

        # Recompute
        witnesses = self.semantic_space.create_witnesses(documents)
        embeddings_concat = np.concatenate([w.embedding for w in witnesses])
        import hashlib
        current_hash = hashlib.sha256(embeddings_concat.tobytes()).hexdigest()[:16]

        return self.audit_logger.verify_reproducibility(original, current_hash)


# Export key classes from submodules
__all__ = [
    'RewaCore',
    'RewaDecision',
    'RewaState',
    'Policy',
    'PolicySpec',
    'RiskPosture',
    'RefusalType',
    'RefusalDecision',
    'Witness'
]
