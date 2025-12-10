"""
Income Admissibility Evaluator

Evaluates income admissibility under Freddie Mac guidelines 5301-5303.

Query: "Is the borrower's income admissible under Freddie Mac guidelines?"

Output:
- APPROVE: Income meets all requirements
- REFUSE: Income does not meet requirements (with reason)
- REQUEST_MORE_INFO: Insufficient documentation to determine

Uses Rewa Core for semantic validation.
"""

import sys
import os
# Add parent rewa_core directory to path
rewa_core_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, rewa_core_dir)

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from src.rewa_space_v2 import RewaSpaceV2, generate_negation_pairs
from freddie_validation.src.policy_corpus import FreddieMacPolicyCorpus, ClauseType
from freddie_validation.src.loan_evidence import LoanFile, LoanEvidenceProcessor


class Decision(Enum):
    """Evaluation decision."""
    APPROVE = "approve"
    REFUSE = "refuse"
    REQUEST_MORE_INFO = "request_more_info"


@dataclass
class EvaluationResult:
    """Result of income admissibility evaluation."""
    decision: Decision
    confidence: float
    reasons: List[str]
    policy_references: List[str]
    evidence_used: List[str]
    flags: List[Dict[str, Any]]

    # Audit fields
    loan_id: str
    policy_version: str
    evaluation_timestamp: str
    evidence_hash: str

    # Rewa metrics
    hemisphere_exists: bool
    entropy: float
    mode_b_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "policy_references": self.policy_references,
            "evidence_used": self.evidence_used,
            "flags": self.flags,
            "loan_id": self.loan_id,
            "policy_version": self.policy_version,
            "evaluation_timestamp": self.evaluation_timestamp,
            "evidence_hash": self.evidence_hash,
            "hemisphere_exists": self.hemisphere_exists,
            "entropy": self.entropy,
            "mode_b_applied": self.mode_b_applied
        }


class IncomeAdmissibilityEvaluator:
    """
    Evaluates income admissibility using Rewa Core + Freddie Mac policies.

    Architecture:
    1. Load policy corpus (bias)
    2. Process loan evidence (witnesses)
    3. Project to Rewa-space
    4. Check hemisphere (contradiction detection)
    5. Evaluate against policy clauses
    6. Make decision with full audit trail
    """

    def __init__(self, train_rewa_space: bool = True):
        """
        Initialize evaluator.

        Args:
            train_rewa_space: Whether to train Rewa-space projection
        """
        # Load policy corpus
        self.policy_corpus = FreddieMacPolicyCorpus()
        self.policy_version = self.policy_corpus.load_income_policies()

        # Initialize Rewa-space
        self.rewa_space = RewaSpaceV2(output_dim=384)
        if train_rewa_space:
            training_pairs = generate_negation_pairs()
            # Add policy-specific pairs
            training_pairs.extend(self._generate_policy_training_pairs())
            self.rewa_space.train(training_pairs, epochs=200, verbose=False)

        # Evidence processor
        self.evidence_processor = LoanEvidenceProcessor()

        # Pre-embed policy clauses
        self._embed_policies()

    def _generate_policy_training_pairs(self) -> List[Tuple[str, str]]:
        """Generate training pairs from policy clauses."""
        pairs = [
            # Income stability pairs
            ("Income is stable and consistent", "Income is unstable and inconsistent"),
            ("Income shows increasing trend", "Income shows declining trend"),
            ("Income has been verified", "Income has not been verified"),
            ("Documentation is complete", "Documentation is incomplete"),

            # Employment pairs
            ("Borrower has continuous employment", "Borrower has employment gaps"),
            ("Employment history is verified", "Employment cannot be verified"),
            ("Currently employed", "Currently unemployed"),
            ("Same employer for 2+ years", "Multiple job changes"),

            # Self-employment pairs
            ("Business is profitable", "Business is operating at a loss"),
            ("Self-employed for 2+ years", "Self-employed for less than 2 years"),
            ("Business has adequate liquidity", "Business lacks adequate liquidity"),

            # Compliance pairs
            ("Meets Freddie Mac requirements", "Does not meet Freddie Mac requirements"),
            ("Income is admissible", "Income is not admissible"),
            ("Loan is conforming", "Loan is non-conforming"),
            ("Approved for purchase", "Subject to repurchase"),

            # Documentation pairs
            ("Tax returns provided", "Tax returns missing"),
            ("W-2 forms verified", "W-2 forms not provided"),
            ("Bank statements confirm income", "Bank statements contradict income"),
        ]
        return pairs

    def _embed_policies(self):
        """Pre-embed all policy clauses."""
        self.policy_embeddings = {}

        for clause in self.policy_version.clauses:
            embedding = self.rewa_space.project(clause.text)
            self.policy_embeddings[clause.id] = {
                "embedding": embedding,
                "clause": clause
            }

    def evaluate(self, loan_file: LoanFile) -> EvaluationResult:
        """
        Evaluate income admissibility for a loan file.

        Args:
            loan_file: Complete loan application package

        Returns:
            EvaluationResult with decision and audit trail
        """
        # Process evidence
        evidence = self.evidence_processor.process_loan_file(loan_file)
        evidence_statements = evidence["evidence_statements"]

        # Project evidence to Rewa-space
        evidence_embeddings = np.array([
            self.rewa_space.project(stmt) for stmt in evidence_statements
        ])

        # Check for contradictions (hemisphere test)
        hemisphere_exists, contradiction_details = self._check_hemisphere(evidence_embeddings)

        # Compute entropy
        entropy = self._compute_entropy(evidence_embeddings)

        # Get relevant policy clauses
        relevant_policies = self._find_relevant_policies(evidence_statements, evidence["flags"])

        # Make decision
        decision, confidence, reasons, policy_refs = self._make_decision(
            evidence=evidence,
            hemisphere_exists=hemisphere_exists,
            contradiction_details=contradiction_details,
            relevant_policies=relevant_policies,
            entropy=entropy
        )

        return EvaluationResult(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            policy_references=policy_refs,
            evidence_used=evidence_statements,
            flags=evidence["flags"],
            loan_id=loan_file.loan_id,
            policy_version=self.policy_version.version_id,
            evaluation_timestamp=datetime.now().isoformat(),
            evidence_hash=evidence["hash"],
            hemisphere_exists=hemisphere_exists,
            entropy=entropy,
            mode_b_applied=entropy > 0.3  # Mode B threshold
        )

    def _check_hemisphere(
        self,
        embeddings: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if evidence embeddings can exist in a common hemisphere.

        If not, evidence is contradictory.
        """
        if len(embeddings) < 2:
            return True, {"message": "Insufficient evidence for hemisphere check"}

        # Check all pairwise similarities
        n = len(embeddings)
        min_sim = float('inf')
        contradictions = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                if sim < min_sim:
                    min_sim = sim

                # If similarity is very negative, we have a contradiction
                if sim < -0.3:
                    contradictions.append({
                        "indices": (i, j),
                        "similarity": float(sim),
                        "angle_deg": float(np.degrees(np.arccos(np.clip(sim, -1, 1))))
                    })

        hemisphere_exists = len(contradictions) == 0

        return hemisphere_exists, {
            "min_similarity": float(min_sim),
            "contradictions": contradictions,
            "num_contradictions": len(contradictions)
        }

    def _compute_entropy(self, embeddings: np.ndarray) -> float:
        """Compute entropy/ambiguity of evidence."""
        if len(embeddings) < 2:
            return 1.0  # Maximum entropy for single evidence

        # Mean direction
        mean_dir = np.mean(embeddings, axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)

        # Angular variance from mean
        dots = embeddings @ mean_dir
        angles = np.arccos(np.clip(dots, -1, 1))
        variance = np.var(angles)

        # Normalize to [0, 1]
        entropy = float(np.clip(variance / (np.pi ** 2), 0, 1))

        return entropy

    def _find_relevant_policies(
        self,
        evidence_statements: List[str],
        flags: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find policy clauses relevant to the evidence and flags."""
        relevant = []

        # Always include mandatory clauses
        for clause in self.policy_corpus.get_mandatory_clauses():
            relevant.append({
                "clause": clause,
                "relevance": "mandatory",
                "score": 1.0
            })

        # Include clauses referenced by flags
        flag_refs = set(f["policy_reference"] for f in flags if "policy_reference" in f)
        for clause in self.policy_version.clauses:
            if clause.id in flag_refs:
                relevant.append({
                    "clause": clause,
                    "relevance": "flagged",
                    "score": 1.0
                })

        # Find semantically similar clauses
        for stmt in evidence_statements:
            stmt_emb = self.rewa_space.project(stmt)

            for clause_id, policy_data in self.policy_embeddings.items():
                sim = np.dot(stmt_emb, policy_data["embedding"])
                if sim > 0.5:  # Relevance threshold
                    # Check if already added
                    if not any(r["clause"].id == clause_id for r in relevant):
                        relevant.append({
                            "clause": policy_data["clause"],
                            "relevance": "semantic",
                            "score": float(sim)
                        })

        return relevant

    def _make_decision(
        self,
        evidence: Dict[str, Any],
        hemisphere_exists: bool,
        contradiction_details: Dict[str, Any],
        relevant_policies: List[Dict[str, Any]],
        entropy: float
    ) -> Tuple[Decision, float, List[str], List[str]]:
        """
        Make final decision based on evidence and policies.

        Returns:
            (decision, confidence, reasons, policy_references)
        """
        reasons = []
        policy_refs = []
        confidence = 0.0

        # Critical flags that force REFUSE
        critical_flags = [f for f in evidence["flags"] if f["severity"] == "critical"]
        high_flags = [f for f in evidence["flags"] if f["severity"] == "high"]

        # REFUSE: Contradictory evidence
        if not hemisphere_exists:
            reasons.append("Evidence contains contradictory information that cannot be reconciled.")
            reasons.append(f"Found {contradiction_details['num_contradictions']} contradiction(s).")
            return Decision.REFUSE, 0.95, reasons, ["5301.2"]

        # REFUSE: Critical issues
        if critical_flags:
            for flag in critical_flags:
                reasons.append(f"Critical issue: {flag['description']}")
                policy_refs.append(flag.get("policy_reference", ""))
            return Decision.REFUSE, 0.95, reasons, list(set(policy_refs))

        # REQUEST_MORE_INFO: Missing documentation
        doc_analysis = evidence["documentation"]
        if not doc_analysis["is_complete"]:
            reasons.append(f"Missing required documentation: {doc_analysis['missing_required']}")
            return Decision.REQUEST_MORE_INFO, 0.9, reasons, ["5301.2"]

        # REFUSE: High severity issues without remediation
        if high_flags:
            # Check if issues can be resolved
            resolvable = []
            unresolvable = []

            for flag in high_flags:
                if flag["type"] in ["DECLINING_INCOME", "INSUFFICIENT_INCOME_HISTORY",
                                   "INSUFFICIENT_SELF_EMPLOYMENT_HISTORY"]:
                    unresolvable.append(flag)
                else:
                    resolvable.append(flag)

            if unresolvable:
                for flag in unresolvable:
                    reasons.append(f"Policy violation: {flag['description']}")
                    policy_refs.append(flag.get("policy_reference", ""))
                return Decision.REFUSE, 0.85, reasons, list(set(policy_refs))

            if resolvable:
                for flag in resolvable:
                    reasons.append(f"Additional documentation required: {flag['description']}")
                    policy_refs.append(flag.get("policy_reference", ""))
                return Decision.REQUEST_MORE_INFO, 0.8, reasons, list(set(policy_refs))

        # Check income requirements
        income_analysis = evidence["income"]
        employment_analysis = evidence["employment"]

        # Insufficient income history
        if income_analysis.get("years_of_history", 0) < 2:
            reasons.append("Less than 2 years of income history documented.")
            return Decision.REFUSE, 0.9, reasons, ["5301.3"]

        # Declining income > 20%
        if income_analysis.get("trend") == "declining":
            decline = income_analysis.get("decline_percentage", 0)
            if decline > 20:
                reasons.append(f"Income declined {decline:.1f}% year-over-year without supporting documentation.")
                return Decision.REFUSE, 0.85, reasons, ["5301.4", "5303.4"]

        # Self-employment checks
        if employment_analysis.get("is_self_employed"):
            years = employment_analysis.get("years_in_current_role", 0)
            if years < 2:
                reasons.append(f"Self-employment history of {years:.1f} years is less than required 2 years.")
                return Decision.REFUSE, 0.9, reasons, ["5303.2"]

        # All checks passed - APPROVE
        reasons.append("Income documentation is complete and verified.")
        reasons.append("Income history demonstrates stability over required period.")
        reasons.append("All Freddie Mac income requirements are satisfied.")

        # Confidence based on entropy (lower entropy = higher confidence)
        confidence = max(0.7, 1.0 - entropy)

        return Decision.APPROVE, confidence, reasons, ["5301.1", "5301.2", "5301.3"]


class BaselineRAGEvaluator:
    """
    Baseline RAG + LLM evaluator for comparison.

    Simulates typical RAG behavior:
    - Retrieves relevant policy text
    - Uses LLM-like reasoning (simplified)
    - Tends to over-approve ambiguous cases
    """

    def __init__(self):
        self.policy_corpus = FreddieMacPolicyCorpus()
        self.policy_corpus.load_income_policies()
        self.evidence_processor = LoanEvidenceProcessor()

    def evaluate(self, loan_file: LoanFile) -> Dict[str, Any]:
        """
        Evaluate using baseline RAG approach.

        This simulates what a typical RAG + LLM system would do:
        - Retrieve policy snippets
        - Apply them without geometric constraints
        - Tend to find justifications for approval
        """
        evidence = self.evidence_processor.process_loan_file(loan_file)

        # RAG-style: just check if we have documents
        has_docs = len(loan_file.documents) > 0
        has_income = evidence["income"].get("has_income_data", False)
        has_employment = evidence["employment"].get("has_employment_data", False)

        # Baseline tends to approve if basic requirements seem met
        # (This simulates over-approval tendency)

        if not has_docs:
            decision = "refuse"
            reason = "No documentation provided"
        elif not has_income:
            decision = "refuse"
            reason = "No income data found"
        elif not has_employment:
            decision = "refuse"
            reason = "No employment data found"
        else:
            # Baseline often approves ambiguous cases
            # Check for obvious issues
            flags = evidence["flags"]
            critical = [f for f in flags if f["severity"] == "critical"]

            if critical:
                decision = "refuse"
                reason = critical[0]["description"]
            else:
                # Baseline tends to approve if income exists
                # Even with declining income or gaps
                decision = "approve"
                reason = "Income and employment documentation provided"

        return {
            "decision": decision,
            "reason": reason,
            "confidence": 0.7,  # Lower confidence
            "method": "baseline_rag"
        }
