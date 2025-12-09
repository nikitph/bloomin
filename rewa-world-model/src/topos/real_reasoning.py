"""
Real Reasoning Layer for REWA World Model.

This module implements data-driven reasoning using the geometric and topos machinery:
- Query feasibility via Fisher geometry (not hardcoded rules)
- Modifier detection via topos section comparison (not string matching)
- Inference via semantic RG multiscale analysis
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .logic import ToposLogic, LocalSection, Proposition

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Standardized result from reasoning operations."""
    status: str  # 'success', 'contradiction', 'impossible', 'modified', 'no_inference'
    derived_facts: List[str]
    explanation: str
    confidence: float
    affected_entities: Optional[Set[str]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ModifierResult:
    """Result from modifier detection."""
    has_conflict: bool
    conflicting_properties: List[str]
    explanation: str
    confidence: float


@dataclass
class DocumentMetric:
    """Stores geometric information for a document."""
    doc_id: str
    witness_distribution: Dict[str, float]
    signature: np.ndarray  # REWA signature
    embedding: Optional[np.ndarray] = None


class RealReasoningLayer:
    """
    Data-driven reasoning layer using geometric and topos machinery.

    Replaces hardcoded rules with:
    - Fisher geometry for feasibility checks
    - Topos gluing for consistency/contradiction detection
    - Semantic RG for multiscale inference
    """

    def __init__(
        self,
        topos_logic: Optional[ToposLogic] = None,
        feasibility_threshold: float = 10.0,  # Increased - KL can be large for text
        consistency_threshold: float = 0.3,
        transitivity_threshold: float = 0.5,
        modifier_shift_threshold: float = 3.0  # Threshold for significant distribution shift
    ):
        self.topos = topos_logic if topos_logic else ToposLogic()
        self.feasibility_threshold = feasibility_threshold
        self.consistency_threshold = consistency_threshold
        self.transitivity_threshold = transitivity_threshold
        self.modifier_shift_threshold = modifier_shift_threshold

        # Index of known documents for feasibility checks
        self.document_index: Dict[str, DocumentMetric] = {}

    def index_document(
        self,
        doc_id: str,
        witness_distribution: Dict[str, float],
        signature: np.ndarray,
        embedding: Optional[np.ndarray] = None
    ):
        """Add a document to the index for feasibility checks."""
        self.document_index[doc_id] = DocumentMetric(
            doc_id=doc_id,
            witness_distribution=witness_distribution,
            signature=signature,
            embedding=embedding
        )

    def compute_distribution_distance(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> float:
        """
        Compute symmetric KL divergence between two distributions.
        This approximates Fisher-Rao distance for small differences.
        """
        all_keys = set(dist1.keys()) | set(dist2.keys())

        kl_forward = 0.0
        kl_backward = 0.0

        for key in all_keys:
            p = dist1.get(key, 1e-10)
            q = dist2.get(key, 1e-10)

            # Ensure positive
            p = max(p, 1e-10)
            q = max(q, 1e-10)

            kl_forward += p * np.log(p / q)
            kl_backward += q * np.log(q / p)

        # Symmetric KL (Jensen-Shannon style)
        return (kl_forward + kl_backward) / 2

    def compute_signature_distance(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray
    ) -> float:
        """Compute normalized Hamming distance between signatures."""
        if len(sig1) == 0 or len(sig2) == 0:
            return 1.0

        # Handle different types (bool vs float)
        s1 = sig1.astype(bool)
        s2 = sig2.astype(bool)

        hamming = np.sum(s1 != s2)
        return hamming / len(sig1)

    def check_query_feasibility(
        self,
        query_distribution: Dict[str, float],
        query_signature: Optional[np.ndarray] = None
    ) -> ReasoningResult:
        """
        Check if a query is feasible by measuring its geometric distance
        to the indexed document manifold.

        A query is infeasible if it's too far from any known document,
        meaning no documents support the query's witness distribution.
        """
        if len(self.document_index) == 0:
            # No index, assume feasible
            return ReasoningResult(
                status="success",
                derived_facts=["No document index - assuming feasible"],
                explanation="Document index is empty, cannot verify feasibility.",
                confidence=0.5
            )

        # Find minimum distance to any indexed document
        min_dist_kl = float('inf')
        min_dist_sig = float('inf')
        nearest_doc = None

        for doc_id, doc_metric in self.document_index.items():
            # Distribution distance (KL-based)
            dist_kl = self.compute_distribution_distance(
                query_distribution,
                doc_metric.witness_distribution
            )

            if dist_kl < min_dist_kl:
                min_dist_kl = dist_kl
                nearest_doc = doc_id

            # Signature distance (if available)
            if query_signature is not None and doc_metric.signature is not None:
                dist_sig = self.compute_signature_distance(
                    query_signature,
                    doc_metric.signature
                )
                min_dist_sig = min(min_dist_sig, dist_sig)

        # Combine distances
        combined_distance = min_dist_kl
        if min_dist_sig < float('inf'):
            # Weight signature distance less (it's coarser)
            combined_distance = 0.7 * min_dist_kl + 0.3 * min_dist_sig * 10  # scale sig dist

        # Check against threshold
        if combined_distance > self.feasibility_threshold:
            # Find which witnesses are unsupported
            unsupported = self._find_unsupported_witnesses(query_distribution)

            return ReasoningResult(
                status="impossible",
                derived_facts=[],
                explanation=f"Query is geometrically isolated (distance={combined_distance:.3f}). "
                           f"Unsupported witnesses: {unsupported[:5]}",  # limit to 5
                confidence=1.0 - (1.0 / (1.0 + combined_distance)),
                details={
                    "distance": combined_distance,
                    "nearest_doc": nearest_doc,
                    "unsupported_witnesses": unsupported
                }
            )

        return ReasoningResult(
            status="success",
            derived_facts=[f"Query feasible, nearest doc: {nearest_doc}"],
            explanation=f"Query is within feasibility threshold (distance={combined_distance:.3f})",
            confidence=1.0 / (1.0 + combined_distance),
            details={
                "distance": combined_distance,
                "nearest_doc": nearest_doc
            }
        )

    def _find_unsupported_witnesses(
        self,
        query_distribution: Dict[str, float]
    ) -> List[str]:
        """Find witnesses in query that have no support in any indexed document."""
        unsupported = []

        for witness, prob in query_distribution.items():
            if prob < 0.01:  # Skip low-probability witnesses
                continue

            # Check if any document has this witness
            has_support = False
            for doc_metric in self.document_index.values():
                if witness in doc_metric.witness_distribution:
                    if doc_metric.witness_distribution[witness] > 0.01:
                        has_support = True
                        break

            if not has_support:
                unsupported.append(witness)

        return unsupported

    def detect_modifier_effect(
        self,
        base_distribution: Dict[str, float],
        modified_distribution: Dict[str, float],
        base_section: Optional[LocalSection] = None,
        modified_section: Optional[LocalSection] = None
    ) -> ModifierResult:
        """
        Detect if a modifier changes the semantic properties of a concept.

        Uses topos section comparison to find property divergences,
        instead of hardcoded modifier word lists.
        """
        # Build sections if not provided
        if base_section is None:
            base_section = self.topos.build_section('base', base_distribution)
        if modified_section is None:
            modified_section = self.topos.build_section('modified', modified_distribution)

        # Check gluing consistency
        consistent, conflicts = self.topos.check_gluing_consistency(
            base_section, modified_section
        )

        if not consistent:
            # Analyze the nature of the conflicts
            conflict_details = self._analyze_conflicts(
                base_section, modified_section, conflicts
            )

            return ModifierResult(
                has_conflict=True,
                conflicting_properties=conflicts,
                explanation=f"Modifier negates properties: {conflict_details}",
                confidence=1.0 - len(conflicts) * 0.1  # reduce confidence per conflict
            )

        # Even if consistent, check for significant distribution shift
        dist_shift = self.compute_distribution_distance(
            base_distribution, modified_distribution
        )

        if dist_shift > self.modifier_shift_threshold:  # Significant shift
            # Find which witnesses changed most
            changed = self._find_changed_witnesses(
                base_distribution, modified_distribution
            )

            return ModifierResult(
                has_conflict=True,
                conflicting_properties=changed[:5],
                explanation=f"Significant semantic shift (distance={dist_shift:.3f})",
                confidence=0.7
            )

        return ModifierResult(
            has_conflict=False,
            conflicting_properties=[],
            explanation="No significant modifier effect detected",
            confidence=1.0
        )

    def _analyze_conflicts(
        self,
        section1: LocalSection,
        section2: LocalSection,
        conflicts: List[str]
    ) -> str:
        """Provide detailed analysis of conflicting properties."""
        details = []

        props1 = {p.predicate: p.confidence for p in section1.propositions}
        props2 = {p.predicate: p.confidence for p in section2.propositions}

        for conflict in conflicts[:3]:  # Limit to 3
            conf1 = props1.get(conflict, 0.0)
            conf2 = props2.get(conflict, 0.0)
            details.append(f"{conflict}: {conf1:.2f} vs {conf2:.2f}")

        return "; ".join(details)

    def _find_changed_witnesses(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> List[str]:
        """Find witnesses with largest probability changes."""
        changes = []

        all_keys = set(dist1.keys()) | set(dist2.keys())
        for key in all_keys:
            p1 = dist1.get(key, 0.0)
            p2 = dist2.get(key, 0.0)
            change = abs(p1 - p2)
            if change > 0.05:  # Significant change
                changes.append((key, change))

        # Sort by change magnitude
        changes.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in changes]

    def infer_from_sections(
        self,
        sections: List[LocalSection]
    ) -> ReasoningResult:
        """
        Perform inference by attempting to glue local sections.

        If gluing succeeds, we derive global propositions.
        If it fails, we identify contradictions.
        """
        if len(sections) == 0:
            return ReasoningResult(
                status="no_inference",
                derived_facts=[],
                explanation="No sections to infer from",
                confidence=0.0
            )

        # Attempt to glue all sections
        global_props = self.topos.glue_sections(sections)

        if global_props is None:
            # Find specific contradictions
            contradictions = []
            for i in range(len(sections)):
                for j in range(i + 1, len(sections)):
                    consistent, conflicts = self.topos.check_gluing_consistency(
                        sections[i], sections[j]
                    )
                    if not consistent:
                        contradictions.append({
                            "sections": (sections[i].region_id, sections[j].region_id),
                            "conflicts": conflicts
                        })

            return ReasoningResult(
                status="contradiction",
                derived_facts=[],
                explanation=f"Gluing failed: {len(contradictions)} contradictions found",
                confidence=1.0,
                details={"contradictions": contradictions}
            )

        # Successful gluing - extract derived facts
        derived = []
        for prop in global_props:
            if prop.confidence >= self.topos.confidence_threshold:
                derived.append(f"{prop.predicate}={prop.confidence:.2f}")

        return ReasoningResult(
            status="success",
            derived_facts=derived,
            explanation=f"Successfully derived {len(derived)} global propositions",
            confidence=min([p.confidence for p in global_props]) if global_props else 1.0,
            details={"global_propositions": [
                {"predicate": p.predicate, "confidence": p.confidence}
                for p in global_props
            ]}
        )

    def check_transitive_relation(
        self,
        section_a: LocalSection,
        section_b: LocalSection,
        section_c: LocalSection
    ) -> ReasoningResult:
        """
        Check if a transitive relation holds: if A->B and B->C, then A->C.

        Uses mutual information between sections to determine relatedness.
        """
        # Compute pairwise MI (using distribution overlap as proxy)
        def section_to_dist(s: LocalSection) -> Dict[str, float]:
            return {p.predicate: p.confidence for p in s.propositions}

        dist_a = section_to_dist(section_a)
        dist_b = section_to_dist(section_b)
        dist_c = section_to_dist(section_c)

        # MI approximation: overlap / total
        def approx_mi(d1: Dict[str, float], d2: Dict[str, float]) -> float:
            overlap = set(d1.keys()) & set(d2.keys())
            if not overlap:
                return 0.0

            overlap_mass = sum(min(d1[k], d2[k]) for k in overlap)
            total_mass = sum(d1.values()) + sum(d2.values())

            if total_mass == 0:
                return 0.0
            return 2 * overlap_mass / total_mass

        mi_ab = approx_mi(dist_a, dist_b)
        mi_bc = approx_mi(dist_b, dist_c)
        mi_ac = approx_mi(dist_a, dist_c)

        # Transitivity holds if: mi_ab > 0 AND mi_bc > 0 implies mi_ac > 0
        if mi_ab > self.transitivity_threshold and mi_bc > self.transitivity_threshold:
            if mi_ac > self.transitivity_threshold * 0.5:  # Weaker threshold for derived
                return ReasoningResult(
                    status="success",
                    derived_facts=[f"Transitive relation: {section_a.region_id} -> {section_c.region_id}"],
                    explanation=f"MI(A,B)={mi_ab:.2f}, MI(B,C)={mi_bc:.2f} => MI(A,C)={mi_ac:.2f}",
                    confidence=min(mi_ab, mi_bc, mi_ac),
                    details={
                        "mi_ab": mi_ab,
                        "mi_bc": mi_bc,
                        "mi_ac": mi_ac
                    }
                )
            else:
                return ReasoningResult(
                    status="modified",
                    derived_facts=[],
                    explanation=f"Weak transitive relation: MI(A,C)={mi_ac:.2f} below threshold",
                    confidence=mi_ac,
                    details={
                        "mi_ab": mi_ab,
                        "mi_bc": mi_bc,
                        "mi_ac": mi_ac
                    }
                )

        return ReasoningResult(
            status="no_inference",
            derived_facts=[],
            explanation="Insufficient relation strength for transitivity",
            confidence=0.0,
            details={
                "mi_ab": mi_ab,
                "mi_bc": mi_bc,
                "mi_ac": mi_ac
            }
        )