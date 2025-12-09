"""
Real Reasoning Filter for REWA World Model.

This module implements data-driven context verification using:
- Witness extraction for semantic representation
- REWA signatures for fast similarity
- Topos gluing for consistency checking
- Semantic RG for multiscale coherence analysis
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from witnesses import Witness, WitnessType, WitnessExtractor, estimate_witness_distribution
from encoding import REWAConfig, REWAEncoder, hamming_distance
from topos import ToposLogic, LocalSection, Proposition
from topos.real_reasoning import RealReasoningLayer, ReasoningResult, ModifierResult
from topos.section_extractor import SectionExtractor
from .models import VerificationResult, ChunkVerification

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for the reasoning filter."""
    # REWA encoding
    rewa_num_positions: int = 256
    rewa_num_hashes: int = 4
    rewa_delta_gap: float = 0.1

    # Thresholds
    consistency_threshold: float = 0.3
    feasibility_threshold: float = 10.0  # Increased - KL between texts can be large
    signature_similarity_threshold: float = 0.5  # Lowered - signatures won't match exactly
    semantic_drift_threshold: float = 5.0  # Increased - pairwise chunk drift threshold
    modifier_shift_threshold: float = 3.0  # When to flag as modifier effect

    # Witness extraction
    witness_types: List[WitnessType] = None

    def __post_init__(self):
        if self.witness_types is None:
            self.witness_types = [WitnessType.BOOLEAN, WitnessType.NATURAL]


class RealReasoningFilter:
    """
    Data-driven middleware that uses geometric and topos machinery
    to verify RAG context consistency.

    Replaces string matching with:
    - Witness-based semantic representation
    - REWA signature similarity
    - Topos gluing consistency
    - Distribution-based modifier detection
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()

        # Initialize components
        self.extractor = WitnessExtractor(self.config.witness_types)

        self.encoder = REWAEncoder(REWAConfig(
            input_dim=1000,  # Will be adjusted dynamically
            num_positions=self.config.rewa_num_positions,
            num_hashes=self.config.rewa_num_hashes,
            delta_gap=self.config.rewa_delta_gap
        ))

        self.topos = ToposLogic(
            confidence_threshold=self.config.consistency_threshold
        )

        self.reasoning = RealReasoningLayer(
            topos_logic=self.topos,
            feasibility_threshold=self.config.feasibility_threshold,
            consistency_threshold=self.config.consistency_threshold,
            modifier_shift_threshold=self.config.modifier_shift_threshold
        )

        # Section extractor for compositional reasoning
        self.section_extractor = SectionExtractor()

    def _extract_and_encode(
        self,
        text: str,
        doc_id: str
    ) -> Tuple[List[Witness], Dict[str, float], np.ndarray]:
        """Extract witnesses, compute distribution, and encode signature."""
        doc = {'id': doc_id, 'text': text}
        witnesses = self.extractor.extract(doc)
        distribution = estimate_witness_distribution(witnesses)
        signature = self.encoder.encode(witnesses)
        return witnesses, distribution, signature

    def _build_section(
        self,
        region_id: str,
        distribution: Dict[str, float]
    ) -> LocalSection:
        """Build a topos section from a witness distribution."""
        return self.topos.build_section(region_id, distribution)

    def _compute_signature_similarity(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray
    ) -> float:
        """Compute similarity score from signatures (1.0 = identical)."""
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0

        # Hamming distance normalized
        distance = hamming_distance(sig1.astype(bool), sig2.astype(bool))
        max_dist = len(sig1)

        return 1.0 - (distance / max_dist)

    def _compute_distribution_similarity(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> float:
        """Compute similarity from KL divergence (1.0 = identical)."""
        kl_dist = self.reasoning.compute_distribution_distance(dist1, dist2)
        return 1.0 / (1.0 + kl_dist)

    def verify_context(
        self,
        query: str,
        chunks: List[str]
    ) -> VerificationResult:
        """
        Verify that retrieved chunks are consistent with query and each other.

        Uses real geometric/topos machinery instead of string matching.

        Args:
            query: User's natural language query
            chunks: List of text chunks from retrieval

        Returns:
            VerificationResult with detailed analysis
        """
        # 1. Process query
        query_witnesses, query_dist, query_sig = self._extract_and_encode(query, 'query')
        query_section = self._build_section('query', query_dist)

        # 2. Check query feasibility (if we have indexed documents)
        feasibility = self.reasoning.check_query_feasibility(query_dist, query_sig)

        if feasibility.status == "impossible":
            return VerificationResult(
                query=query,
                verified_chunks=[],
                overall_confidence=0.0,
                global_warnings=[
                    f"INFEASIBLE QUERY: {feasibility.explanation}"
                ]
            )

        # 3. Process each chunk
        chunk_data = []  # (witnesses, distribution, signature, section)
        chunk_verifications = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f'chunk_{i}'

            # Extract and encode
            c_witnesses, c_dist, c_sig = self._extract_and_encode(chunk_text, chunk_id)
            c_section = self._build_section(chunk_id, c_dist)

            chunk_data.append((c_witnesses, c_dist, c_sig, c_section))

            # A. Signature similarity to query
            sig_similarity = self._compute_signature_similarity(query_sig, c_sig)

            # B. Distribution similarity to query
            dist_similarity = self._compute_distribution_similarity(query_dist, c_dist)

            # C. Topos consistency with query
            consistent, conflicts = self.topos.check_gluing_consistency(
                query_section, c_section
            )

            # D. Modifier effect detection
            modifier_result = self.reasoning.detect_modifier_effect(
                query_dist, c_dist, query_section, c_section
            )

            # Compute scores
            consistency_score = 1.0 if consistent else max(0.3, 1.0 - len(conflicts) * 0.15)

            # Signature score (REWA-based)
            sig_score = sig_similarity

            # Modifier score (penalize if modifier effect detected)
            modifier_score = modifier_result.confidence if not modifier_result.has_conflict else 0.5

            # Build flags
            flags = []
            explanation_parts = []

            if not consistent:
                flags.append("TOPOS_INCONSISTENT")
                explanation_parts.append(f"Conflicts on: {conflicts[:3]}")

            if modifier_result.has_conflict:
                flags.append("MODIFIER_EFFECT")
                explanation_parts.append(modifier_result.explanation)

            if sig_similarity < self.config.signature_similarity_threshold:
                flags.append("LOW_SIGNATURE_MATCH")
                explanation_parts.append(f"Signature similarity: {sig_similarity:.2f}")

            if dist_similarity < 0.3:
                flags.append("SEMANTIC_DRIFT")
                explanation_parts.append(f"Distribution similarity: {dist_similarity:.2f}")

            chunk_verifications.append(ChunkVerification(
                text=chunk_text,
                is_consistent=consistent and not modifier_result.has_conflict,
                consistency_score=consistency_score,
                modifier_score=modifier_score * sig_score,  # Combined score
                flags=flags,
                explanation="; ".join(explanation_parts) if explanation_parts else "OK"
            ))

        # 4. Global consistency: can all chunks glue together?
        global_warnings = []

        if len(chunk_data) > 1:
            all_sections = [query_section] + [cd[3] for cd in chunk_data]

            # Try to glue all sections
            inference_result = self.reasoning.infer_from_sections(all_sections)

            if inference_result.status == "contradiction":
                global_warnings.append(
                    f"GLOBAL CONTRADICTION: {inference_result.explanation}"
                )

                # Add details about which chunks conflict
                if inference_result.details and "contradictions" in inference_result.details:
                    for contradiction in inference_result.details["contradictions"][:3]:
                        s1, s2 = contradiction["sections"]
                        conflicts = contradiction["conflicts"]
                        global_warnings.append(f"  {s1} vs {s2}: {conflicts}")

            # Check pairwise chunk consistency
            for i in range(len(chunk_data)):
                for j in range(i + 1, len(chunk_data)):
                    _, _, _, section_i = chunk_data[i]
                    _, _, _, section_j = chunk_data[j]

                    consistent, conflicts = self.topos.check_gluing_consistency(
                        section_i, section_j
                    )

                    if not consistent:
                        global_warnings.append(
                            f"Chunks {i} and {j} conflict on: {conflicts[:3]}"
                        )

        # 5. Semantic coherence check (simplified RG-style)
        if len(chunk_data) > 1:
            # Check if chunks are semantically coherent (similar distributions)
            chunk_dists = [cd[1] for cd in chunk_data]

            # Compute pairwise distribution distances
            total_drift = 0.0
            pair_count = 0

            for i in range(len(chunk_dists)):
                for j in range(i + 1, len(chunk_dists)):
                    drift = self.reasoning.compute_distribution_distance(
                        chunk_dists[i], chunk_dists[j]
                    )
                    total_drift += drift
                    pair_count += 1

            if pair_count > 0:
                avg_drift = total_drift / pair_count

                if avg_drift > 2.0:  # High semantic drift
                    global_warnings.append(
                        f"HIGH SEMANTIC DRIFT: Chunks are semantically divergent "
                        f"(avg distance={avg_drift:.2f})"
                    )

        # 6. Compute overall confidence
        if not chunk_verifications:
            overall_confidence = 0.0
        else:
            # Weighted combination of scores
            scores = []
            for cv in chunk_verifications:
                combined = cv.consistency_score * cv.modifier_score
                scores.append(combined)

            overall_confidence = np.mean(scores)

            # Penalize for global warnings
            overall_confidence *= (1.0 - 0.1 * len(global_warnings))
            overall_confidence = max(0.0, overall_confidence)

        return VerificationResult(
            query=query,
            verified_chunks=chunk_verifications,
            overall_confidence=overall_confidence,
            global_warnings=global_warnings
        )

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index documents for feasibility checking.

        Args:
            documents: List of {'id': ..., 'text': ...} dicts
        """
        for doc in documents:
            doc_id = doc['id']
            text = doc['text']

            witnesses, distribution, signature = self._extract_and_encode(text, doc_id)

            self.reasoning.index_document(
                doc_id=doc_id,
                witness_distribution=distribution,
                signature=signature
            )

        logger.info(f"Indexed {len(documents)} documents for feasibility checking")

    def get_verification_summary(self, result: VerificationResult) -> str:
        """Generate human-readable summary of verification result."""
        lines = [
            f"Query: {result.query[:50]}...",
            f"Overall Confidence: {result.overall_confidence:.2%}",
            f"Chunks Verified: {len(result.verified_chunks)}",
            ""
        ]

        # Per-chunk summary
        for i, cv in enumerate(result.verified_chunks):
            status = "OK" if cv.is_consistent else "ISSUE"
            lines.append(
                f"  Chunk {i}: [{status}] "
                f"consistency={cv.consistency_score:.2f}, "
                f"modifier={cv.modifier_score:.2f}"
            )
            if cv.flags:
                lines.append(f"    Flags: {cv.flags}")
            if cv.explanation and cv.explanation != "OK":
                lines.append(f"    Note: {cv.explanation[:80]}")

        # Global warnings
        if result.global_warnings:
            lines.append("")
            lines.append("Global Warnings:")
            for warning in result.global_warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)

    def verify_compositional(
        self,
        query: str,
        chunks: List[str]
    ) -> Dict:
        """
        Verify chunks using compositional reasoning with structured sections.

        This extracts noun phrases with their bound modifiers and checks
        if the query's required predicates appear together in any chunk's sections.

        Args:
            query: Natural language query (e.g., "red bike")
            chunks: List of text chunks to verify

        Returns:
            Dict with:
                - query_predicates: Set of predicates extracted from query
                - chunk_results: Per-chunk analysis with sections and matches
                - matching_chunks: Indices of chunks that satisfy the query
                - best_match: Index of best matching chunk (or None)
        """
        # Extract query predicates
        query_preds = self.section_extractor.query_to_predicates(query)

        chunk_results = []
        matching_indices = []

        for i, chunk_text in enumerate(chunks):
            # Extract structured sections from chunk
            sections = self.section_extractor.extract_sections(chunk_text, f"chunk_{i}")

            # Find sections that match ALL query predicates
            matches = self.section_extractor.find_matching_sections(sections, query_preds)

            has_match = len(matches) > 0

            if has_match:
                matching_indices.append(i)

            chunk_results.append({
                "chunk_index": i,
                "text": chunk_text,
                "sections": [
                    {
                        "id": s.region_id,
                        "predicates": [p.predicate for p in s.propositions]
                    }
                    for s in sections
                ],
                "matching_sections": [
                    {
                        "id": s.region_id,
                        "predicates": [p.predicate for p in s.propositions]
                    }
                    for s in matches
                ],
                "has_compositional_match": has_match
            })

        return {
            "query": query,
            "query_predicates": list(query_preds),
            "chunk_results": chunk_results,
            "matching_chunks": matching_indices,
            "best_match": matching_indices[0] if matching_indices else None
        }
