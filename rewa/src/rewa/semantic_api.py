"""
REWA Semantic API

A cleaner API that uses real embeddings instead of brittle regex patterns.
This is the recommended way to use REWA.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from rewa.models import (
    Chart,
    Entity,
    Fact,
    Rule,
    QueryIntent,
    LocalWorld,
    RewaResponse,
    RewaStatus,
    Contradiction,
    Impossibility,
    ValidationResult,
    ValueConstraint,
    ComparisonOp,
    Vector,
)
from rewa.embeddings import Embedder, SemanticMatcher
from rewa.geometry.charts import ChartManager, create_chart_from_embeddings
from rewa.geometry.selector import ChartSelector
from rewa.validation.contradiction import ContradictionDetector
from rewa.rules.engine import RuleEngine
from rewa.rules.loader import get_default_rules


@dataclass
class SemanticRewaConfig:
    """Configuration for Semantic REWA."""
    embedding_model: str = "all-MiniLM-L6-v2"
    type_threshold: float = 0.45
    property_threshold: float = 0.45
    impossibility_threshold: float = 0.75  # Higher to reduce false positives
    contradiction_enabled: bool = True
    min_confidence: float = 0.4


@dataclass
class SemanticChunk:
    """A chunk with its embedding."""
    id: str
    text: str
    embedding: Optional[Vector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0


class SemanticREWA:
    """
    REWA with real semantic embeddings.

    This version uses sentence-transformers for semantic understanding
    instead of brittle regex patterns.
    """

    def __init__(self, config: Optional[SemanticRewaConfig] = None):
        self.config = config or SemanticRewaConfig()
        self.embedder = Embedder(self.config.embedding_model)
        self.matcher = SemanticMatcher(self.embedder)
        self.contradiction_detector = ContradictionDetector()

        # Pre-compute anchor embeddings
        print("Pre-computing semantic anchors...")
        self.matcher.precompute_anchors()
        print("Ready!")

        # Rule engine for hard constraints
        self.rule_engine = RuleEngine()
        self.rule_engine.add_rules(get_default_rules())

    def verify(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> RewaResponse:
        """
        Verify retrieved information against a query using semantic understanding.

        Args:
            query: Natural language query
            chunks: Retrieved chunks, each with "id", "text", optional "embedding"

        Returns:
            RewaResponse with validation results
        """
        # Step 1: Check for impossibility FIRST
        impossibilities = self._check_impossibility(query)
        if impossibilities:
            return RewaResponse(
                status=RewaStatus.IMPOSSIBLE,
                safe_facts=[],
                impossibilities=impossibilities,
                explanation=impossibilities[0].explanation or impossibilities[0].reason,
                confidence=1.0,
            )

        # Step 2: Embed query and chunks
        query_embedding = self.embedder.embed(query)
        embedded_chunks = self._embed_chunks(chunks)

        # Step 3: Detect query requirements semantically
        query_types = self.matcher.detect_types(query, self.config.type_threshold)
        query_props = self.matcher.detect_properties(query, self.config.property_threshold)

        # Step 4: Process each chunk
        all_entities: List[Entity] = []
        all_facts: List[Fact] = []
        chunk_scores: Dict[str, float] = {}

        for chunk in embedded_chunks:
            # Score chunk relevance to query
            if chunk.embedding is not None:
                relevance = float(np.dot(query_embedding, chunk.embedding))
            else:
                relevance = 0.5
            chunk_scores[chunk.id] = relevance

            # Extract entities and facts semantically
            entities, facts = self._extract_from_chunk(chunk)
            all_entities.extend(entities)
            all_facts.extend(facts)

        # Step 5: Check for contradictions
        contradictions = []
        if self.config.contradiction_enabled:
            contradictions = self.contradiction_detector.detect(all_facts)

            unresolved = [c for c in contradictions if not c.resolved_fact]
            if unresolved:
                return RewaResponse(
                    status=RewaStatus.CONFLICT,
                    safe_facts=all_facts,
                    contradictions=unresolved,
                    explanation=f"Found {len(unresolved)} contradictory facts",
                    confidence=0.5,
                )

        # Step 6: Validate entities against query requirements
        valid_entities = self._validate_entities(
            all_entities, all_facts, query_types, query_props
        )

        if not valid_entities:
            return RewaResponse(
                status=RewaStatus.INSUFFICIENT_EVIDENCE,
                safe_facts=all_facts,
                contradictions=contradictions,
                explanation="No entities satisfy query requirements",
                confidence=0.3,
            )

        # Step 7: Collect safe facts from valid entities
        safe_facts = []
        for entity, _ in valid_entities:
            entity_facts = [f for f in all_facts if f.subject.id == entity.id]
            safe_facts.extend(entity_facts)

        avg_confidence = np.mean([conf for _, conf in valid_entities])

        return RewaResponse(
            status=RewaStatus.VALID,
            safe_facts=safe_facts,
            contradictions=contradictions,
            validation_results=[
                ValidationResult(entity=e, satisfied=True, confidence=c)
                for e, c in valid_entities
            ],
            explanation=f"Found {len(valid_entities)} entities matching query",
            confidence=float(avg_confidence),
        )

    def _check_impossibility(self, query: str) -> List[Impossibility]:
        """Check for semantic impossibility patterns."""
        impossibilities = []

        # Use semantic matcher
        detected = self.matcher.check_impossibility(
            query, self.config.impossibility_threshold
        )

        for reason, confidence in detected:
            impossibilities.append(Impossibility(
                reason=reason,
                explanation=reason,
            ))

        return impossibilities

    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """Embed chunks that don't already have embeddings."""
        result = []

        for chunk in chunks:
            text = chunk.get("text", "")
            existing_emb = chunk.get("embedding")

            if existing_emb is not None:
                embedding = existing_emb
            else:
                embedding = self.embedder.embed(text)

            result.append(SemanticChunk(
                id=chunk.get("id", str(len(result))),
                text=text,
                embedding=embedding,
                metadata=chunk.get("metadata", {}),
                score=chunk.get("score", 1.0),
            ))

        return result

    def _extract_from_chunk(
        self,
        chunk: SemanticChunk
    ) -> Tuple[List[Entity], List[Fact]]:
        """Extract entities and facts from a chunk using semantic analysis."""
        entities = []
        facts = []

        text = chunk.text

        # Detect entity types
        type_scores = self.matcher.detect_types(text, self.config.type_threshold)

        if type_scores:
            # Create entity for highest-scoring type
            best_type = max(type_scores.items(), key=lambda x: x[1])
            entity_type, confidence = best_type

            entity = Entity(
                id=f"entity_{chunk.id}",
                type=entity_type,
                name=text[:50],  # Use first 50 chars as name
                properties={},
                confidence=confidence,
                source_chunk_id=chunk.id,
                embedding=chunk.embedding,
            )
            entities.append(entity)

            # Detect properties and create facts
            prop_scores = self.matcher.detect_properties(text, self.config.property_threshold)

            for prop_name, (value, confidence) in prop_scores.items():
                fact = Fact(
                    id=f"fact_{chunk.id}_{prop_name}",
                    subject=entity,
                    predicate=prop_name,
                    value=value,
                    confidence=confidence,
                    source_chunk_id=chunk.id,
                )
                facts.append(fact)

        return entities, facts

    def _validate_entities(
        self,
        entities: List[Entity],
        facts: List[Fact],
        required_types: Dict[str, float],
        required_props: Dict[str, Tuple[Any, float]],
    ) -> List[Tuple[Entity, float]]:
        """Validate entities against query requirements."""
        valid = []

        for entity in entities:
            confidence_factors = []

            # Check type match
            if required_types:
                if entity.type in required_types:
                    confidence_factors.append(required_types[entity.type])
                else:
                    continue  # Type doesn't match

            # Check property requirements
            entity_facts = [f for f in facts if f.subject.id == entity.id]
            fact_predicates = {f.predicate: f for f in entity_facts}

            props_matched = True
            for prop_name, (req_value, req_conf) in required_props.items():
                if prop_name in fact_predicates:
                    fact = fact_predicates[prop_name]
                    if fact.value == req_value:
                        confidence_factors.append(fact.confidence)
                    else:
                        props_matched = False
                        break

            if not props_matched:
                continue

            # Compute overall confidence
            if confidence_factors:
                avg_conf = np.mean(confidence_factors)
            else:
                avg_conf = entity.confidence

            if avg_conf >= self.config.min_confidence:
                valid.append((entity, float(avg_conf)))

        return valid


def create_semantic_rewa(config: Optional[SemanticRewaConfig] = None) -> SemanticREWA:
    """Create a SemanticREWA instance."""
    return SemanticREWA(config)
