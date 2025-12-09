"""
REWA Integration Tests

Tests the complete REWA system end-to-end.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rewa import REWA, RewaStatus, RewaResponse
from rewa.models import (
    Chart, Entity, Fact, Rule, QueryIntent,
    ValueConstraint, ComparisonOp, RuleConstraint,
)
from rewa.geometry import (
    normalize_embedding, cosine_similarity, angular_distance,
    ChartManager, create_chart_from_embeddings,
)
from rewa.extraction import EntityExtractor, FactExtractor, compile_query
from rewa.rules import RuleEngine, ConstraintSolver
from rewa.validation import ContradictionDetector, ImpossibilityChecker
import numpy as np


class TestGeometry:
    """Tests for geometry module."""

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        vec = np.array([3.0, 4.0, 0.0])
        normalized = normalize_embedding(vec)
        assert np.isclose(np.linalg.norm(normalized), 1.0)
        assert np.isclose(normalized[0], 0.6)
        assert np.isclose(normalized[1], 0.8)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = normalize_embedding(np.array([1.0, 0.0, 0.0]))
        b = normalize_embedding(np.array([1.0, 0.0, 0.0]))
        c = normalize_embedding(np.array([0.0, 1.0, 0.0]))

        assert np.isclose(cosine_similarity(a, b), 1.0)
        assert np.isclose(cosine_similarity(a, c), 0.0)

    def test_angular_distance(self):
        """Test angular distance calculation."""
        a = normalize_embedding(np.array([1.0, 0.0, 0.0]))
        b = normalize_embedding(np.array([0.0, 1.0, 0.0]))

        dist = angular_distance(a, b)
        assert np.isclose(dist, np.pi / 2)

    def test_create_chart(self):
        """Test chart creation from embeddings."""
        embeddings = [
            normalize_embedding(np.random.randn(128))
            for _ in range(10)
        ]

        chart = create_chart_from_embeddings(
            embeddings,
            domain_tags={"test"},
            chart_id="test_chart"
        )

        assert chart.id == "test_chart"
        assert "test" in chart.domain_tags
        assert chart.radius > 0
        assert np.isclose(np.linalg.norm(chart.witness_embedding), 1.0)

    def test_chart_manager(self):
        """Test chart manager operations."""
        manager = ChartManager()

        # Create and add charts
        for i in range(3):
            emb = normalize_embedding(np.random.randn(128))
            chart = Chart(
                id=f"chart_{i}",
                witness_embedding=emb,
                radius=0.5,
                intrinsic_dim=10,
                domain_tags={f"domain_{i}"},
            )
            manager.add_chart(chart)

        assert len(manager) == 3

        # Test retrieval
        chart = manager.get_chart("chart_0")
        assert chart is not None
        assert chart.id == "chart_0"

        # Test domain filtering
        domain_charts = manager.get_charts_by_domain("domain_1")
        assert len(domain_charts) == 1


class TestExtraction:
    """Tests for extraction module."""

    def test_entity_extraction(self):
        """Test entity extraction from text."""
        from rewa.extraction.entity_extractor import Chunk

        extractor = EntityExtractor()
        chunk = Chunk(
            id="test",
            text="The gun is a Glock 19 semi-automatic pistol used for self-defense."
        )

        entities = extractor.extract(chunk)
        assert len(entities) > 0

        # Should extract weapon-related entity
        weapon_entities = [e for e in entities if e.type == "Weapon"]
        assert len(weapon_entities) > 0

    def test_fact_extraction(self):
        """Test fact extraction from text."""
        extractor = FactExtractor()
        entity = Entity(id="e1", type="Weapon", name="gun", properties={})

        text = "The gun is dangerous and can be used for self-defense."
        facts = extractor.extract(text, [entity], chunk_id="test")

        # Should extract danger-related facts
        assert len(facts) > 0

    def test_query_compilation(self):
        """Test query compilation."""
        intent = compile_query("Find a gun for self-defense")

        assert "Weapon" in intent.required_types
        assert len(intent.required_properties) > 0


class TestRules:
    """Tests for rules module."""

    def test_rule_engine(self):
        """Test rule engine evaluation."""
        engine = RuleEngine()

        # Add a test rule
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="Test",
            preconditions=[
                RuleConstraint(
                    property_path="entity.type",
                    constraint=ValueConstraint(op=ComparisonOp.EQ, value="Weapon")
                )
            ],
            postconditions=[
                RuleConstraint(
                    property_path="fact.dangerous",
                    constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
                )
            ],
            hardness="hard",
            domain_tags={"weapon"},
        )
        engine.add_rule(rule)

        # Create test entity and facts
        entity = Entity(id="e1", type="Weapon", name="gun")
        facts = [
            Fact(id="f1", subject=entity, predicate="dangerous", value=True)
        ]

        # Evaluate
        eval_result = engine.evaluate_rule(rule, entity, facts)
        assert eval_result.preconditions_met
        assert eval_result.postconditions_met
        assert eval_result.is_satisfied

    def test_constraint_solver(self):
        """Test constraint solver."""
        engine = RuleEngine()
        solver = ConstraintSolver(engine)

        entity = Entity(id="e1", type="Weapon", name="gun", properties={"real": True})
        facts = [
            Fact(id="f1", subject=entity, predicate="dangerous", value=True)
        ]

        intent = QueryIntent(
            required_types={"Weapon"},
            required_properties={
                "dangerous": ValueConstraint(op=ComparisonOp.EQ, value=True)
            },
        )

        result = solver.validate_entity(entity, facts, intent)
        assert result.satisfied


class TestValidation:
    """Tests for validation module."""

    def test_contradiction_detection(self):
        """Test contradiction detection."""
        detector = ContradictionDetector()

        entity = Entity(id="e1", type="Drug", name="test drug")
        facts = [
            Fact(id="f1", subject=entity, predicate="safe", value=True),
            Fact(id="f2", subject=entity, predicate="safe", value=False),
        ]

        contradictions = detector.detect(facts)
        assert len(contradictions) > 0

    def test_impossibility_detection(self):
        """Test impossibility detection."""
        checker = ImpossibilityChecker()

        # Test cancer cure without side effects
        intent = compile_query("Drug that cures cancer with no side effects")
        impossibilities = checker.check(intent)

        assert len(impossibilities) > 0


class TestREWAIntegration:
    """Integration tests for complete REWA system."""

    def test_valid_query(self):
        """Test valid query processing."""
        rewa = REWA()

        chunks = [
            {
                "id": "c1",
                "text": "The Glock 19 is a real semi-automatic pistol "
                       "commonly used for self-defense. It is a dangerous "
                       "firearm that requires proper training."
            }
        ]

        response = rewa.verify("Find a gun for self-defense", chunks)

        # Should process successfully
        assert response.status in [RewaStatus.VALID, RewaStatus.INSUFFICIENT_EVIDENCE]

    def test_impossible_query(self):
        """Test impossible query detection."""
        rewa = REWA()

        chunks = [
            {
                "id": "c1",
                "text": "MiracleCure cures all cancers with zero side effects."
            }
        ]

        response = rewa.verify(
            "Drug that cures cancer with zero side effects",
            chunks
        )

        assert response.status == RewaStatus.IMPOSSIBLE
        assert len(response.impossibilities) > 0

    def test_negation_sensitivity(self):
        """Test that system distinguishes real from toy."""
        rewa = REWA()

        # Toy gun chunk
        toy_chunks = [
            {
                "id": "c1",
                "text": "The Nerf N-Strike is a toy gun that shoots foam darts. "
                       "It is NOT a real weapon and is completely safe for children."
            }
        ]

        response = rewa.verify("Real gun for self-defense", toy_chunks)

        # Should not validate toy as real gun
        assert response.status != RewaStatus.VALID or len(response.safe_facts) == 0

    def test_contradiction_handling(self):
        """Test handling of contradictory information."""
        rewa = REWA()

        chunks = [
            {
                "id": "c1",
                "text": "Product X is completely safe and non-toxic."
            },
            {
                "id": "c2",
                "text": "Product X is dangerous and can cause serious harm."
            }
        ]

        response = rewa.verify("Is Product X safe?", chunks)

        # Should detect conflict or handle appropriately
        assert response.status in [
            RewaStatus.CONFLICT,
            RewaStatus.INSUFFICIENT_EVIDENCE,
            RewaStatus.VALID  # If resolved
        ]


class TestExperiments:
    """Tests for experiment framework."""

    def test_impossible_query_experiment(self):
        """Test impossible query experiment."""
        from rewa.experiments import ImpossibleQueryExperiment

        experiment = ImpossibleQueryExperiment()
        cases = experiment.get_test_cases()

        assert len(cases) > 0

        # Run a single case
        impossible_case = cases[0]
        result = experiment.run_case(impossible_case)

        assert result.experiment_name == "impossible_query_detection"

    def test_experiment_runner(self):
        """Test experiment runner."""
        from rewa.experiments import ExperimentRunner

        runner = ExperimentRunner()

        # Just verify runner initializes correctly
        assert len(runner.EXPERIMENTS) == 5


def run_quick_validation():
    """Quick validation of core functionality."""
    print("=" * 60)
    print("REWA Quick Validation")
    print("=" * 60)

    # Test 1: Basic geometry
    print("\n[1/5] Testing geometry...")
    vec = normalize_embedding(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(np.linalg.norm(vec), 1.0)
    print("  OK: Embedding normalization")

    # Test 2: Entity extraction
    print("\n[2/5] Testing entity extraction...")
    from rewa.extraction.entity_extractor import Chunk
    extractor = EntityExtractor()
    chunk = Chunk(id="test", text="The gun is dangerous")
    entities = extractor.extract(chunk)
    assert len(entities) > 0
    print(f"  OK: Extracted {len(entities)} entities")

    # Test 3: Query compilation
    print("\n[3/5] Testing query compilation...")
    intent = compile_query("Find a weapon for self-defense")
    assert "Weapon" in intent.required_types
    print(f"  OK: Compiled query with types {intent.required_types}")

    # Test 4: Impossibility detection
    print("\n[4/5] Testing impossibility detection...")
    checker = ImpossibilityChecker()
    intent = compile_query("Cancer cure with zero side effects")
    impossibilities = checker.check(intent)
    assert len(impossibilities) > 0
    print(f"  OK: Detected {len(impossibilities)} impossibilities")

    # Test 5: Full REWA integration
    print("\n[5/5] Testing REWA integration...")
    rewa = REWA()
    response = rewa.verify(
        "Cancer cure with no side effects",
        [{"id": "c1", "text": "MiraclePill cures cancer with no side effects"}]
    )
    assert response.status == RewaStatus.IMPOSSIBLE
    print(f"  OK: Response status = {response.status.value}")

    print("\n" + "=" * 60)
    print("All quick validation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_validation()
