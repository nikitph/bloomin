"""
Comprehensive unit tests for REWA-Causal Engine.

Tests all layers:
1. Witness Extraction
2. Geometry (Spherical Hulls)
3. Causal Graph
4. Identification (Backdoor, Front-door)
5. Refusal & Safety
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from witness import WitnessSet, extract_witnesses, normalize_witness
from geometry import (
    SphericalConvexHull,
    spherical_convex_hull,
    satisfies_hemisphere_constraint,
    frechet_mean,
    compute_dispersion,
    geodesic_distance,
    hull_overlap,
    weighted_union
)
from causal_graph import CausalGraph
from identification import (
    backdoor_adjustment,
    frontdoor_adjustment,
    interventional_region,
    causal_effect
)
from refusal import validate_region, validate_causal_claim, RefusalResult, ValidationConfig
from utils import (
    generate_confounded_dataset,
    generate_mediated_dataset,
    generate_frontdoor_dataset
)


class TestWitnessExtraction(unittest.TestCase):
    """Tests for the Witness Extraction Layer."""

    def test_witness_set_creation(self):
        """Test creating a WitnessSet."""
        witnesses = np.random.randn(10, 64)
        ws = WitnessSet(variable='X', witnesses=witnesses)

        self.assertEqual(ws.variable, 'X')
        self.assertEqual(ws.n_witnesses, 10)
        self.assertEqual(ws.dimension, 64)

    def test_witness_normalization(self):
        """Test that witnesses are normalized to unit sphere."""
        witnesses = np.random.randn(5, 32) * 10  # Large values
        ws = WitnessSet(variable='X', witnesses=witnesses)

        # Check all witnesses have unit norm
        norms = np.linalg.norm(ws.witnesses, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5))

    def test_normalize_witness_function(self):
        """Test the normalize_witness utility function."""
        w = np.array([3, 4, 0])
        w_norm = normalize_witness(w)

        self.assertAlmostEqual(np.linalg.norm(w_norm), 1.0)
        np.testing.assert_array_almost_equal(w_norm, [0.6, 0.8, 0.0])

    def test_mean_witness(self):
        """Test computing mean witness."""
        # Create witnesses clustered around a point
        center = np.array([1, 0, 0, 0])
        noise = np.random.randn(20, 4) * 0.1
        witnesses = center + noise
        ws = WitnessSet(variable='X', witnesses=witnesses)

        mean = ws.mean_witness()
        self.assertAlmostEqual(np.linalg.norm(mean), 1.0)
        # Mean should be close to original center direction
        self.assertGreater(np.dot(mean, center / np.linalg.norm(center)), 0.9)

    def test_pairwise_similarities(self):
        """Test pairwise similarity computation."""
        witnesses = np.eye(3)  # Orthogonal vectors
        ws = WitnessSet(variable='X', witnesses=witnesses)

        sims = ws.pairwise_similarities()
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(sims), [1, 1, 1])
        # Off-diagonal should be 0 (orthogonal)
        self.assertAlmostEqual(sims[0, 1], 0.0)

    def test_extract_witnesses_auto(self):
        """Test automatic witness extraction."""
        # Dictionary data
        data = {'credit_score': 750, 'dti': 0.35}
        ws = extract_witnesses(data, variable='loan', method='attribute',
                               schema={'credit_score': {'type': 'numeric', 'min': 300, 'max': 850},
                                       'dti': {'type': 'numeric', 'min': 0, 'max': 1}})

        self.assertEqual(ws.variable, 'loan')
        self.assertEqual(ws.n_witnesses, 1)


class TestGeometry(unittest.TestCase):
    """Tests for the Geometry Layer."""

    def test_spherical_hull_creation(self):
        """Test creating a spherical convex hull."""
        points = np.random.randn(10, 64)
        hull = spherical_convex_hull(points)

        self.assertEqual(hull.n_points, 10)
        self.assertEqual(hull.dimension, 64)
        self.assertIsNotNone(hull.centroid)

    def test_hull_normalization(self):
        """Test that hull points are normalized."""
        points = np.random.randn(5, 32) * 5
        hull = SphericalConvexHull(points=points)

        norms = np.linalg.norm(hull.points, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5))

    def test_hemisphere_constraint_valid(self):
        """Test hemisphere constraint for valid set."""
        # Points clustered in one hemisphere
        center = np.array([1, 0, 0])
        noise = np.random.randn(20, 3) * 0.2
        points = center + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        is_valid, violations = satisfies_hemisphere_constraint(points)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

    def test_hemisphere_constraint_invalid(self):
        """Test hemisphere constraint for contradictory set."""
        # Points on opposite sides of sphere
        points = np.array([
            [1, 0, 0],
            [-1, 0, 0]  # Antipodal
        ])

        is_valid, violations = satisfies_hemisphere_constraint(points, threshold=-0.5)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)

    def test_frechet_mean(self):
        """Test Fréchet mean computation."""
        # Points clustered around [1, 0, 0]
        center = np.array([1, 0, 0])
        noise = np.random.randn(50, 3) * 0.1
        points = center + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        mean = frechet_mean(points)
        self.assertAlmostEqual(np.linalg.norm(mean), 1.0)
        # Mean should be close to center
        self.assertGreater(np.dot(mean, center), 0.95)

    def test_geodesic_distance(self):
        """Test geodesic distance computation."""
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 1, 0])  # Orthogonal

        dist = geodesic_distance(p1, p2)
        self.assertAlmostEqual(dist, np.pi / 2, places=5)

        # Same point
        dist_same = geodesic_distance(p1, p1)
        self.assertAlmostEqual(dist_same, 0.0, places=5)

        # Antipodal
        dist_anti = geodesic_distance(p1, -p1)
        self.assertAlmostEqual(dist_anti, np.pi, places=5)

    def test_compute_dispersion(self):
        """Test dispersion metrics."""
        # Tight cluster
        center = np.array([1, 0, 0, 0])
        noise = np.random.randn(30, 4) * 0.05
        tight_points = center + noise
        tight_points = tight_points / np.linalg.norm(tight_points, axis=1, keepdims=True)

        # Spread cluster
        spread_noise = np.random.randn(30, 4) * 0.5
        spread_points = center + spread_noise
        spread_points = spread_points / np.linalg.norm(spread_points, axis=1, keepdims=True)

        tight_disp = compute_dispersion(tight_points)
        spread_disp = compute_dispersion(spread_points)

        self.assertLess(tight_disp['radius'], spread_disp['radius'])
        self.assertLess(tight_disp['spread'], spread_disp['spread'])

    def test_hull_overlap(self):
        """Test hull overlap detection."""
        # Two overlapping hulls
        center1 = np.array([1, 0, 0, 0])
        center2 = np.array([0.9, 0.3, 0, 0])
        center2 = center2 / np.linalg.norm(center2)

        points1 = center1 + np.random.randn(20, 4) * 0.2
        points1 = points1 / np.linalg.norm(points1, axis=1, keepdims=True)

        points2 = center2 + np.random.randn(20, 4) * 0.2
        points2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)

        hull1 = spherical_convex_hull(points1)
        hull2 = spherical_convex_hull(points2)

        overlap = hull_overlap(hull1, hull2)
        self.assertIn('has_overlap', overlap)
        self.assertIn('overlap_fraction', overlap)

    def test_weighted_union(self):
        """Test weighted union of hulls."""
        hull1 = spherical_convex_hull(np.random.randn(10, 32))
        hull2 = spherical_convex_hull(np.random.randn(10, 32))

        combined = weighted_union([hull1, hull2], weights=[0.7, 0.3])

        self.assertEqual(combined.n_points, 20)
        self.assertIsNotNone(combined.centroid)


class TestCausalGraph(unittest.TestCase):
    """Tests for the Causal Graph Layer."""

    def test_graph_creation(self):
        """Test creating a causal graph."""
        graph = CausalGraph(
            nodes=['X', 'Y', 'Z'],
            edges=[('X', 'Y'), ('Z', 'Y')]
        )

        self.assertEqual(len(graph.nodes()), 3)
        self.assertEqual(len(graph.edges()), 2)

    def test_parents_children(self):
        """Test parent and child retrieval."""
        graph = CausalGraph(
            nodes=['X', 'Y', 'Z'],
            edges=[('X', 'Y'), ('Z', 'Y')]
        )

        self.assertEqual(graph.parents('Y'), {'X', 'Z'})
        self.assertEqual(graph.children('X'), {'Y'})
        self.assertEqual(graph.parents('X'), set())

    def test_ancestors_descendants(self):
        """Test ancestor and descendant retrieval."""
        graph = CausalGraph(
            nodes=['A', 'B', 'C', 'D'],
            edges=[('A', 'B'), ('B', 'C'), ('C', 'D')]
        )

        self.assertEqual(graph.ancestors('D'), {'A', 'B', 'C'})
        self.assertEqual(graph.descendants('A'), {'B', 'C', 'D'})

    def test_backdoor_criterion_valid(self):
        """Test valid backdoor adjustment."""
        # X → Y, Z → X, Z → Y (Z is confounder)
        graph = CausalGraph(
            nodes=['X', 'Y', 'Z'],
            edges=[('Z', 'X'), ('Z', 'Y'), ('X', 'Y')]
        )

        is_valid, details = graph.verify_backdoor({'Z'}, 'X', 'Y')
        self.assertTrue(is_valid)

    def test_backdoor_criterion_invalid(self):
        """Test invalid backdoor adjustment (descendant in adjustment set)."""
        # X → M → Y (M is mediator, descendant of X)
        graph = CausalGraph(
            nodes=['X', 'M', 'Y'],
            edges=[('X', 'M'), ('M', 'Y')]
        )

        # Including M (descendant of X) violates backdoor criterion
        is_valid, details = graph.verify_backdoor({'M'}, 'X', 'Y')
        self.assertFalse(is_valid)

    def test_frontdoor_criterion(self):
        """Test front-door criterion verification."""
        # X → M → Y, U → X, U → Y
        graph = CausalGraph(
            nodes=['U', 'X', 'M', 'Y'],
            edges=[('U', 'X'), ('U', 'Y'), ('X', 'M'), ('M', 'Y')]
        )
        graph._nodes['U'].is_latent = True

        is_valid, details = graph.verify_frontdoor('M', 'X', 'Y')
        self.assertTrue(is_valid)

    def test_find_adjustment_set(self):
        """Test automatic adjustment set finding."""
        graph = CausalGraph(
            nodes=['X', 'Y', 'Z'],
            edges=[('Z', 'X'), ('Z', 'Y'), ('X', 'Y')]
        )

        adjustment = graph.find_valid_adjustment_set('X', 'Y')
        self.assertIsNotNone(adjustment)
        self.assertIn('Z', adjustment)


class TestIdentification(unittest.TestCase):
    """Tests for the Identification Layer."""

    def test_backdoor_adjustment(self):
        """Test backdoor adjustment computation."""
        dataset = generate_confounded_dataset(n_samples=50, dimension=32)

        # Add Z as an observed proxy for the confounder
        Z_witnesses = dataset.witnesses['U'].witnesses.copy()
        dataset.witnesses['Z'] = WitnessSet(variable='Z', witnesses=Z_witnesses)
        dataset.graph.add_node('Z')
        dataset.graph.add_edge('Z', 'X')
        dataset.graph.add_edge('Z', 'Y')

        result = backdoor_adjustment(
            X='X',
            x_target='high',
            Z={'Z'},
            Y='Y',
            witnesses=dataset.witnesses,
            graph=dataset.graph
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.region)
        self.assertEqual(result.method, 'backdoor')

    def test_frontdoor_adjustment(self):
        """Test front-door adjustment computation."""
        dataset = generate_frontdoor_dataset(n_samples=50, dimension=32)

        result = frontdoor_adjustment(
            X='X',
            x_target='high',
            M='M',
            Y='Y',
            witnesses=dataset.witnesses,
            graph=dataset.graph
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.region)
        self.assertEqual(result.method, 'frontdoor')

    def test_causal_effect(self):
        """Test causal effect computation."""
        dataset = generate_mediated_dataset(n_samples=50, dimension=32)

        effect = causal_effect(
            X='X',
            x0='low',
            x1='high',
            Y='Y',
            witnesses=dataset.witnesses,
            graph=dataset.graph
        )

        self.assertIsNotNone(effect.region_x0)
        self.assertIsNotNone(effect.region_x1)
        self.assertGreater(effect.centroid_shift, 0)


class TestRefusal(unittest.TestCase):
    """Tests for the Refusal & Safety Layer."""

    def test_validate_valid_region(self):
        """Test validation of a valid region."""
        points = np.random.randn(20, 32) * 0.2
        center = np.array([1] + [0] * 31)
        points = center + points
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        hull = spherical_convex_hull(points)
        result = validate_region(hull)

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.is_valid)

    def test_validate_insufficient_witnesses(self):
        """Test refusal for insufficient witnesses."""
        points = np.random.randn(2, 32)
        hull = spherical_convex_hull(points)

        result = validate_region(hull, ValidationConfig(min_witnesses=5))

        self.assertEqual(result.status, "refuse")
        self.assertEqual(result.reason.value, "insufficient_witnesses")

    def test_validate_hemisphere_violation(self):
        """Test refusal for hemisphere violation."""
        # Create contradictory points
        points = np.array([
            [1, 0, 0, 0],
            [-0.9, 0, 0, 0],  # Nearly antipodal
            [0, 1, 0, 0],
            [0, -0.9, 0, 0]
        ])

        hull = SphericalConvexHull(points=points)
        result = validate_region(hull, ValidationConfig(hemisphere_threshold=-0.5))

        self.assertEqual(result.status, "refuse")
        self.assertEqual(result.reason.value, "hemisphere_violation")

    def test_validate_causal_claim_valid(self):
        """Test validation of a valid causal claim."""
        # Two distinct regions
        center1 = np.array([1, 0, 0, 0])
        center2 = np.array([0, 1, 0, 0])

        points1 = center1 + np.random.randn(20, 4) * 0.1
        points1 = points1 / np.linalg.norm(points1, axis=1, keepdims=True)

        points2 = center2 + np.random.randn(20, 4) * 0.1
        points2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)

        hull1 = spherical_convex_hull(points1)
        hull2 = spherical_convex_hull(points2)

        result = validate_causal_claim(hull1, hull2)

        self.assertEqual(result.status, "ok")

    def test_validate_causal_claim_overlapping(self):
        """Test refusal for heavily overlapping regions."""
        # Same region, slight noise
        center = np.array([1, 0, 0, 0])

        points1 = center + np.random.randn(20, 4) * 0.1
        points1 = points1 / np.linalg.norm(points1, axis=1, keepdims=True)

        points2 = center + np.random.randn(20, 4) * 0.1
        points2 = points2 / np.linalg.norm(points2, axis=1, keepdims=True)

        hull1 = spherical_convex_hull(points1)
        hull2 = spherical_convex_hull(points2)

        result = validate_causal_claim(
            hull1, hull2,
            ValidationConfig(min_effect_size=0.5, max_overlap_fraction=0.3)
        )

        self.assertEqual(result.status, "refuse")


class TestSyntheticDatasets(unittest.TestCase):
    """Tests for synthetic data generators."""

    def test_confounded_dataset(self):
        """Test confounded dataset generation."""
        dataset = generate_confounded_dataset(n_samples=30, dimension=16)

        self.assertIn('X', dataset.witnesses)
        self.assertIn('Y', dataset.witnesses)
        self.assertIn('U', dataset.witnesses)

        self.assertEqual(dataset.witnesses['X'].n_witnesses, 30)
        self.assertTrue(dataset.ground_truth['requires_adjustment'])

    def test_mediated_dataset(self):
        """Test mediated dataset generation."""
        dataset = generate_mediated_dataset(n_samples=30, dimension=16)

        self.assertIn('X', dataset.witnesses)
        self.assertIn('M', dataset.witnesses)
        self.assertIn('Y', dataset.witnesses)

        self.assertFalse(dataset.ground_truth['requires_adjustment'])

    def test_frontdoor_dataset(self):
        """Test front-door dataset generation."""
        dataset = generate_frontdoor_dataset(n_samples=30, dimension=16)

        self.assertIn('X', dataset.witnesses)
        self.assertIn('M', dataset.witnesses)
        self.assertIn('Y', dataset.witnesses)

        self.assertTrue(dataset.ground_truth['requires_frontdoor'])
        self.assertEqual(dataset.ground_truth['mediator'], 'M')


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_pipeline_backdoor(self):
        """Test full pipeline with backdoor adjustment."""
        # Generate data
        dataset = generate_confounded_dataset(n_samples=100, dimension=32)

        # Add observed confounder proxy
        Z_witnesses = dataset.witnesses['U'].witnesses.copy()
        dataset.witnesses['Z'] = WitnessSet(variable='Z', witnesses=Z_witnesses)
        dataset.graph.add_node('Z')
        dataset.graph.add_edge('Z', 'X')
        dataset.graph.add_edge('Z', 'Y')

        # Compute causal effect
        effect = causal_effect(
            X='X', x0='low', x1='high', Y='Y',
            witnesses=dataset.witnesses,
            graph=dataset.graph
        )

        # Validate
        if effect.region_x0 and effect.region_x1:
            result = validate_causal_claim(effect.region_x0, effect.region_x1)

            # Should produce some result
            self.assertIn(result.status, ['ok', 'refuse'])

    def test_refusal_on_contradictory_evidence(self):
        """Test that contradictory evidence triggers refusal."""
        # Create contradictory witnesses
        witnesses = np.array([
            [1, 0, 0, 0],
            [-1, 0, 0, 0],  # Directly antipodal
            [0, 1, 0, 0],
            [0, -1, 0, 0]   # Directly antipodal
        ])

        hull = spherical_convex_hull(witnesses)
        result = validate_region(hull)

        self.assertEqual(result.status, "refuse")
        self.assertEqual(result.reason.value, "hemisphere_violation")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWitnessExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestIdentification))
    suite.addTests(loader.loadTestsFromTestCase(TestRefusal))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticDatasets))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_tests()
