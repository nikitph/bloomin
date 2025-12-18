#!/usr/bin/env python3
"""
Unit tests for the Unified Regulator Engine.
"""

import numpy as np
import unittest
from scipy import sparse

from ure_core import (
    Mode,
    RegulatorParams,
    UnifiedRegulatorEngine,
    build_knn_graph,
    build_grid_graph,
    op_wave,
    op_telegrapher,
    op_fokker_planck,
    op_schrodinger_imag,
    op_cahn_hilliard,
    op_fisher_kpp,
    op_poisson,
    normalize_field,
    regulator,
    initialize_state,
)


class TestGraphConstruction(unittest.TestCase):
    """Test graph building functions."""

    def test_knn_graph(self):
        """Test k-NN graph construction."""
        np.random.seed(42)
        vectors = np.random.randn(100, 32).astype(np.float32)
        A, L = build_knn_graph(vectors, k=5)

        # Check shapes
        self.assertEqual(A.shape, (100, 100))
        self.assertEqual(L.shape, (100, 100))

        # Check symmetry
        diff = np.abs(A - A.T).max()
        self.assertLess(diff, 1e-10)

        # Check Laplacian properties (should have zero row sum)
        row_sums = np.abs(np.array(L.sum(axis=1)).flatten())
        self.assertTrue(np.all(row_sums < 1e-10))

    def test_grid_graph(self):
        """Test grid graph construction."""
        A, L = build_grid_graph(10, 10, connectivity=4)

        self.assertEqual(A.shape, (100, 100))

        # Check corner nodes have degree 2
        corner_degree = A[0].sum()
        self.assertEqual(corner_degree, 2)

        # Check center nodes have degree 4
        center_idx = 5 * 10 + 5  # (5, 5)
        center_degree = A[center_idx].sum()
        self.assertEqual(center_degree, 4)


class TestOperators(unittest.TestCase):
    """Test individual operators."""

    def setUp(self):
        """Set up test fixtures."""
        self.A, self.L = build_grid_graph(10, 10)
        self.n = 100
        self.psi = np.random.rand(self.n)
        self.psi = self.psi / np.sum(self.psi)

    def test_wave_operator(self):
        """Test wave operator (Laplacian)."""
        dpsi = op_wave(self.psi, self.L)
        self.assertEqual(dpsi.shape, (self.n,))
        # Laplacian of constant should be zero
        const = np.ones(self.n) / self.n
        d_const = op_wave(const, self.L)
        self.assertTrue(np.allclose(d_const, 0, atol=1e-10))

    def test_telegrapher_operator(self):
        """Test telegrapher (damped wave) operator."""
        u = np.random.rand(self.n)
        u_t = np.random.rand(self.n)
        gamma = 0.1

        du, du_tt = op_telegrapher(u, u_t, self.L, gamma)

        self.assertEqual(du.shape, (self.n,))
        self.assertEqual(du_tt.shape, (self.n,))
        # du should equal u_t
        self.assertTrue(np.allclose(du, u_t))

    def test_fokker_planck_operator(self):
        """Test Fokker-Planck operator."""
        p = np.random.rand(self.n)
        p = p / np.sum(p)
        v_drift = np.random.randn(self.n) * 0.1
        D = 0.1

        dp = op_fokker_planck(p, v_drift, self.L, self.A, D)
        self.assertEqual(dp.shape, (self.n,))

    def test_schrodinger_imag_operator(self):
        """Test imaginary-time Schrödinger operator."""
        V_loss = np.random.rand(self.n)
        dpsi = op_schrodinger_imag(self.psi, V_loss, self.L)
        self.assertEqual(dpsi.shape, (self.n,))

    def test_cahn_hilliard_operator(self):
        """Test Cahn-Hilliard operator."""
        u = np.random.randn(self.n) * 0.5
        epsilon = 0.1
        du = op_cahn_hilliard(u, self.L, epsilon)
        self.assertEqual(du.shape, (self.n,))

    def test_fisher_kpp_operator(self):
        """Test Fisher-KPP operator."""
        u = np.random.rand(self.n)
        du = op_fisher_kpp(u, self.L)
        self.assertEqual(du.shape, (self.n,))

    def test_poisson_solver(self):
        """Test Poisson solver."""
        rho = np.random.rand(self.n)
        rho = rho - np.mean(rho)  # Zero mean for solvability
        phi = op_poisson(rho, self.L)
        self.assertEqual(phi.shape, (self.n,))


class TestRegulatorModes(unittest.TestCase):
    """Test the three regulator modes."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dim = 32
        self.n = 200

        # Generate clustered data
        centers = np.random.randn(4, self.dim)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        self.data = []
        self.labels = []
        for i, c in enumerate(centers):
            pts = c + np.random.randn(50, self.dim) * 0.2
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            self.data.append(pts)
            self.labels.extend([i] * 50)

        self.data = np.vstack(self.data).astype(np.float32)
        self.labels = np.array(self.labels)

        self.params = RegulatorParams(T_explore=15, T_select=15, tau=0.2)

    def test_retrieval_mode(self):
        """Test retrieval mode."""
        engine = UnifiedRegulatorEngine(params=self.params)
        engine.build_index(self.data, k=10)

        # Query from cluster 0
        query = self.data[0] + np.random.randn(self.dim) * 0.05
        query = query / np.linalg.norm(query)

        result = engine.retrieve(query, k=10)

        self.assertEqual(result.mode, Mode.RETRIEVAL)
        self.assertIsInstance(result.output, list)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)

    def test_clustering_mode(self):
        """Test clustering mode."""
        engine = UnifiedRegulatorEngine(params=self.params)

        result = engine.cluster(self.data)

        self.assertEqual(result.mode, Mode.CLUSTERING)
        self.assertIsInstance(result.output, np.ndarray)
        self.assertEqual(len(result.output), self.n)
        self.assertGreater(result.metadata["n_clusters"], 0)

    def test_decision_mode(self):
        """Test decision mode."""
        candidates = self.data[:10]
        V_loss = np.ones(10) * 0.5
        V_loss[3] = 0.1  # Make one clearly better

        engine = UnifiedRegulatorEngine(params=self.params)
        result = engine.decide(candidates, V_loss=V_loss)

        self.assertEqual(result.mode, Mode.DECISION)
        self.assertIsInstance(result.output, int)
        self.assertGreaterEqual(result.output, 0)
        self.assertLess(result.output, 10)


class TestConfidenceAndRefusal(unittest.TestCase):
    """Test confidence computation and refusal logic."""

    def test_refusal_on_low_confidence(self):
        """Test that system refuses when confidence is below threshold."""
        np.random.seed(42)
        candidates = np.random.randn(10, 32).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Uniform loss = no clear winner = low confidence
        V_loss = np.ones(10) * 0.5

        params = RegulatorParams(T_explore=20, T_select=20, tau=0.5)  # High threshold
        engine = UnifiedRegulatorEngine(params=params)

        result = engine.decide(candidates, V_loss=V_loss)

        # Should have low confidence (< tau)
        self.assertTrue(result.refused or result.confidence >= 0.5)

    def test_high_confidence_with_clear_signal(self):
        """Test high confidence when signal is clear."""
        np.random.seed(42)
        candidates = np.random.randn(10, 32).astype(np.float32)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Clear winner
        V_loss = np.ones(10) * 0.9
        V_loss[0] = 0.0  # Very clear winner

        params = RegulatorParams(T_explore=30, T_select=30, tau=0.15)
        engine = UnifiedRegulatorEngine(params=params)

        result = engine.decide(candidates, V_loss=V_loss)

        # Should pick the winner
        self.assertEqual(result.output, 0)
        # Should not refuse
        self.assertFalse(result.refused)


class TestUnifiedSemantics(unittest.TestCase):
    """Test that all modes share unified confidence semantics."""

    def test_confidence_is_mass_fraction(self):
        """Test that confidence = mass_i / total_mass in all modes."""
        np.random.seed(42)
        data = np.random.randn(50, 16).astype(np.float32)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        params = RegulatorParams(T_explore=10, T_select=10)
        engine = UnifiedRegulatorEngine(params=params)
        engine.build_index(data, k=5)

        # All results should have confidence in [0, 1]
        query = data[0]
        retrieval_result = engine.retrieve(query, k=5)
        self.assertGreaterEqual(retrieval_result.confidence, 0)
        self.assertLessEqual(retrieval_result.confidence, 1)

        cluster_result = engine.cluster()
        self.assertGreaterEqual(cluster_result.confidence, 0)
        self.assertLessEqual(cluster_result.confidence, 1)

        decision_result = engine.decide(data[:10])
        self.assertGreaterEqual(decision_result.confidence, 0)
        self.assertLessEqual(decision_result.confidence, 1)

        # Confidence per item should sum close to 1
        if len(retrieval_result.confidence_per_item) > 0:
            self.assertAlmostEqual(
                np.sum(retrieval_result.confidence_per_item), 1.0, places=2
            )


class TestCascade(unittest.TestCase):
    """Test cascaded mode execution."""

    def test_retrieval_then_decision(self):
        """Test cascading retrieval → decision."""
        np.random.seed(42)
        data = np.random.randn(100, 32).astype(np.float32)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        params = RegulatorParams(T_explore=10, T_select=10, tau=0.1)
        engine = UnifiedRegulatorEngine(params=params)
        engine.build_index(data, k=5)

        query = data[0]
        results = engine.cascade(query, [Mode.RETRIEVAL, Mode.DECISION])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].mode, Mode.RETRIEVAL)
        self.assertEqual(results[1].mode, Mode.DECISION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
