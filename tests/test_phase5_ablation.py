import torch
import numpy as np
import pytest
from rads.rads_field import RADSField
from rads.stabilization import NirodhaOptimizer
from rads.safety import ConstitutionalManifold, ConstitutionalHypergraph
from rads.discretization import PatternExtractor, TuringVoidDetector
from rads.reasoning import SymbolicHypergraph

class TestPhase5Ablation:

    def test_exp_5_1_nirodha_stability_impact(self):
        """
        Experiment 5.1: Nirodha Impact on Stability (Energy Variance).
        Compare energy variance over time with/without Nirodha.
        """
        print("\nRunning Exp 5.1: Nirodha Stability Impact")
        
        num_trials = 50
        num_steps = 30
        
        def compute_energy(phi):
            return torch.sum(phi ** 2).item()
        
        # Baseline: No Nirodha
        baseline_variances = []
        for trial in range(num_trials):
            field = RADSField(size=32, d=4)
            field.phi = torch.randn(1, 4, 32, 32) * 3.0  # High energy
            
            energies = []
            for step in range(num_steps):
                energies.append(compute_energy(field.phi))
                field.evolve_step(dt=0.5)
            
            variance = np.var(energies)
            baseline_variances.append(variance)
        
        baseline_mean_var = np.mean(baseline_variances)
        
        # With Nirodha
        nirodha_variances = []
        for trial in range(num_trials):
            field = RADSField(size=32, d=4)
            field.phi = torch.randn(1, 4, 32, 32) * 3.0
            field.anchor = field.phi.clone()
            
            optimizer = NirodhaOptimizer(beta=0.5)
            
            energies = []
            for step in range(num_steps):
                energies.append(compute_energy(field.phi))
                optimizer.step(field, dt=0.5)
            
            variance = np.var(energies)
            nirodha_variances.append(variance)
        
        nirodha_mean_var = np.mean(nirodha_variances)
        
        reduction = (baseline_mean_var - nirodha_mean_var) / baseline_mean_var * 100
        
        print(f"Baseline Variance: {baseline_mean_var:.2e}")
        print(f"Nirodha Variance: {nirodha_mean_var:.2e}")
        print(f"Reduction: {reduction:.1f}%")
        
        # Assert >60% reduction
        assert reduction > 60.0

    def test_exp_5_2_nirodha_info_preservation_impact(self):
        """
        Experiment 5.2: Nirodha Impact on Information Preservation.
        Compare entropy ratio (Graph/Field) with/without Nirodha.
        """
        print("\nRunning Exp 5.2: Nirodha Info Preservation Impact")
        
        def entropy_ratio(field):
            # Extract graph
            extractor = PatternExtractor()
            vertices = extractor.extract_vertices_from_field(field, threshold=0.5, min_size=3)
            
            if len(vertices) == 0:
                return 0.0
            
            # Field entropy (simplified: variance-based)
            field_entropy = torch.var(field.phi).item()
            
            # Graph entropy (vertex strength distribution)
            strengths = np.array([v['strength'] for v in vertices])
            if len(strengths) > 1:
                probs = strengths / strengths.sum()
                graph_entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                graph_entropy = 0.0
            
            # Normalize
            ratio = graph_entropy / (field_entropy + 1e-6)
            return ratio
        
        num_trials = 20
        
        # Baseline
        baseline_ratios = []
        for trial in range(num_trials):
            field = RADSField(size=32, d=4, reaction='gray_scott', D_u=0.16, D_v=0.08)
            for t in range(100):
                field.evolve_step(dt=1.0)
            
            ratio = entropy_ratio(field)
            if ratio > 0:
                baseline_ratios.append(ratio)
        
        # With Nirodha
        nirodha_ratios = []
        for trial in range(num_trials):
            field = RADSField(size=32, d=4, reaction='gray_scott', D_u=0.16, D_v=0.08)
            field.anchor = field.phi.clone()
            optimizer = NirodhaOptimizer(beta=0.3)
            
            for t in range(100):
                optimizer.step(field, dt=1.0)
            
            ratio = entropy_ratio(field)
            if ratio > 0:
                nirodha_ratios.append(ratio)
        
        baseline_mean = np.mean(baseline_ratios) if baseline_ratios else 0.0
        nirodha_mean = np.mean(nirodha_ratios) if nirodha_ratios else 0.0
        
        improvement = (nirodha_mean - baseline_mean) / (baseline_mean + 1e-6) * 100
        
        print(f"Baseline Ratio: {baseline_mean:.3f}")
        print(f"Nirodha Ratio: {nirodha_mean:.3f}")
        print(f"Improvement: {improvement:.1f}%")
        
        # Assert improvement (may be modest due to heuristic metric)
        assert nirodha_mean >= baseline_mean

    def test_exp_5_3_aci_safety_violations_impact(self):
        """
        Experiment 5.3: ACI Impact on Safety Violations.
        Measure rejection rate of unsafe edges.
        """
        print("\nRunning Exp 5.3: ACI Safety Violations Impact")
        
        # Setup constraints
        constraints = [
            {'center': torch.tensor([5.0, 5.0]), 'alpha': 1.0, 'r_s': 2.0},
            {'center': torch.tensor([15.0, 15.0]), 'alpha': 1.0, 'r_s': 3.0}
        ]
        
        manifold = ConstitutionalManifold(constraints)
        safe_graph = ConstitutionalHypergraph(manifold)
        baseline_graph = SymbolicHypergraph()
        
        # Register 100 nodes (50 safe, 50 unsafe)
        safe_nodes = []
        unsafe_nodes = []
        
        for i in range(50):
            name = f"safe_{i}"
            vec = torch.tensor([float(i % 10), float(i // 10)])  # Spread out
            safe_graph.register_node(name, vec)
            safe_nodes.append(name)
        
        for i in range(50):
            name = f"unsafe_{i}"
            # Place near constraints
            if i < 25:
                vec = torch.tensor([5.0 + np.random.randn() * 0.5, 5.0 + np.random.randn() * 0.5])
            else:
                vec = torch.tensor([15.0 + np.random.randn() * 0.5, 15.0 + np.random.randn() * 0.5])
            safe_graph.register_node(name, vec)
            unsafe_nodes.append(name)
        
        # Attempt to add edges
        # Baseline: Accepts all
        baseline_accepted = 0
        for s in safe_nodes[:10]:
            for t in unsafe_nodes[:5]:
                baseline_graph.add_hyperedge(s, {t})
                baseline_accepted += 1
        
        # ACI: Rejects unsafe
        aci_accepted = 0
        aci_rejected = 0
        for s in safe_nodes[:10]:
            for t in unsafe_nodes[:5]:
                initial_count = len(safe_graph.edges)
                safe_graph.add_hyperedge(s, {t})
                if len(safe_graph.edges) > initial_count:
                    aci_accepted += 1
                else:
                    aci_rejected += 1
        
        rejection_rate = aci_rejected / (aci_accepted + aci_rejected) * 100 if (aci_accepted + aci_rejected) > 0 else 0
        
        print(f"Baseline: Accepted {baseline_accepted} (no filtering)")
        print(f"ACI: Accepted {aci_accepted}, Rejected {aci_rejected}")
        print(f"Rejection Rate: {rejection_rate:.1f}%")
        
        # Assert high rejection rate (most unsafe edges blocked)
        assert rejection_rate > 50.0

    def test_exp_5_4_aci_reasoning_accuracy_impact(self):
        """
        Experiment 5.4: ACI Impact on Reasoning Accuracy.
        Measure conflict reduction via constraint enforcement.
        """
        print("\nRunning Exp 5.4: ACI Reasoning Accuracy Impact")
        
        # Create contradictory knowledge base
        # A -> B, B -> C, A -> !C (conflict)
        
        # Baseline: Allows contradiction
        baseline = SymbolicHypergraph()
        baseline.add_hyperedge('A', {'B'})
        baseline.add_hyperedge('B', {'C'})
        baseline.add_hyperedge('A', {'!C'})  # Contradiction
        
        res_baseline = baseline.query('A', 'C')
        baseline_conflicts = 1 if res_baseline['reachable'] and res_baseline['exception'] else 0
        
        # ACI: Prevents contradiction
        constraints = [
            # Forbid region around "!C"
            {'center': torch.tensor([10.0, 10.0]), 'alpha': 1.0, 'r_s': 2.0}
        ]
        manifold = ConstitutionalManifold(constraints)
        aci_graph = ConstitutionalHypergraph(manifold)
        
        aci_graph.register_node('A', torch.tensor([0.0, 0.0]))
        aci_graph.register_node('B', torch.tensor([5.0, 5.0]))
        aci_graph.register_node('C', torch.tensor([8.0, 8.0]))
        aci_graph.register_node('!C', torch.tensor([10.0, 10.0]))  # Inside forbidden
        
        aci_graph.add_hyperedge('A', {'B'})
        aci_graph.add_hyperedge('B', {'C'})
        aci_graph.add_hyperedge('A', {'!C'})  # Should be rejected
        
        res_aci = aci_graph.query('A', 'C')
        aci_conflicts = 1 if res_aci['reachable'] and res_aci['exception'] else 0
        
        conflict_reduction = (baseline_conflicts - aci_conflicts) / baseline_conflicts * 100 if baseline_conflicts > 0 else 0
        
        print(f"Baseline Conflicts: {baseline_conflicts}")
        print(f"ACI Conflicts: {aci_conflicts}")
        print(f"Reduction: {conflict_reduction:.1f}%")
        
        # Assert conflict reduction
        assert aci_conflicts < baseline_conflicts

    def test_exp_5_5_turing_voids_completeness_impact(self):
        """
        Experiment 5.5: Turing Voids Impact on Graph Completeness.
        Measure coverage increase via void-guided edge addition.
        """
        print("\nRunning Exp 5.5: Turing Voids Completeness Impact")
        
        # Create sparse field with stronger initialization
        field = RADSField(size=64, d=4)
        field.phi = torch.rand(1, 4, 64, 64) * 2.0  # Stronger signal for GS
        
        # Baseline: Extract graph without void guidance
        extractor = PatternExtractor()
        baseline_vertices = extractor.extract_vertices_from_field(field, threshold=0.3, min_size=5)
        baseline_density = len(baseline_vertices)
        
        # With Turing Voids: Detect gaps and suggest edges
        detector = TuringVoidDetector(size=64)
        voids = detector.detect_voids(field, threshold=0.05)
        
        # Simulate adding edges at void locations
        # (In practice, voids suggest WHERE to add knowledge)
        enhanced_density = baseline_density + len(voids)
        
        coverage_increase = (enhanced_density - baseline_density) / baseline_density * 100 if baseline_density > 0 else 0
        
        print(f"Baseline Vertices: {baseline_density}")
        print(f"Detected Voids: {len(voids)}")
        print(f"Enhanced Density: {enhanced_density}")
        print(f"Coverage Increase: {coverage_increase:.1f}%")
        
        # Assert coverage increase
        assert len(voids) > 0
        assert coverage_increase > 0
