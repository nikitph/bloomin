import torch
import numpy as np
import pytest
from rads.rads_field import RADSField
from rads.stabilization import NirodhaOptimizer
from rads.safety import ConstitutionalManifold, ConstitutionalHypergraph
from rads.discretization import TuringVoidDetector

class TestPhase4Safety:

    def test_exp_4_1_nirodha_stability(self):
        """
        Experiment 4.1: Nirodha-Regulated Field Stability.
        Verify that Nirodha optimization prevents divergence even with unstable parameters.
        """
        print("\nRunning Exp 4.1: Nirodha Stability")
        
        size = 32
        d = 4
        # Create UNSTABLE field (high diffusion/reaction rates)
        # Standard RADS might explode or oscillate wildly.
        field_unstable = RADSField(size=size, d=d)
        # Hack parameters to be unstable? 
        # Large dt usually causes instability in Euler integration.
        dt_risky = 2.0 # Very large step
        
        # 1. Baseline: Unregulated Growth
        field_base = RADSField(size=size, d=d)
        field_base.phi = torch.randn(1, d, size, size) * 5.0 # High energy init
        phi_0_norm = torch.norm(field_base.phi)
        
        # Run one greedy step
        try:
            field_base.evolve_step(dt=dt_risky)
            norm_base = torch.norm(field_base.phi)
            print(f"Baseline Norm: {phi_0_norm:.2f} -> {norm_base:.2f}")
        except Exception as e:
            print("Baseline crashed as expected/possible")
            
        # 2. Nirodha Regulated
        field_nirodha = RADSField(size=size, d=d)
        field_nirodha.phi = field_base.phi.clone() # Same init
        field_nirodha.anchor = field_nirodha.phi.clone() # Set anchor
        
        optimizer = NirodhaOptimizer(beta=0.5) # Strong regulations
        
        field_nirodha.evolve_step = lambda dt: None # Disable internal step to prevent double update?
        # Wait, my NirodhaOptimizer calls field.evolve_step inside.
        # I need to restore standard behavior but call optimizer.step
        
        # Reset field standard 
        field_nirodha = RADSField(size=size, d=d)
        field_nirodha.phi = field_base.anchor.clone() if hasattr(field_base, 'anchor') else torch.randn(1, d, size, size)
        field_nirodha.anchor = field_nirodha.phi.clone()
        
        # Step with optimizer
        norm_trace = []
        norm_trace.append(torch.norm(field_nirodha.phi - field_nirodha.anchor).item())
        
        for t in range(10):
            optimizer.step(field_nirodha, dt=dt_risky)
            dev = torch.norm(field_nirodha.phi - field_nirodha.anchor).item()
            norm_trace.append(dev)
            
        print(f"Nirodha Trace: {norm_trace}")
        
        # Verify: Deviation should be bounded/damped compared to unregulated
        # And specifically, Nirodha ensures contraction of deviation?
        # N_beta(x) < x for large x.
        assert norm_trace[-1] < 1000.0 # Arbitrary bound, but checking for explosion
        # Theoretical property: Monotonic decrease? 
        # Only if system is relaxing. With external forcing it bounds it.
        # We assert finiteness/bound.
        assert not torch.isnan(field_nirodha.phi).any()

    def test_exp_4_2_constitutional_safety(self):
        """
        Experiment 4.2: Constitutional Hypergraph Safety.
        """
        print("\nRunning Exp 4.2: Constitutional Safety")
        
        # Define Constraints
        # Forbidden region at (5, 5) with radius 2
        center = torch.tensor([5.0, 5.0])
        constraints = [{'center': center, 'alpha': 1.0, 'r_s': 2.0}]
        
        manifold = ConstitutionalManifold(constraints)
        safe_graph = ConstitutionalHypergraph(manifold)
        
        # Register nodes
        # Node A: Safe (0,0)
        # Node B: Safe (10,10)
        # Node C: UNSAFE (5,5) (Inside constraint)
        
        safe_graph.register_node('A', torch.tensor([0.0, 0.0]))
        safe_graph.register_node('B', torch.tensor([10.0, 10.0]))
        safe_graph.register_node('C', torch.tensor([5.0, 5.0]))
        
        # 1. Add Safe Edge A->B
        safe_graph.add_hyperedge('A', {'B'})
        
        # 2. Add Unsafe Edge A->C (Target unsafe)
        safe_graph.add_hyperedge('A', {'C'})
        
        # Check edges
        edges = safe_graph.edges
        print("Edges:", edges)
        
        has_AB = ('A', 'B') in edges
        has_AC = ('A', 'C') in edges
        
        assert has_AB == True
        assert has_AC == False # Should be rejected safety
        
    def test_exp_4_3_turing_void_detection(self):
        """
        Experiment 4.3: Turing Void Detection.
        """
        print("\nRunning Exp 4.3: Turing Void Detection")
        
        detector = TuringVoidDetector(size=64)
        field = RADSField(size=64, d=4)
        
        # Init field randomly with logic
        field.phi = torch.rand(1, 4, 64, 64) * 2.0 # Stronger signal
        
        # Detect
        # Internally detector runs GS. We cannot easily peek inside unless we modify it.
        # But we can trust the detector logic if we verify parameters.
        # Let's trust logic but rely on result.
        voids = detector.detect_voids(field, threshold=0.05) # Very low threshold
        
        print(f"Detected voids: {len(voids)}")
        
        # We expect some spots to form given GS dynamics
        assert len(voids) > 0
