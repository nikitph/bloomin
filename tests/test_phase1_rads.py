import torch
import numpy as np
import pytest
import matplotlib.pyplot as plt
from rads.rads_field import RADSField

class TestPhase1RADS:

    def test_exp_1_1_lyapunov_stability(self):
        """
        Experiment 1.1: Prove continuous evolution is stable via Lyapunov function.
        """
        print("\nRunning Exp 1.1: Lyapunov Stability")
        field = RADSField(size=50, d=16, D=0.1) # Smaller size for speed
        
        # Random init
        field.phi = torch.randn(1, 16, 50, 50) 
        
        # Energy functional E = sum((grad phi)^2)
        def compute_energy(phi):
            gx = field.gradient_x(phi)
            gy = field.gradient_y(phi)
            return torch.sum(gx**2 + gy**2).item()
        
        energies = []
        energies.append(compute_energy(field.phi))
        
        # Evolve
        for t in range(50):
            field.step(dt=0.01)
            energies.append(compute_energy(field.phi))
            
        # Check simple monotonicity for diffusion (should decrease or stay same)
        # Note: slight discretization noise might cause tiny fluctuations, but trend should be down.
        # We allow small numerical epsilon.
        decreases = 0
        for i in range(len(energies)-1):
            if energies[i+1] <= energies[i] + 1e-6:
                decreases += 1
                
        # We expect nearly monotonic decrease
        print(f"Energy start: {energies[0]}, end: {energies[-1]}")
        assert energies[-1] < energies[0]
        assert decreases >= len(energies) * 0.9 # Allow few numerical blips
        
    def test_exp_1_2_turing_patterns(self):
        """
        Experiment 1.2: Turing Pattern Formation (Gray-Scott).
        """
        print("\nRunning Exp 1.2: Turing Patterns")
        # Reaction-diffusion params for spots/stripes
        field = RADSField(
            size=64, # 64x64 is enough for patterns
            d=2, 
            reaction='gray_scott',
            D_u=0.16,
            D_v=0.08
        )
        
        # Evolve
        # Needs many steps to form patterns
        for t in range(1000): # 1000 steps might be barely enough, but let's try
            field.step(dt=1.0)
            
        # extract stable patterns
        patterns = field.find_stable_regions(threshold=0.2) # Threshold for 'v'
        
        # Check we have some structure (not empty, not full)
        num_points = patterns.shape[0]
        total_points = 64 * 64
        fill_ratio = num_points / total_points
        
        print(f"Pattern fill ratio: {fill_ratio}")
        assert 0.01 < fill_ratio < 0.5 # patterns shouldn't fill everything or nothing
        
    def test_exp_1_3_advection_directionality(self):
        """
        Experiment 1.3: Advection Directionality.
        """
        print("\nRunning Exp 1.3: Advection")
        field = RADSField(size=100, d=1, v='rightward', D=0.0) # Pure advection if possible, or low D
        
        # Construct 2D field acting like 1D for test simplicity or just measure X-shift
        # Inject signal at left
        field.phi = torch.zeros(1, 1, 100, 100)
        field.phi[:, :, :, 10:20] = 1.0 # Vertical bar at x=10..20
        
        positions = []
        
        for t in range(20):
            # Center of mass in X
            mass_x = torch.sum(field.phi, dim=2) # sum over Y -> (1, 1, W)
            x_coords = torch.arange(100).float().view(1, 1, 100)
            center = torch.sum(mass_x * x_coords) / (torch.sum(mass_x) + 1e-8)
            positions.append(center.item())
            
            field.step(dt=0.5)
            
        # Verify rightward movement
        velocity = positions[-1] - positions[0]
        print(f"Advection displacement: {velocity}, Positions: {positions}")
        assert velocity > 2.0 # Should move right
        
    def test_exp_1_4_multi_scale_hierarchy(self):
        """
        Experiment 1.4: Multi-Scale Hierarchy.
        """
        print("\nRunning Exp 1.4: Multi-Scale")
        field = RADSField(size=32, d=16, num_scales=3)
        
        # Create checkerboard high freq pattern
        x = torch.arange(32).float().view(1, 32)
        y = torch.arange(32).float().view(32, 1)
        pattern = torch.sin(x) * torch.sin(y)
        field.phi[0, 0] = pattern
        
        scales = field.build_scale_hierarchy()
        
        assert len(scales) == 3
        # Check smoothness increases (variance decreases)
        # Scale 0 is original, Scale 1 is pooled, Scale 2 is pooled again
        
        vars = [np.var(s) for s in scales]
        print(f"Variances across scales: {vars}")
        
        # Pooling averages out high freq, so variance should decrease for this specific pattern?
        # Or at least smoothness metrics (gradient energy) should decrease.
        # Let's check gradient energy approximation (sum of diffs)
        
        def grad_energy(img):
            dy, dx = np.gradient(img)
            return np.sum(dx**2 + dy**2) / img.size # Normalize by size
            
        energies = [grad_energy(s) for s in scales]
        print(f"Gradient energies: {energies}")
        
        # Coarser scales = smoother = lower gradient energy density
        assert energies[1] < energies[0]
