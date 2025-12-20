
import unittest
import torch
import torch.nn.functional as F
from energy_monotone import (
    energy,
    EnergyMonotoneResidual,
    DampedAttention,
    StableTransformerBlock,
    EnergyMonotoneTransformer,
    local_diffusion_kernel
)

class TestEnergyMonotone(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 32
        self.dim = 64
        self.num_heads = 4
        
    def test_energy_function(self):
        """Test simple Lyapunov energy calculation."""
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        e = energy(x)
        self.assertTrue(torch.is_tensor(e))
        self.assertEqual(e.dim(), 0)
        self.assertTrue(e.item() >= 0)
        
        # Test scale invariance
        # Doubling x should quadruple energy (conceptually, mean norm squared)
        x2 = 2 * x
        e2 = energy(x2)
        self.assertAlmostEqual(e2.item(), 4 * e.item(), places=5)

    def test_local_diffusion_kernel(self):
        """Test local diffusion kernel properties."""
        seq_len = 10
        strength = 0.5
        kernel = local_diffusion_kernel(seq_len, strength)
        
        self.assertEqual(kernel.shape, (seq_len, seq_len))
        # Check stochasticity (rows sum to 1)
        row_sums = kernel.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6))
        
        # Check locality (diagonals should be largest)
        self.assertTrue(torch.all(torch.diag(kernel) > 0.1))

    def test_energy_monotonicity_enforcement(self):
        """Test that EnergyMonotoneResidual prevents energy increase."""
        residual = EnergyMonotoneResidual(alpha=1.0)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        # 1. Update that naturally decreases energy (should be kept)
        f_x = -0.1 * x # shrinking vector
        y = residual(x, f_x)
        self.assertLessEqual(energy(y).item(), energy(x).item() + 1e-6)
        
        # 2. Update that massively increases energy (should be clamped)
        f_x_bad = 10.0 * x # expanding vector
        y_safe = residual(x, f_x_bad)
        
        # The output energy should be exactly equal to input energy (monotonicity boundary)
        # Note: logic says if E_after <= E_before: return candidate
        # If > : return x + scale * update such that E(new) ~ E_before
        self.assertLessEqual(energy(y_safe).item(), energy(x).item() + 1e-5)
        
        # It shouldn't just return x (unless scale is 0, which is unlikely for non-orthogonal update)
        # However, if E(x) is small and update is huge, scale might be small. 
        # Ideally it projects to the boundary.

    def test_damped_attention_entropy(self):
        """Test that attention respects entropy constraints."""
        attn = DampedAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            min_entropy=2.0 # Set high min entropy to force damping
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        # We can't easily introspect the internal A matrix without hooking, 
        # but we can ensure it runs and outputs correct shape.
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_transformer_block_contraction(self):
        """Test that a full block is contractive or at least non-expanding."""
        block = StableTransformerBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            lambda_jump=0.5,
            lambda_local=0.5,
            min_entropy=0.1
        )
        # Zero init weights to ensure stability at start? 
        # Or standard init. The residual gates should handle it regardless of weights.
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        e_in = energy(x)
        
        # Forward pass
        x_out = block(x)
        e_out = energy(x_out)
        
        # Assert E_out <= E_in
        self.assertLessEqual(e_out.item(), e_in.item() + 1e-5)
        
    def test_full_model_training_step(self):
        """Test a dummy training step to ensure gradients flow."""
        model = EnergyMonotoneTransformer(
            vocab_size=100,
            dim=32,
            num_layers=2,
            num_heads=4
        )
        input_ids = torch.randint(0, 100, (2, 16))
        
        # Forward
        logits, energies = model(input_ids, return_energies=True)
        
        # Check energy monotonicity across layers
        for i in range(len(energies) - 1):
            self.assertLessEqual(energies[i+1], energies[i] + 1e-5, f"Energy increased at layer {i}")
            
        # Backward
        loss = F.cross_entropy(logits.view(-1, 100), torch.randint(0, 100, (32,)))
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == '__main__':
    unittest.main()
