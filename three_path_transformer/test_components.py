import unittest
import torch
import numpy as np
from model import ThreePathTransformer
from memory import ConceptMemory
from data_generation import generate_color_dataset
from utils import set_seed

class TestThreePathComponents(unittest.TestCase):
    
    def setUp(self):
        set_seed(42)
        self.vocab_size = 20
        self.dim = 32
        self.model = ThreePathTransformer(self.vocab_size, dim=self.dim, n_layers=2)
        self.memory = ConceptMemory()
        
    def test_model_shapes(self):
        # Input: [batch=2, seq=1] (just concept IDs)
        x = torch.randint(0, self.vocab_size, (2, 1))
        
        # Fast Path
        out_fast = self.model.encode(x, path='fast')
        self.assertEqual(out_fast.shape, (2, self.dim))
        # Check probability property (sum approx 1)
        self.assertTrue(torch.allclose(out_fast.sum(dim=-1), torch.ones(2), atol=1e-5))
        
        # Slow Path
        out_slow = self.model.encode(x, path='slow')
        self.assertEqual(out_slow.shape, (2, self.dim))
        self.assertTrue(torch.allclose(out_slow.sum(dim=-1), torch.ones(2), atol=1e-5))
        
    def test_mixing(self):
        emb1 = torch.rand(2, self.dim)
        emb2 = torch.rand(2, self.dim)
        
        mixed = self.model.mix(emb1, emb2, method='geometric')
        self.assertEqual(mixed.shape, (2, self.dim))
        
    def test_memory_storage(self):
        emb = torch.randn(self.dim)
        self.memory.store('TestConcept', emb, generation=0)
        
        retrieved = self.memory.retrieve('TestConcept')
        self.assertTrue(torch.equal(emb, retrieved))
        
        hierarchy = self.memory.get_hierarchy(0)
        self.assertIn('TestConcept', hierarchy)
        
    def test_sharpness_measure(self):
        # Sharp distro
        sharp = torch.zeros(self.dim)
        sharp[0] = 1.0
        self.memory.store('Sharp', sharp)
        
        # Flat distro
        flat = torch.ones(self.dim) / self.dim
        self.memory.store('Flat', flat)
        
        stats = self.memory.measure_sharpness()
        # Flat entropy = ln(32) approx 3.4
        # Sharp entropy = 0
        # Mean approx 1.7
        self.assertGreater(stats['mean_entropy'], 0)
        
    def test_data_generation(self):
        examples, concepts = generate_color_dataset(n_samples=10, dim=self.dim)
        self.assertEqual(len(examples), 13) # 10 mixed + 3 identity
        self.assertIn('Red', concepts)
        self.assertIn('Mauve', concepts)
        
        # Check ground truth geometry
        # Purple = sqrt(Red * Blue)
        p = concepts['Purple']
        r = concepts['Red']
        b = concepts['Blue']
        
        # Verify supports overlap
        # peak locs are different so overlap might be small, but should be geometric mean
        expected = torch.sqrt(r * b)
        expected = expected / expected.sum()
        
        # Using tolerance because of float precision
        self.assertTrue(torch.allclose(p, expected, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
