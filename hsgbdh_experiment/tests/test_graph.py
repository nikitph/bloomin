import torch
import pytest
from hsgbdh.graph import BlockSparseGraph, differentiable_closure
from hsgbdh.semiring import AdaptiveSemiring

def test_block_sparse_add_edge():
    G = BlockSparseGraph(n=64, block_size=32)
    G.add_edge(0, 1, 0.5)
    G.add_edge(33, 34, 0.8)
    
    dense = G.to_dense()
    assert dense[0, 1] == 0.5
    assert dense[33, 34] == 0.8
    assert dense[10, 10] == 0.0

def test_differentiable_closure():
    semiring = AdaptiveSemiring()
    semiring.set_temperature(1e-6) # Hard for verifying logic
    
    # Simple chain: 0->1->2
    G = torch.zeros(3, 3)
    G[0, 1] = 1.0
    G[1, 2] = 1.0
    
    G_star = differentiable_closure(G, semiring, K=2)
    
    # Expected: 0->2 exists
    assert G_star[0, 2] == 1.0
    # 0->1 exists
    assert G_star[0, 1] == 1.0
