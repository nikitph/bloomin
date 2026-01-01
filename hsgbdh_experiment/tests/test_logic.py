import torch
import pytest
from hsgbdh.logic import LogicHead, LogicGatedGraphUpdate
from hsgbdh.semiring import AdaptiveSemiring

def test_logic_head_consistency():
    d = 16
    head = LogicHead(d_model=d, num_logic_heads=1)
    
    q = torch.randn(d)
    k = torch.randn(d)
    
    trans_q, cons = head.forward(q, k, head_idx=0)
    assert -1.0 <= cons <= 1.0 # cosine sim should be -1 to 1

def test_logic_gated_update():
    d = 16
    head = LogicHead(d_model=d)
    semiring = AdaptiveSemiring()
    updater = LogicGatedGraphUpdate(head, semiring)
    
    x_i = torch.randn(1, d)
    x_j = torch.randn(1, d)
    
    # Mock high correlation
    bdh_corr = 0.9
    
    # We can't guarantee logic head approves random vectors, 
    # but we can check it runs without error and returns 0 or non-zero
    weight, idx = updater.propose_and_verify(x_i, x_j, bdh_corr)
    
    assert isinstance(weight, torch.Tensor)
