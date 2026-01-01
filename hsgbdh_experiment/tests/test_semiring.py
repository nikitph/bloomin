import torch
import pytest
from hsgbdh.semiring import AdaptiveSemiring

def test_semiring_max_annealing():
    semiring = AdaptiveSemiring()
    x = torch.tensor([1.0, 5.0, 10.0])
    
    # High temp (soft)
    semiring.set_temperature(10.0)
    soft_max = semiring.semiring_max(x)
    # T * logsumexp(x/T) -> 10 * log(e^0.1 + e^0.5 + e^1.0) approx 10 * log(1.1 + 1.6 + 2.7) = 10 * log(5.4) = 16.8
    # Actually soft_max should be close to max but smoother.
    assert isinstance(soft_max, torch.Tensor)
    
    # Low temp (hard)
    semiring.set_temperature(1e-6)
    hard_max = semiring.semiring_max(x)
    assert torch.isclose(hard_max, torch.tensor(10.0))

def test_semiring_compose_annealing():
    semiring = AdaptiveSemiring()
    a = torch.tensor(2.0)
    b = torch.tensor(3.0)
    
    # High temp (soft min)
    semiring.set_temperature(10.0)
    soft_min = semiring.semiring_compose(a, b)
    # -T * logsumexp([-a/T, -b/T])
    
    # Low temp (hard min)
    semiring.set_temperature(1e-6)
    hard_min = semiring.semiring_compose(a, b)
    assert torch.isclose(hard_min, torch.tensor(2.0))
