import torch
from nirodha.core import Nirodha

def test_nirodha_output():
    beta = 0.5
    nirodha = Nirodha(beta=beta)
    
    # Test values
    v_values = torch.tensor([0.0, 0.1, 1.0, 10.0, 100.0, -10.0])
    v_suppressed = nirodha(v_values)
    
    print(f"Beta: {beta}")
    for v, vs in zip(v_values, v_suppressed):
        print(f"V: {v:7.2f} -> V_suppressed: {vs:7.2f} (reduction: {v - vs:7.2f})")

if __name__ == "__main__":
    test_nirodha_output()
