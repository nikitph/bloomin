import torch
from nirodha.core import Nirodha

def rewa_yoga_demo():
    print("REWA-Yoga Integration Demo")
    
    # REWA context: Witness aggregate and activation
    # Assume witness_anchor is the baseline state of the witness field
    witness_anchor = torch.randn(1024)
    
    # New witness activation (potentially unstable/explosive)
    witness = witness_anchor + torch.randn(1024) * 5.0
    print(f"Initial witness anomaly norm: {torch.norm(witness - witness_anchor):.4f}")
    
    # Nirodha operator
    nirodha = Nirodha(beta=0.8)
    
    # Mapping Yoga to REWA:
    # witness_energy = witness - witness_anchor (Vritti)
    # witness_energy = nirodha(witness_energy)     (Nirodha)
    # witness = witness_anchor + witness_energy    (Full Reconstruction)
    
    witness_energy = witness - witness_anchor
    witness_energy_suppressed = nirodha(witness_energy)
    
    witness_regulated = witness_anchor + witness_energy_suppressed
    
    print(f"Regulated witness anomaly norm: {torch.norm(witness_regulated - witness_anchor):.4f}")
    
    # Show suppression effect
    reduction_ratio = torch.norm(witness_energy_suppressed) / torch.norm(witness_energy)
    print(f"Energy suppression ratio: {reduction_ratio:.4f}")

if __name__ == "__main__":
    rewa_yoga_demo()
