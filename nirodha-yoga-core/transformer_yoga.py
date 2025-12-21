import torch
import torch.nn as nn
from nirodha.core import CognitiveState, YogaRegulator

def transformer_yoga_demo():
    print("Transformer Yoga-Regulated Layer Demo")
    
    # Mock parameters
    batch_size = 1
    seq_len = 8
    d_model = 512
    beta = 1.0
    
    # Mock residual stream h
    h = torch.randn(batch_size, seq_len, d_model)
    print(f"Initial h norm: {torch.norm(h):.4f}")
    
    # Initialize Regulator
    yoga_regulator = YogaRegulator(beta=beta)
    
    # Traditional Residual Stream: h = h + attention(h)
    # Yoga-Regulated:
    
    # 1. Wrap current state
    state = CognitiveState(h)
    
    # 2. Compute update
    # Mocking attention(h) output
    delta = torch.randn(batch_size, seq_len, d_model) * 1.5
    print(f"Update delta norm: {torch.norm(delta):.4f}")
    
    # 3. Apply regulation
    state = yoga_regulator(state, delta)
    
    # 4. Extract new latent state
    h_new = state.C
    print(f"Regulated h norm: {torch.norm(h_new):.4f}")
    
    # Comparison with unregulated update
    h_unregulated = h + delta
    print(f"Unregulated h norm: {torch.norm(h_unregulated):.4f}")
    
    diff_regulated = torch.norm(h_new - h)
    diff_unregulated = torch.norm(h_unregulated - h)
    
    print(f"\nRegulated change norm: {diff_regulated:.4f}")
    print(f"Unregulated change norm: {diff_unregulated:.4f}")

if __name__ == "__main__":
    transformer_yoga_demo()
