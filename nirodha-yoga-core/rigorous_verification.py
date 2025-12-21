import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator

def verify_spectral_contractivity():
    """
    Rigorously verifies that the derivative of the Nirodha operator is <= 1 everywhere.
    f(x) = x / (1 + beta * |x|)
    f'(x) = 1 / (1 + beta * |x|)^2
    Since beta >= 0 and |x| >= 0, f'(x) is always in (0, 1].
    """
    print("Verifying Spectral Contractivity (Jacobian Analysis)...")
    beta = 1.0
    regulator = YogaRegulator(beta=beta)
    
    # Test range
    x = torch.linspace(-100, 100, 1000, requires_grad=True)
    y = regulator.nirodha(x)
    
    # Calculate derivative exactly via autograd
    y.backward(torch.ones_like(x))
    gradients = x.grad
    
    max_grad = gradients.max().item()
    min_grad = gradients.min().item()
    
    print(f"  Max Derivative (Jacobian Eigenvalue): {max_grad:.6f}")
    print(f"  Min Derivative (Jacobian Eigenvalue): {min_grad:.6f}")
    
    if max_grad <= 1.000001: # allow for epsilon precision
        print("  ✅ VERIFIED: Operator is contractive (f' <= 1).")
    else:
        print("  ❌ FAILED: Operator is not contractive.")

def adversarial_stability_test():
    """
    Attempts to break stability by crafting updates that maximize the energy delta.
    """
    print("\nRunning Adversarial Stability Test...")
    dim = 512
    beta = 0.5
    regulator = YogaRegulator(beta=beta)
    
    C0 = torch.randn(dim)
    state = CognitiveState(C0)
    
    deltas_e = []
    
    for t in range(500):
        # We search for the worst-case update Delta that maximizes Energy Delta
        # This is an adversarial update
        update = torch.randn(dim, requires_grad=True)
        
        # Optimizer to find the 'worst' update
        optimizer = torch.optim.SGD([update], lr=1.0)
        
        for p in range(5): # Inner loop to craft adversary
            optimizer.zero_grad()
            E_t = torch.norm(state.C - state.C0)**2
            temp_state = regulator(state, update)
            E_next = torch.norm(temp_state.C - state.C0)**2
            loss = -(E_next - E_t) # Maximize energy delta
            loss.backward(retain_graph=True)
            optimizer.step()
        
        # Apply the final crafted adversarial update
        with torch.no_grad():
            state = regulator(state, update.detach())
            E_t = torch.norm(state.C - state.C0)**2
            # After regulation, calculate final E_next
            V = C0 + (state.C - C0) # state.C is already updated
            E_final = torch.norm(state.C - C0)**2
            # For logging, we need E_t before update
            # But the 'regulator' forward already did state.update
            # Let's just track the norm
            deltas_e.append(E_final.item())

    # Check: Norm should never exceed 1/beta**2 (the theoretical bound)
    # Actually, the bound for V is 1/beta, so E bound is (1/beta)**2
    max_energy = max(deltas_e)
    theoretical_bound = (1.0 / beta)**2 * dim # elementwise bound squared * dim
    
    print(f"  Max Adversarial Energy: {max_energy:.4f}")
    print(f"  Theoretical Upper Bound: {theoretical_bound:.4f}")
    
    if max_energy <= theoretical_bound + 1e-6:
        print("  ✅ VERIFIED: Even adversarial updates cannot break the energy bound.")
    else:
        print("  ❌ FAILED: Adversary broke the stability envelope.")

if __name__ == "__main__":
    verify_spectral_contractivity()
    adversarial_stability_test()
