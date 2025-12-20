
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from energy_monotone import EnergyMonotoneTransformer
import numpy as np
import matplotlib.pyplot as plt
import os

def run_demo():
    print("Initializing Energy-Monotone Transformer Demo...")
    
    # Hyperparameters
    vocab_size = 100 # Reduced from 1000
    dim = 256
    num_layers = 4
    num_heads = 8
    seq_len = 32 # Reduced from 64
    batch_size = 32
    steps = 500 # Increased from 100
    
    # Model init
    model = EnergyMonotoneTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
        tau=1.0,
        lambda_jump=0.6,
        lambda_local=0.4,
        min_entropy=0.5
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # Increased LR
    
    # Dummy data generator (Copy task: repeat first half of sequence)
    def get_batch():
        data = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Make it a copy task to verify learning capability
        # First half random, second half = first half
        half = seq_len // 2
        data[:, half:] = data[:, :half]
        return data

    losses = []
    energy_profiles = [] # To store energy values per layer over training
    
    print(f"Starting training for {steps} steps on Copy Task (vocab={vocab_size})...")
    model.train()
    
    for step in range(steps):
        inputs = get_batch()
        targets = inputs.clone()
        
        # Forward pass
        logits, energies = model(inputs, return_energies=True)
        
        # Standard CLM objective
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check grad norms
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, Grad Norm = {total_norm:.4f}")
            # Store energy profile to visualize monotonicity
            energy_profiles.append(energies)

    print("Training complete.")
    
    # Verify Energy Monotonicity on final model
    print("\nVerifying Energy Monotonicity on evaluation batch:")
    model.eval()
    with torch.no_grad():
        test_inputs = get_batch()
        _, final_energies = model(test_inputs, return_energies=True)
        
        print(f"Layer Energies: {[f'{e:.4f}' for e in final_energies]}")
        monotone = all(final_energies[i+1] <= final_energies[i] + 1e-5 for i in range(len(final_energies)-1))
        print(f"Monotonicity Satisfied: {'YES' if monotone else 'NO'}")
        
    return losses, energy_profiles

if __name__ == "__main__":
    losses, energy_profiles = run_demo()
    
    # Optional: could plot results if needed
    # plt.plot(losses)
    # plt.savefig('loss_curve.png')
