
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from energy_monotone import EnergyMonotoneTransformer
from baseline_transformer import StandardTransformer

def run_benchmark():
    print("Initializing Benchmark: Energy-Monotone vs Standard Transformer")
    
    # Common Config
    config = {
        'vocab_size': 100,
        'dim': 64,
        'num_layers': 2, # Reduced from 6
        'num_heads': 4,
        'seq_len': 32,
        'batch_size': 64,
        'steps': 300, # Reduced from 600
        'lr': 1e-3,
        'device': 'cpu' # Use CPU for simplicity/safety here
    }
    
    # Init Models
    model_ours = EnergyMonotoneTransformer(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['seq_len']
    )
    
    model_base = StandardTransformer(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['seq_len']
    )
    
    print(f"Params Ours: {model_ours.get_num_params():,}")
    print(f"Params Base: {model_base.get_num_params():,}")
    
    # Optimizers
    opt_ours = optim.AdamW(model_ours.parameters(), lr=config['lr'])
    opt_base = optim.AdamW(model_base.parameters(), lr=config['lr'])
    
    # Data Generator (Harder Copy Task: Reverse Copy)
    # Learning to reverse a sequence forces robust attention
    def get_batch():
        data = torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len']))
        # Task: First half is random. Second half is REVERSE of first half.
        half = config['seq_len'] // 2
        
        # Create reverse copy targets
        source = data[:, :half]
        target = torch.flip(source, dims=[1])
        data[:, half:] = target
        return data

    # Metrics
    history = {
        'ours': {'loss': [], 'grad_norm': [], 'energy_final_step': []},
        'base': {'loss': [], 'grad_norm': [], 'energy_final_step': []}
    }
    
    def train_step(model, optimizer, name):
        model.train()
        inputs = get_batch()
        targets = inputs.clone()
        
        # Forward
        logits, energies = model(inputs, return_energies=True)
        
        # Save energy profile occasionally (last step)
        history[name]['energy_final_step'] = energies
        
        # Loss (Predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, config['vocab_size']), shift_labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history[name]['loss'].append(loss.item())
        history[name]['grad_norm'].append(total_norm)
        
        return loss.item(), total_norm

    print("Starting Training Race...")
    for step in tqdm(range(config['steps'])):
        loss_o, grad_o = train_step(model_ours, opt_ours, 'ours')
        loss_b, grad_b = train_step(model_base, opt_base, 'base')
        
    # --- PLOTTING ---
    print("Generating Plots...")
    
    # 1. Loss Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(history['ours']['loss'], label='Energy-Monotone (Ours)', alpha=0.9)
    plt.plot(history['base']['loss'], label='Standard Baseline', alpha=0.7, linestyle='--')
    plt.title('Training Loss Convergence (Reverse Copy Task)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmark_loss.png')
    plt.close()
    
    # 2. Gradient Stability (Moving Average)
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(history['ours']['grad_norm'], 20), label='Energy-Monotone (Ours)', alpha=0.9)
    plt.plot(smooth(history['base']['grad_norm'], 20), label='Standard Baseline', alpha=0.7, linestyle='--')
    plt.title('Gradient Norm Stability (Smoothed)')
    plt.xlabel('Step')
    plt.ylabel('Grad Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmark_grads.png')
    plt.close()
    
    # 3. Layer-wise Energy Profile (Final Step)
    plt.figure(figsize=(8, 6))
    layers = list(range(len(history['ours']['energy_final_step'])))
    
    plt.plot(layers, history['ours']['energy_final_step'], 'o-', linewidth=2, label='Energy-Monotone (Ours)')
    plt.plot(layers, history['base']['energy_final_step'], 'x--', linewidth=2, label='Standard Baseline')
    
    # Add theoretical boundary line for Ours (non-increasing)
    plt.axhline(y=history['ours']['energy_final_step'][0], color='green', linestyle=':', alpha=0.5, label='Initial Energy Level (Ours)')
    
    plt.title('Layer-wise Energy Evolution (Final State)')
    plt.xlabel('Layer Depth (Input -> Output)')
    plt.ylabel('Energy E(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmark_energy.png')
    plt.close()
    
    print("Benchmark Complete. Plots saved: benchmark_loss.png, benchmark_grads.png, benchmark_energy.png")

if __name__ == "__main__":
    run_benchmark()
