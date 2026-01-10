"""
RSSA-v3 Phase 3: Ablation Sweep
Systematically explore the hyperparameter space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rssa_v3 import RSSA, train_step
import pandas as pd
import itertools
import time
import os

class AmnesiacRSSA(RSSA):
    """RSSA with Identity instead of SSM to isolate observer performance"""
    def __init__(self, *args, use_ssm=False, **kwargs):
        super().__init__(*args, **kwargs)
        from rssa_v3 import ProcessorLayer
        self.layers = nn.ModuleList([
            ProcessorLayer(
                kwargs.get('d_model', 256),
                n_heads=kwargs.get('n_heads', 8),
                window_size=kwargs.get('processor_window', 256),
                dropout=kwargs.get('dropout', 0.1),
                use_ssm=use_ssm
            ) for _ in range(kwargs.get('n_layers', 4))
        ])

def generate_heavy_needle(batch_size, vocab_size, delay=128, needle_len=32):
    """
    HEAVY NEEDLE: [Token x 32] [Delay] [Query] -> [Token x 32]
    """
    pad_token = vocab_size - 1
    query_token = vocab_size - 2
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        token_val = torch.randint(1, vocab_size-2, (1,))
        needle = token_val.repeat(needle_len)
        
        in_seq = torch.cat([
            needle,
            torch.full((delay,), pad_token),
            torch.tensor([query_token]),
            torch.full((needle_len,), pad_token)
        ])
        
        target = torch.cat([
            torch.full((needle_len + delay + 1,), -100),
            needle
        ])
        
        inputs.append(in_seq)
        targets.append(target)
        
    return torch.stack(inputs), torch.stack(targets)

def run_experiment(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 10
    
    # We use a smaller d_model to speed up the sweep while keeping d_obs variable
    model = AmnesiacRSSA(
        vocab_size=vocab_size,
        d_model=128, 
        d_observer=config['d_observer'],
        n_layers=2, # Shorter model to emphasize observer path
        processor_window=config['window_size'],
        observer_tau=config['tau'],
        use_ssm=False
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    start_time = time.time()
    
    # Run for 2000 steps to see if hard task converges
    n_steps = 2000 
    for step in range(n_steps):
        # HARDER TASK: delay=512, needle_len=4
        # This requires much higher compression and longer persistence
        x, y = generate_heavy_needle(16, vocab_size, delay=512, needle_len=4)
        x, y = x.to(device), y.to(device)
        
        metrics = train_step(model, x, y, optimizer)
        losses.append(metrics['loss'])
        
    training_time = time.time() - start_time
    final_loss = sum(losses[-100:]) / 100 # Average of last 100 steps
    
    return {
        **config,
        'final_loss': final_loss,
        'training_time': training_time,
        'n_params': sum(p.numel() for p in model.parameters())
    }

def main():
    # Sweep parameters - Focused on revealing failure modes
    windows = [32, 64, 128, 256]
    d_observers = [16, 64, 256, 1024]
    # Tau is critical for long delays. 0.9 will likely forget.
    taus = [0.95, 0.99, 0.999, 0.9999]
    
    results = []
    
    # Sweep 1: Window Size
    print("\n" + "="*70)
    print("HARD SWEEP 1: WINDOW SIZE (W) | Delay=512, Needle=4")
    print("="*70)
    for w in windows:
        config = {'window_size': w, 'd_observer': 64, 'tau': 0.99}
        print(f"> Testing W={w:3d} | d_obs=64 | tau=0.99", end=' ', flush=True)
        res = run_experiment(config)
        results.append(res)
        print(f"| Loss: {res['final_loss']:.4f}")

    # Sweep 2: Observer Dimension (d_obs)
    print("\n" + "="*70)
    print("HARD SWEEP 2: OBSERVER DIMENSION (D_OBS) | Delay=512, Needle=4")
    print("="*70)
    for d in d_observers:
        if d == 64: continue 
        config = {'window_size': 32, 'd_observer': d, 'tau': 0.99}
        print(f"> Testing W=32  | d_obs={d:4d} | tau=0.99", end=' ', flush=True)
        res = run_experiment(config)
        results.append(res)
        print(f"| Loss: {res['final_loss']:.4f}")

    # Sweep 3: Tau (Persistence Rate)
    print("\n" + "="*70)
    print("HARD SWEEP 3: UPDATE RATE (TAU) | Delay=512, Needle=4")
    print("="*70)
    for t in taus:
        if t == 0.99: continue 
        config = {'window_size': 32, 'd_observer': 64, 'tau': t}
        print(f"> Testing W=32  | d_obs=64 | tau={t:.4f}", end=' ', flush=True)
        res = run_experiment(config)
        results.append(res)
        print(f"| Loss: {res['final_loss']:.4f}")

    # Process and save
    df = pd.DataFrame(results)
    output_path = "experiments/scaling/ablation_results_hard.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("HARD SWEEP COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_path}")
    
    # Print nice table
    print("\nSorted Configurations (by loss):")
    print(df.sort_values('final_loss').to_string(index=False))


if __name__ == "__main__":
    main()
