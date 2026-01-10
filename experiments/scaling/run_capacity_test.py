"""
Experiment 2: Capacity Stress Test
Find the point where observer becomes NECESSARY, not just helpful
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rssa_v3 import RSSA, train_step
import numpy as np

class CapacityLimitedRSSA(RSSA):
    """
    Model with explicit capacity limit on processor.
    Observer has separate capacity via obs_to_proc projection.
    """
    def __init__(self, *args, d_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_processor = d_processor
        
        if d_processor < self.d_model:
            # Add bottleneck to processor path BEFORE layers
            # This forces information through a narrow pipe
            self.processor_bottleneck = nn.Linear(self.d_model, d_processor)
            self.processor_expand = nn.Linear(d_processor, self.d_model)
        else:
            self.processor_bottleneck = None
            
    def _process_chunk(self, chunk, observer, ablate_observer=False):
        # Inject observer
        if not ablate_observer:
            obs_context = self.obs_to_proc(observer)
            h = chunk + obs_context
        else:
            h = chunk
            
        # Apply capacity bottleneck to the hidden state h
        # This limits the "working memory" capacity of the processor
        if self.processor_bottleneck is not None:
            h = self.processor_bottleneck(h)
            h = self.processor_expand(h)
            
        # Process through layers
        for layer in self.layers:
            h = layer(h)
            
        # Extract observer update
        candidate_update = self.proc_to_obs(h)
        gate = self.update_gate(h)
        chunk_summary = (candidate_update * gate).mean(dim=1, keepdim=True)
        
        # EMA update
        if not ablate_observer:
            new_observer = (
                self.observer_tau * observer +
                (1 - self.observer_tau) * chunk_summary
            )
        else:
            new_observer = torch.zeros_like(observer)
            
        return h, new_observer

def generate_hard_copy_task(batch_size, seq_len, vocab_size, delay):
    """
    STRICT Copy Task: [Input] [Delay] [Query] -> [Input]
    Model MUST remember Input from the beginning.
    """
    input_len = 16 # Short but enough to test memory
    
    inputs = []
    targets = []
    
    query_token = vocab_size - 2
    pad_token = vocab_size - 1
    
    for _ in range(batch_size):
        # Random input sequence
        val = torch.randint(1, vocab_size-2, (input_len,))
        
        # Input: [VAL] [PAD x delay] [QUERY] [PAD x input_len]
        # At QUERY, start predicting VAL
        in_seq = torch.cat([
            val,
            torch.full((delay,), pad_token),
            torch.tensor([query_token]),
            torch.full((input_len,), pad_token) # Changed from input_len - 1 to input_len
        ])
        
        # Target: [IGNORE x input_len + delay + 1] [VAL]
        target = torch.cat([
            torch.full((input_len + delay + 1,), -100),
            val
        ])
        
        inputs.append(in_seq)
        targets.append(target)
        
    return torch.stack(inputs), torch.stack(targets)

def test_capacity_config(d_model, d_processor, d_observer, n_steps=500):
    """Test one capacity configuration"""
    print(f"\nEvaluating d_processor={d_processor} (Capacity: {d_processor/d_model*100:.1f}%)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Common params
    args = (20,) # vocab_size
    kwargs = {
        'd_model': d_model,
        'd_observer': d_observer,
        'n_layers': 4,
        'processor_window': 128,
        'observer_tau': 0.95,
        'chunk_size': 64,
        'd_processor': d_processor
    }
    
    # 1. Model WITH Observer
    model_with = CapacityLimitedRSSA(*args, **kwargs).to(device)
    opt_with = torch.optim.AdamW(model_with.parameters(), lr=1e-3)
    
    # 2. Model WITHOUT Observer (No-Observer Control)
    model_no = CapacityLimitedRSSA(*args, **kwargs).to(device)
    opt_no = torch.optim.AdamW(model_no.parameters(), lr=1e-3)
    
    # Training Loop
    losses_with = []
    losses_no = []
    
    for step in range(n_steps):
        # Generate task (hard coded delay=200 > window=128)
        x_task, y_task = generate_hard_copy_task(8, 256, 20, 200)
        x_task, y_task = x_task.to(device), y_task.to(device)
        
        # Train WITH observer
        metrics_with = train_step(model_with, x_task, y_task, opt_with)
        losses_with.append(metrics_with['loss'])
        
        # Train WITHOUT observer (ablate both influence and state propagation)
        # We simulate no-observer by calling model with ablate_observer=True during training
        model_no.train()
        logits_no, _ = model_no(x_task, reset_observer=True, ablate_observer=True)
        loss_no = F.cross_entropy(
            logits_no.reshape(-1, logits_no.size(-1)),
            y_task.reshape(-1),
            ignore_index=-100
        )
        opt_no.zero_grad()
        loss_no.backward()
        torch.nn.utils.clip_grad_norm_(model_no.parameters(), 1.0)
        opt_no.step()
        losses_no.append(loss_no.item())
        
        if (step + 1) % 100 == 0:
            avg_w = sum(losses_with[-100:]) / 100
            avg_n = sum(losses_no[-100:]) / 100
            print(f"  Step {step+1:3d} | With: {avg_w:.4f} | No: {avg_n:.4f} | Gap: {avg_n - avg_w:.4f}")
            
    final_with = sum(losses_with[-50:]) / 50
    final_no = sum(losses_no[-50:]) / 50
    gap = final_no - final_with
    
    return {
        'd_processor': d_processor,
        'cap_pct': d_processor / d_model * 100,
        'loss_with': final_with,
        'loss_no': final_no,
        'gap': gap,
        'necessary': gap > 0.5 # Threshold for declaring "necessary"
    }

def main():
    print("="*70)
    print("CAPACITY STRESS TEST: Necessity Probe")
    print("="*70)
    print("Hypothesis: Observer gap grows as processor capacity is restricted.")
    
    d_model = 256
    d_observer = 64
    
    # Sweep from 100% capacity down to 12.5%
    capacities = [256, 128, 64, 32]
    
    results = []
    for d_proc in capacities:
        res = test_capacity_config(d_model, d_proc, d_observer)
        results.append(res)
        
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Cap %':<10} {'Loss w/ Obs':<15} {'Loss w/o Obs':<15} {'Gap':<10} {'Necessary?'}")
    print("-" * 70)
    
    for r in results:
        nec = "YES" if r['necessary'] else "no"
        print(f"{r['cap_pct']:>5.1f}%     {r['loss_with']:<15.4f} {r['loss_no']:<15.4f} {r['gap']:<10.4f} {nec}")
        
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    # Find the threshold
    necessary_configs = [r for r in results if r['necessary']]
    if necessary_configs:
        threshold = necessary_configs[0]['cap_pct']
        print(f"✅ Observer becomes NECESSARY at ~{threshold:.1f}% capacity.")
    else:
        print("❌ Observer never became strictly necessary in this sweep.")

if __name__ == "__main__":
    main()
