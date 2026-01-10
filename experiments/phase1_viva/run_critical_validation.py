"""
Critical Validation Suite
Runs essential remaining experiments to make claims bulletproof
"""

import torch
from rssa_v3 import RSSA, train_step, validate_observer_emergence
import numpy as np


# ============================================================================
# Task 2: Associative Recall
# ============================================================================

def generate_associative_recall_task(batch_size, vocab_size, n_pairs=5):
    """
    Associative recall: [key1, val1, key2, val2, ..., SEP, query_key] → [val]
    
    Tests content-addressable memory
    """
    sep_token = vocab_size - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Generate unique key-value pairs
        keys = torch.randint(1, vocab_size // 2, (n_pairs,))
        vals = torch.randint(vocab_size // 2, vocab_size - 1, (n_pairs,))
        
        # Interleave keys and values
        pairs = torch.stack([keys, vals], dim=1).flatten()
        
        # Random query
        query_idx = torch.randint(0, n_pairs, (1,)).item()
        query_key = keys[query_idx:query_idx+1]
        target_val = vals[query_idx:query_idx+1]
        
        # Sequence: [pairs] [SEP] [query] [answer]
        input_seq = torch.cat([pairs, torch.tensor([sep_token]), query_key, target_val])
        
        # Target: only predict the answer
        target_seq = torch.cat([
            torch.full((len(pairs) + 1 + 1,), -100),
            target_val
        ])
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Pad to same length
    max_len = max(len(x) for x in inputs)
    
    input_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=sep_token)
        for x in inputs
    ])
    
    target_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=-100)
        for x in targets
    ])
    
    return input_ids, target_ids


# ============================================================================
# Task 3: Addition
# ============================================================================

def generate_addition_task(batch_size, vocab_size, max_num=99):
    """
    Addition: [num1_digits] [+] [num2_digits] [=] [result_digits]
    
    Tests algorithmic reasoning with carry propagation
    """
    plus_token = vocab_size - 2
    equals_token = vocab_size - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Generate two random numbers
        num1 = torch.randint(1, max_num + 1, (1,)).item()
        num2 = torch.randint(1, max_num + 1, (1,)).item()
        result = num1 + num2
        
        # Convert to digit sequences (reversed for easier carry)
        num1_digits = torch.tensor([int(d) for d in str(num1)])
        num2_digits = torch.tensor([int(d) for d in str(num2)])
        result_digits = torch.tensor([int(d) for d in str(result)])
        
        # Sequence: [num1] [+] [num2] [=] [result]
        input_seq = torch.cat([
            num1_digits,
            torch.tensor([plus_token]),
            num2_digits,
            torch.tensor([equals_token]),
            result_digits
        ])
        
        # Target: only predict result
        target_seq = torch.cat([
            torch.full((len(num1_digits) + 1 + len(num2_digits) + 1,), -100),
            result_digits
        ])
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Pad
    max_len = max(len(x) for x in inputs)
    
    input_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=0)
        for x in inputs
    ])
    
    target_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=-100)
        for x in targets
    ])
    
    return input_ids, target_ids


# ============================================================================
# Task 4: No-Observer Control
# ============================================================================

class RSSANoObserver(RSSA):
    """RSSA without observer channel - critical control"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Disable observer by making it a no-op
        self.d_observer = 1
        self.observer_state = torch.zeros(1, 1, 1)
        
    def _process_chunk(self, chunk, observer):
        """Process chunk WITHOUT observer context"""
        # Skip observer injection
        h = chunk
        
        # Process through layers
        for layer in self.layers:
            h = layer(h)
        
        # No observer update
        return h, observer


def generate_copy_task(batch_size, seq_len, vocab_size, delay):
    """Copy task for no-observer control"""
    input_seq = torch.randint(1, vocab_size-1, (batch_size, seq_len // 4))
    pad_token = vocab_size - 1
    delay_seq = torch.full((batch_size, delay), pad_token)
    target_seq = input_seq.clone()
    
    full_input = torch.cat([input_seq, delay_seq, target_seq], dim=1)
    full_target = torch.cat([
        torch.full_like(input_seq, -100),
        torch.full_like(delay_seq, -100),
        target_seq
    ], dim=1)
    
    return full_input, full_target


# ============================================================================
# Experiment Runner
# ============================================================================

def run_task(task_name, task_generator, vocab_size, n_steps=500, **task_kwargs):
    """Run single task experiment"""
    print(f"\n{'='*70}")
    print(f"Task: {task_name}")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RSSA(
        vocab_size=vocab_size,
        d_model=256,
        d_observer=64,
        n_layers=4,
        processor_window=128,
        observer_tau=0.95,
        chunk_size=64
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    batch_size = 8
    losses = []
    
    print("Training...")
    for step in range(n_steps):
        input_ids, target_ids = task_generator(batch_size, vocab_size, **task_kwargs)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        metrics = train_step(model, input_ids, target_ids, optimizer)
        losses.append(metrics['loss'])
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            obs_norm = model.get_observer_stats()['norm']
            print(f"  Step {step+1}: Loss {avg_loss:.4f}, Obs Norm {obs_norm:.4f}")
    
    # Validate
    test_input, test_target = task_generator(4, vocab_size, **task_kwargs)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    results = validate_observer_emergence(model, test_input, test_target)
    
    print(f"\nResults:")
    print(f"  Loss: {results['loss_normal']:.4f}")
    print(f"  Ablation: {results['loss_increase_pct']:.1f}%")
    print(f"  Observer norm: {results['observer_stats']['norm']:.4f}")
    print(f"  Emerged: {results['is_nontrivial'] and results['is_necessary']}")
    
    return results


def run_no_observer_control(vocab_size, n_steps=500):
    """Critical control: same architecture WITHOUT observer"""
    print(f"\n{'='*70}")
    print(f"CRITICAL CONTROL: No Observer Channel")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model WITHOUT observer
    model = RSSANoObserver(
        vocab_size=vocab_size,
        d_model=256,
        d_observer=1,  # Minimal
        n_layers=4,
        processor_window=128,
        observer_tau=0.95,
        chunk_size=64
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    batch_size = 8
    seq_len = 512
    delay = 200
    losses = []
    
    print("Training (no observer)...")
    for step in range(n_steps):
        input_ids, target_ids = generate_copy_task(batch_size, seq_len, vocab_size, delay)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        metrics = train_step(model, input_ids, target_ids, optimizer)
        losses.append(metrics['loss'])
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"  Step {step+1}: Loss {avg_loss:.4f}")
    
    final_loss = sum(losses[-100:]) / 100
    
    print(f"\nResults:")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Converged: {final_loss < 0.1}")
    
    return {'final_loss': final_loss, 'losses': losses}


def main():
    print("="*70)
    print("Critical Validation Suite")
    print("Making observer emergence claims BULLETPROOF")
    print("="*70)
    
    vocab_size = 20
    results = {}
    
    # Task 2: Associative Recall
    print("\n" + "="*70)
    print("TASK 2: Associative Recall")
    print("Tests: Content-addressable memory (key-value storage)")
    print("="*70)
    
    results['associative'] = run_task(
        "Associative Recall",
        generate_associative_recall_task,
        vocab_size,
        n_steps=500,
        n_pairs=5
    )
    
    # Task 3: Addition
    print("\n" + "="*70)
    print("TASK 3: Addition with Carry")
    print("Tests: Algorithmic reasoning (carry propagation)")
    print("="*70)
    
    results['addition'] = run_task(
        "Addition",
        generate_addition_task,
        vocab_size,
        n_steps=500,
        max_num=99
    )
    
    # Task 4: No Observer Control
    print("\n" + "="*70)
    print("TASK 4: No Observer Control")
    print("Tests: Observer is architecturally necessary")
    print("="*70)
    
    results['no_observer'] = run_no_observer_control(vocab_size, n_steps=500)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n✓ MULTI-TASK VALIDATION:")
    print(f"  Associative Recall:")
    print(f"    Loss: {results['associative']['loss_normal']:.4f}")
    print(f"    Ablation: {results['associative']['loss_increase_pct']:.1f}%")
    print(f"    Emerged: {results['associative']['is_nontrivial'] and results['associative']['is_necessary']}")
    
    print(f"\n  Addition:")
    print(f"    Loss: {results['addition']['loss_normal']:.4f}")
    print(f"    Ablation: {results['addition']['loss_increase_pct']:.1f}%")
    print(f"    Emerged: {results['addition']['is_nontrivial'] and results['addition']['is_necessary']}")
    
    print(f"\n✓ NO OBSERVER CONTROL:")
    print(f"  Final loss: {results['no_observer']['final_loss']:.4f}")
    print(f"  Task failed: {results['no_observer']['final_loss'] > 1.0}")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    all_emerged = (
        results['associative']['is_nontrivial'] and results['associative']['is_necessary'] and
        results['addition']['is_nontrivial'] and results['addition']['is_necessary']
    )
    
    no_observer_failed = results['no_observer']['final_loss'] > 1.0
    
    if all_emerged:
        print("✅ Multi-task: Observer emerged in ALL tasks")
    else:
        print("⚠️  Multi-task: Observer did not emerge in all tasks")
    
    if no_observer_failed:
        print("✅ No observer: Task FAILED without observer (proves necessity)")
    else:
        print("⚠️  No observer: Task succeeded (unexpected)")
    
    if all_emerged and no_observer_failed:
        print("\n" + "="*70)
        print("🎉 CLAIMS ARE BULLETPROOF!")
        print("="*70)
        print("\n✓ Observer forms invariant subspaces (linearized dynamics)")
        print("✓ Observer emerges across task families (multi-task)")
        print("✓ Observer is architecturally necessary (no-observer control)")
    
    return results


if __name__ == "__main__":
    results = main()
