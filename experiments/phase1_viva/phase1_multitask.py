"""
Phase 1: Multi-Task Validation
Validates observer emergence across diverse task families
"""

import torch
import numpy as np
from rssa_v3 import RSSA, train_step, validate_observer_emergence


# ============================================================================
# Task 1: Copy Task (already validated)
# ============================================================================

def generate_copy_task(batch_size, seq_len, vocab_size, delay):
    """Generate copy task: [input] [PAD × delay] [target]"""
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
# Task 2: Associative Recall
# ============================================================================

def generate_associative_recall_task(batch_size, vocab_size, n_pairs=5):
    """
    Generate associative recall task
    
    Format: [key1, val1] [key2, val2] ... [keyN, valN] [SEP] [query_key] → [val]
    
    Tests content-addressable memory
    """
    sep_token = vocab_size - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Generate key-value pairs
        keys = torch.randint(1, vocab_size // 2, (n_pairs,))
        vals = torch.randint(vocab_size // 2, vocab_size - 1, (n_pairs,))
        
        # Interleave keys and values
        pairs = torch.stack([keys, vals], dim=1).flatten()
        
        # Add separator
        sep = torch.tensor([sep_token])
        
        # Random query (one of the keys)
        query_idx = torch.randint(0, n_pairs, (1,)).item()
        query_key = keys[query_idx:query_idx+1]
        target_val = vals[query_idx:query_idx+1]
        
        # Construct sequence
        input_seq = torch.cat([pairs, sep, query_key])
        
        # Target: ignore everything except the answer
        target_seq = torch.cat([
            torch.full((len(pairs) + 1 + 1,), -100),  # Ignore pairs, sep, query
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
# Task 3: Addition with Carry
# ============================================================================

def generate_addition_task(batch_size, vocab_size, max_digits=3):
    """
    Generate addition task
    
    Format: [d1_1, d1_2, ..., d1_n] [+] [d2_1, d2_2, ..., d2_n] [=] [result]
    
    Tests algorithmic reasoning with carry propagation
    """
    plus_token = vocab_size - 2
    equals_token = vocab_size - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Generate two random numbers
        n_digits = torch.randint(2, max_digits + 1, (1,)).item()
        
        num1_digits = torch.randint(0, 10, (n_digits,))
        num2_digits = torch.randint(0, 10, (n_digits,))
        
        # Compute sum
        num1 = sum(d.item() * (10 ** i) for i, d in enumerate(reversed(num1_digits)))
        num2 = sum(d.item() * (10 ** i) for i, d in enumerate(reversed(num2_digits)))
        result = num1 + num2
        
        # Convert result to digits
        result_str = str(result)
        result_digits = torch.tensor([int(d) for d in result_str])
        
        # Construct sequence
        input_seq = torch.cat([
            num1_digits,
            torch.tensor([plus_token]),
            num2_digits,
            torch.tensor([equals_token]),
            result_digits
        ])
        
        # Target: ignore everything except result
        target_seq = torch.cat([
            torch.full((len(num1_digits) + 1 + len(num2_digits) + 1,), -100),
            result_digits
        ])
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Pad to same length
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
# Multi-Task Experiment Runner
# ============================================================================

def run_task_experiment(
    task_name: str,
    task_generator,
    vocab_size: int,
    n_steps: int = 1000,
    **task_kwargs
):
    """Run experiment on a single task"""
    print(f"\n{'='*70}")
    print(f"Task: {task_name}")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
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
    
    # Training
    batch_size = 8
    losses = []
    observer_norms = []
    
    print("Training...")
    for step in range(n_steps):
        # Generate batch
        input_ids, target_ids = task_generator(batch_size, vocab_size, **task_kwargs)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Train step
        metrics = train_step(model, input_ids, target_ids, optimizer)
        losses.append(metrics['loss'])
        observer_norms.append(model.get_observer_stats()['norm'])
        
        # Log
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"Step {step+1:4d} | Loss: {avg_loss:.4f} | "
                  f"Obs Norm: {observer_norms[-1]:.4f}")
    
    # Validation
    print("\nValidation:")
    test_input, test_target = task_generator(4, vocab_size, **task_kwargs)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    results = validate_observer_emergence(model, test_input, test_target)
    
    print(f"  Normal loss: {results['loss_normal']:.4f}")
    print(f"  Ablated loss: {results['loss_ablated']:.4f}")
    print(f"  Loss increase: {results['loss_increase_pct']:.1f}%")
    print(f"  Observer norm: {results['observer_stats']['norm']:.4f}")
    print(f"  Observer std: {results['observer_stats']['std']:.4f}")
    print(f"  Non-trivial: {results['is_nontrivial']}")
    print(f"  Necessary: {results['is_necessary']}")
    print(f"  Stable: {results['is_stable']}")
    
    if all([results['is_nontrivial'], results['is_necessary']]):
        print(f"\n✓ SUCCESS: Observer emerged for {task_name}!")
    else:
        print(f"\n✗ PARTIAL: Observer did not fully emerge for {task_name}")
    
    return {
        'task_name': task_name,
        'final_loss': results['loss_normal'],
        'ablated_loss': results['loss_ablated'],
        'ablation_impact': results['loss_increase_pct'],
        'observer_norm': results['observer_stats']['norm'],
        'observer_std': results['observer_stats']['std'],
        'is_nontrivial': results['is_nontrivial'],
        'is_necessary': results['is_necessary'],
        'is_stable': results['is_stable'],
        'losses': losses,
        'observer_norms': observer_norms
    }


def main():
    print("="*70)
    print("Phase 1: Multi-Task Validation")
    print("Testing observer emergence across diverse task families")
    print("="*70)
    
    vocab_size = 20
    results = []
    
    # Task 1: Copy Task
    print("\n" + "="*70)
    print("TASK 1: Copy with Delay")
    print("Tests: Long-range memory")
    print("="*70)
    
    result1 = run_task_experiment(
        task_name="Copy Task",
        task_generator=generate_copy_task,
        vocab_size=vocab_size,
        n_steps=1000,
        seq_len=512,
        delay=200
    )
    results.append(result1)
    
    # Task 2: Associative Recall
    print("\n" + "="*70)
    print("TASK 2: Associative Recall")
    print("Tests: Content-addressable memory")
    print("="*70)
    
    result2 = run_task_experiment(
        task_name="Associative Recall",
        task_generator=generate_associative_recall_task,
        vocab_size=vocab_size,
        n_steps=1000,
        n_pairs=5
    )
    results.append(result2)
    
    # Task 3: Addition with Carry
    print("\n" + "="*70)
    print("TASK 3: Addition with Carry")
    print("Tests: Algorithmic reasoning")
    print("="*70)
    
    result3 = run_task_experiment(
        task_name="Addition",
        task_generator=generate_addition_task,
        vocab_size=vocab_size,
        n_steps=1000,
        max_digits=3
    )
    results.append(result3)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Multi-Task Validation")
    print("="*70)
    
    print(f"\n{'Task':<20} {'Loss':<10} {'Ablation %':<12} {'Obs Norm':<12} {'Emerged?':<10}")
    print("-" * 70)
    
    for r in results:
        emerged = "✓" if r['is_nontrivial'] and r['is_necessary'] else "✗"
        print(f"{r['task_name']:<20} {r['final_loss']:<10.4f} {r['ablation_impact']:<12.1f} "
              f"{r['observer_norm']:<12.4f} {emerged:<10}")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    all_emerged = all(r['is_nontrivial'] and r['is_necessary'] for r in results)
    
    if all_emerged:
        print("\n✓ SUCCESS: Observer emerged in ALL tasks!")
        print("  This validates architectural generality.")
    else:
        print("\n✗ PARTIAL: Observer did not emerge in all tasks")
        print("  Further investigation needed.")
    
    # Check consistency
    norms = [r['observer_norm'] for r in results]
    norm_std = np.std(norms)
    
    print(f"\nObserver norm consistency:")
    print(f"  Mean: {np.mean(norms):.4f}")
    print(f"  Std: {norm_std:.4f}")
    
    if norm_std < 0.5:
        print("  ✓ Observer norms are consistent across tasks")
    else:
        print("  ✗ Observer norms vary significantly across tasks")
    
    print("\n" + "="*70)
    print("Multi-task validation complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
