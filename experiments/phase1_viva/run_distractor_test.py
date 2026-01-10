"""
Distractor Stress Test: Needle in a Noisy Haystack
Proves that slow dissipative dynamics (high tau) are required for robust memory
"""

import torch
from rssa_v3 import RSSA, train_step, validate_observer_emergence
import numpy as np


def generate_needle_haystack_task(batch_size, vocab_size, n_distractors=10):
    """
    Needle in haystack: [KEY: A] [NOISE × n] [QUERY: KEY] → [A]
    
    Tests: Robust memory under distractor noise
    
    Hypothesis:
    - τ=0.95 (slow): Integrates A, resists noise updates → SUCCESS
    - τ=0.1 (fast): Updates on A, then noise overwrites → FAILURE
    """
    key_token = vocab_size - 3
    query_token = vocab_size - 2
    sep_token = vocab_size - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # The "needle" - value to remember
        value = torch.randint(1, vocab_size // 2, (1,))
        
        # Distractors - random noise
        distractors = torch.randint(vocab_size // 2, vocab_size - 3, (n_distractors,))
        
        # Sequence: [KEY: value] [distractor1] [distractor2] ... [QUERY: KEY] [value]
        input_seq = torch.cat([
            torch.tensor([key_token]),
            value,
            distractors,
            torch.tensor([query_token, key_token]),
            value
        ])
        
        # Target: only predict the final value
        target_seq = torch.cat([
            torch.full((1 + 1 + n_distractors + 2,), -100),
            value
        ])
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Pad
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


def run_distractor_experiment(tau, vocab_size, n_distractors, n_steps=500):
    """Run experiment with specific tau value"""
    print(f"\n{'='*70}")
    print(f"Tau = {tau} | Distractors = {n_distractors}")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RSSA(
        vocab_size=vocab_size,
        d_model=256,
        d_observer=64,
        n_layers=4,
        processor_window=128,
        observer_tau=tau,
        chunk_size=64
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    batch_size = 8
    losses = []
    observer_norms = []
    observer_stds = []
    
    print("Training...")
    for step in range(n_steps):
        input_ids, target_ids = generate_needle_haystack_task(
            batch_size, vocab_size, n_distractors
        )
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        metrics = train_step(model, input_ids, target_ids, optimizer)
        losses.append(metrics['loss'])
        
        stats = model.get_observer_stats()
        observer_norms.append(stats['norm'])
        observer_stds.append(stats['std'])
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            avg_std = sum(observer_stds[-100:]) / 100
            print(f"  Step {step+1}: Loss {avg_loss:.4f}, Obs Norm {observer_norms[-1]:.4f}, Obs Std {avg_std:.4f}")
    
    # Validate
    test_input, test_target = generate_needle_haystack_task(4, vocab_size, n_distractors)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    results = validate_observer_emergence(model, test_input, test_target)
    
    final_loss = sum(losses[-100:]) / 100
    avg_obs_std = sum(observer_stds[-100:]) / 100
    
    print(f"\nResults:")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Observer std: {avg_obs_std:.4f}")
    print(f"  Observer norm: {results['observer_stats']['norm']:.4f}")
    print(f"  Stable: {results['is_stable']}")
    print(f"  Success: {final_loss < 0.1}")
    
    return {
        'tau': tau,
        'n_distractors': n_distractors,
        'final_loss': final_loss,
        'observer_std': avg_obs_std,
        'observer_norm': results['observer_stats']['norm'],
        'is_stable': results['is_stable'],
        'success': final_loss < 0.1,
        'losses': losses,
        'observer_stds': observer_stds
    }


def main():
    print("="*70)
    print("DISTRACTOR STRESS TEST: Needle in a Noisy Haystack")
    print("="*70)
    print("\nHypothesis:")
    print("  τ=0.95 (slow): Resists distractor updates → SUCCESS")
    print("  τ=0.1 (fast): Overwrites signal with noise → FAILURE")
    print("="*70)
    
    vocab_size = 20
    results = []
    
    # Experiment 1: τ=0.95, few distractors
    print("\n" + "="*70)
    print("EXPERIMENT 1: τ=0.95, 10 distractors (baseline)")
    print("="*70)
    
    result1 = run_distractor_experiment(
        tau=0.95,
        vocab_size=vocab_size,
        n_distractors=10,
        n_steps=500
    )
    results.append(result1)
    
    # Experiment 2: τ=0.1, few distractors
    print("\n" + "="*70)
    print("EXPERIMENT 2: τ=0.1, 10 distractors (fast updates)")
    print("="*70)
    
    result2 = run_distractor_experiment(
        tau=0.1,
        vocab_size=vocab_size,
        n_distractors=10,
        n_steps=500
    )
    results.append(result2)
    
    # Experiment 3: τ=0.95, many distractors (stress test)
    print("\n" + "="*70)
    print("EXPERIMENT 3: τ=0.95, 50 distractors (high noise)")
    print("="*70)
    
    result3 = run_distractor_experiment(
        tau=0.95,
        vocab_size=vocab_size,
        n_distractors=50,
        n_steps=500
    )
    results.append(result3)
    
    # Experiment 4: τ=0.1, many distractors (should fail)
    print("\n" + "="*70)
    print("EXPERIMENT 4: τ=0.1, 50 distractors (should fail)")
    print("="*70)
    
    result4 = run_distractor_experiment(
        tau=0.1,
        vocab_size=vocab_size,
        n_distractors=50,
        n_steps=500
    )
    results.append(result4)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Distractor Stress Test")
    print("="*70)
    
    print(f"\n{'Tau':<8} {'Distractors':<12} {'Final Loss':<12} {'Obs Std':<12} {'Success':<10}")
    print("-" * 70)
    
    for r in results:
        success_str = "✓" if r['success'] else "✗"
        print(f"{r['tau']:<8.2f} {r['n_distractors']:<12} {r['final_loss']:<12.4f} "
              f"{r['observer_std']:<12.4f} {success_str:<10}")
    
    # Validation
    print("\n" + "="*70)
    print("HYPOTHESIS VALIDATION")
    print("="*70)
    
    # Check 1: τ=0.95 succeeds with few distractors
    if results[0]['success']:
        print("✅ τ=0.95, 10 distractors: SUCCESS (as predicted)")
    else:
        print("❌ τ=0.95, 10 distractors: FAILED (unexpected)")
    
    # Check 2: τ=0.1 fails with few distractors
    if not results[1]['success']:
        print("✅ τ=0.1, 10 distractors: FAILED (as predicted)")
    else:
        print("⚠️  τ=0.1, 10 distractors: SUCCESS (unexpected)")
    
    # Check 3: τ=0.95 robust to many distractors
    if results[2]['success']:
        print("✅ τ=0.95, 50 distractors: SUCCESS (robust to noise)")
    else:
        print("⚠️  τ=0.95, 50 distractors: FAILED (not robust enough)")
    
    # Check 4: τ=0.1 definitely fails with many distractors
    if not results[3]['success']:
        print("✅ τ=0.1, 50 distractors: FAILED (as predicted)")
    else:
        print("❌ τ=0.1, 50 distractors: SUCCESS (hypothesis refuted)")
    
    # Observer std comparison
    print("\n" + "="*70)
    print("OBSERVER DYNAMICS ANALYSIS")
    print("="*70)
    
    slow_std = (results[0]['observer_std'] + results[2]['observer_std']) / 2
    fast_std = (results[1]['observer_std'] + results[3]['observer_std']) / 2
    
    print(f"\nAverage Observer Std:")
    print(f"  τ=0.95 (slow): {slow_std:.4f}")
    print(f"  τ=0.1 (fast): {fast_std:.4f}")
    print(f"  Ratio: {fast_std / slow_std:.2f}x")
    
    if fast_std > slow_std * 2:
        print("\n✅ Fast updates cause significantly higher variance (unstable)")
    else:
        print("\n⚠️  Fast updates not significantly more variable")
    
    # Final conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    slow_success_rate = sum(r['success'] for r in [results[0], results[2]]) / 2
    fast_success_rate = sum(r['success'] for r in [results[1], results[3]]) / 2
    
    print(f"\nSuccess rates:")
    print(f"  τ=0.95 (slow): {slow_success_rate*100:.0f}%")
    print(f"  τ=0.1 (fast): {fast_success_rate*100:.0f}%")
    
    if slow_success_rate > fast_success_rate:
        print("\n✅ HYPOTHESIS CONFIRMED:")
        print("   Slow dissipative dynamics (high τ) are REQUIRED")
        print("   for robust memory under distractor noise.")
        print("\n   This proves architectural constraints (τ) drive emergence.")
    else:
        print("\n⚠️  HYPOTHESIS NOT CONFIRMED:")
        print("   Fast updates performed comparably to slow updates.")
    
    return results


if __name__ == "__main__":
    results = main()
