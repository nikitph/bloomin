"""
FIXED Distractor Stress Test
Task now REQUIRES observer - answer NOT in input
"""

import torch
from rssa_v3 import RSSA, train_step, validate_observer_emergence
import numpy as np


def generate_needle_haystack_task_FIXED(batch_size, vocab_size, n_distractors=10):
    """
    FIXED: Needle in haystack WITHOUT answer in input
    
    Format: [KEY, value] [distractor1] [distractor2] ... [QUERY, KEY] → [value]
    
    Key difference: Final sequence is [QUERY, KEY] not [QUERY, KEY, value]
    Model MUST remember value from beginning, can't just copy
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
        
        # FIXED: Sequence does NOT include answer at end
        # [KEY, value] [distractors...] [QUERY, KEY] → predict [value]
        input_seq = torch.cat([
            torch.tensor([key_token]),
            value,
            distractors,
            torch.tensor([query_token, key_token])
            # NO value here! Model must remember from position 1
        ])
        
        # Target: predict the value (which is NOT in recent context)
        target_seq = torch.cat([
            torch.full((1 + 1 + n_distractors + 2,), -100),
            value
        ])
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    # Pad
    max_len = max(len(x) for x in inputs) + 1  # +1 for answer position
    
    input_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=sep_token)
        for x in inputs
    ])
    
    target_ids = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=-100)
        for x in targets
    ])
    
    return input_ids, target_ids


def run_distractor_experiment_FIXED(tau, vocab_size, n_distractors, n_steps=500):
    """Run FIXED experiment"""
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
        input_ids, target_ids = generate_needle_haystack_task_FIXED(
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
    test_input, test_target = generate_needle_haystack_task_FIXED(4, vocab_size, n_distractors)
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
    print(f"  Success: {final_loss < 0.5}")  # More lenient threshold
    
    return {
        'tau': tau,
        'n_distractors': n_distractors,
        'final_loss': final_loss,
        'observer_std': avg_obs_std,
        'observer_norm': results['observer_stats']['norm'],
        'is_stable': results['is_stable'],
        'success': final_loss < 0.5,
        'losses': losses,
        'observer_stds': observer_stds
    }


def main():
    print("="*70)
    print("FIXED DISTRACTOR STRESS TEST")
    print("="*70)
    print("\nFIX: Answer NOT in input - model must use memory!")
    print("\nHypothesis:")
    print("  τ=0.95 (slow): Maintains signal through noise → SUCCESS")
    print("  τ=0.1 (fast): Signal degraded by noise → FAILURE")
    print("="*70)
    
    # First verify task is actually hard
    print("\n" + "="*70)
    print("VERIFICATION: Task requires memory")
    print("="*70)
    
    input_ids, target_ids = generate_needle_haystack_task_FIXED(1, 20, n_distractors=10)
    print(f"\nSample sequence:")
    print(f"Input:  {input_ids[0].tolist()}")
    print(f"Target: {target_ids[0].tolist()}")
    
    # Check answer not in input
    answer_val = None
    for i, tgt in enumerate(target_ids[0]):
        if tgt != -100:
            answer_val = tgt.item()
            answer_pos = i
            break
    
    if answer_val is not None:
        # Check if answer appears in recent context (last 5 tokens before answer)
        recent_context = input_ids[0][max(0, answer_pos-5):answer_pos].tolist()
        if answer_val in recent_context:
            print(f"\n❌ Answer {answer_val} still in recent context: {recent_context}")
        else:
            print(f"\n✓ Answer {answer_val} NOT in recent context")
            print(f"  Recent context: {recent_context}")
            print(f"  Model MUST use memory!")
    
    vocab_size = 20
    results = []
    
    # Experiment 1: τ=0.95, moderate distractors
    print("\n" + "="*70)
    print("EXPERIMENT 1: τ=0.95, 20 distractors")
    print("="*70)
    
    result1 = run_distractor_experiment_FIXED(
        tau=0.95,
        vocab_size=vocab_size,
        n_distractors=20,
        n_steps=1000  # More steps for harder task
    )
    results.append(result1)
    
    # Experiment 2: τ=0.1, moderate distractors
    print("\n" + "="*70)
    print("EXPERIMENT 2: τ=0.1, 20 distractors")
    print("="*70)
    
    result2 = run_distractor_experiment_FIXED(
        tau=0.1,
        vocab_size=vocab_size,
        n_distractors=20,
        n_steps=1000
    )
    results.append(result2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: FIXED Distractor Test")
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
    
    if results[0]['success'] and not results[1]['success']:
        print("\n✅ HYPOTHESIS CONFIRMED:")
        print("   τ=0.95 succeeds, τ=0.1 fails")
        print("   Slow dissipative dynamics ARE required!")
    elif results[0]['success'] and results[1]['success']:
        print("\n⚠️  BOTH SUCCEEDED:")
        print("   Task may still be too easy, or architecture very robust")
    else:
        print("\n❌ UNEXPECTED RESULT")
    
    # Observer dynamics
    print(f"\nObserver std comparison:")
    print(f"  τ=0.95: {results[0]['observer_std']:.4f}")
    print(f"  τ=0.1: {results[1]['observer_std']:.4f}")
    print(f"  Ratio: {results[1]['observer_std'] / results[0]['observer_std']:.2f}x")
    
    return results


if __name__ == "__main__":
    results = main()
