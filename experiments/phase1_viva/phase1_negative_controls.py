"""
Phase 1: Negative Controls
Proves observer emergence is conditional, not accidental
"""

import torch
import sys
from rssa_v3 import RSSA, train_step, validate_observer_emergence

def generate_copy_task(batch_size, seq_len, vocab_size, delay):
    """Generate copy task with configurable delay"""
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


def run_control_experiment(
    name: str,
    vocab_size: int,
    delay: int,
    processor_window: int,
    observer_tau: float,
    use_observer: bool,
    n_steps: int = 500
):
    """Run a single control experiment"""
    print(f"\n{'='*70}")
    print(f"Control: {name}")
    print(f"{'='*70}")
    print(f"Delay: {delay}, Window: {processor_window}, Tau: {observer_tau}")
    print(f"Use observer: {use_observer}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    if use_observer:
        model = RSSA(
            vocab_size=vocab_size,
            d_model=256,
            d_observer=64,
            n_layers=4,
            processor_window=processor_window,
            observer_tau=observer_tau,
            chunk_size=64
        ).to(device)
    else:
        # Model without observer (set d_observer=1 and disable updates)
        model = RSSA(
            vocab_size=vocab_size,
            d_model=256,
            d_observer=1,  # Minimal observer
            n_layers=4,
            processor_window=processor_window,
            observer_tau=0.99,
            chunk_size=64
        ).to(device)
        # Freeze observer parameters
        model.obs_to_proc.weight.requires_grad = False
        model.obs_to_proc.bias.requires_grad = False
        for param in model.proc_to_obs.parameters():
            param.requires_grad = False
        for param in model.update_gate.parameters():
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training
    seq_len = 512
    batch_size = 8
    losses = []
    observer_norms = []
    observer_changes = []
    
    for step in range(n_steps):
        input_ids, target_ids = generate_copy_task(batch_size, seq_len, vocab_size, delay)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        metrics = train_step(model, input_ids, target_ids, optimizer)
        losses.append(metrics['loss'])
        observer_norms.append(model.get_observer_stats()['norm'])
        observer_changes.append(metrics['observer_change'])
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"Step {step+1:4d} | Loss: {avg_loss:.4f} | "
                  f"Obs Norm: {observer_norms[-1]:.4f}")
    
    # Validation
    print("\nValidation:")
    test_input, test_target = generate_copy_task(4, seq_len, vocab_size, delay)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    results = validate_observer_emergence(model, test_input, test_target)
    
    print(f"  Final loss: {results['loss_normal']:.4f}")
    print(f"  Observer norm: {results['observer_stats']['norm']:.4f}")
    print(f"  Observer std: {results['observer_stats']['std']:.4f}")
    print(f"  Ablation impact: {results['loss_increase_pct']:.1f}%")
    print(f"  Non-trivial: {results['is_nontrivial']}")
    print(f"  Necessary: {results['is_necessary']}")
    print(f"  Stable: {results['is_stable']}")
    
    return {
        'name': name,
        'final_loss': results['loss_normal'],
        'observer_norm': results['observer_stats']['norm'],
        'observer_std': results['observer_stats']['std'],
        'ablation_impact': results['loss_increase_pct'],
        'is_nontrivial': results['is_nontrivial'],
        'is_necessary': results['is_necessary'],
        'is_stable': results['is_stable'],
        'losses': losses,
        'observer_norms': observer_norms
    }


def main():
    print("="*70)
    print("Phase 1: Negative Controls")
    print("Proving observer emergence is conditional, not accidental")
    print("="*70)
    
    vocab_size = 20
    results = []
    
    # Control 1: Short delay (delay < window)
    # Observer should NOT emerge - task is solvable without it
    print("\n" + "="*70)
    print("CONTROL 1: Short Delay (delay < window)")
    print("Expected: Observer does NOT emerge")
    print("="*70)
    
    result1 = run_control_experiment(
        name="Short Delay",
        vocab_size=vocab_size,
        delay=64,  # Less than window
        processor_window=128,
        observer_tau=0.95,
        use_observer=True,
        n_steps=500
    )
    results.append(result1)
    
    # Control 2: Fast updates (τ → 0)
    # Observer updates too fast to crystallize
    print("\n" + "="*70)
    print("CONTROL 2: Fast Updates (τ → 0)")
    print("Expected: Observer unstable, high variance")
    print("="*70)
    
    result2 = run_control_experiment(
        name="Fast Updates",
        vocab_size=vocab_size,
        delay=200,
        processor_window=128,
        observer_tau=0.1,  # Very fast updates
        use_observer=True,
        n_steps=500
    )
    results.append(result2)
    
    # Control 3: No observer channel
    # Baseline failure - task cannot be solved
    print("\n" + "="*70)
    print("CONTROL 3: No Observer Channel")
    print("Expected: Task fails, high loss")
    print("="*70)
    
    result3 = run_control_experiment(
        name="No Observer",
        vocab_size=vocab_size,
        delay=200,
        processor_window=128,
        observer_tau=0.95,
        use_observer=False,
        n_steps=500
    )
    results.append(result3)
    
    # Baseline: Normal (for comparison)
    print("\n" + "="*70)
    print("BASELINE: Normal Configuration")
    print("Expected: Observer emerges successfully")
    print("="*70)
    
    result_baseline = run_control_experiment(
        name="Baseline (Normal)",
        vocab_size=vocab_size,
        delay=200,
        processor_window=128,
        observer_tau=0.95,
        use_observer=True,
        n_steps=500
    )
    results.append(result_baseline)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Negative Controls")
    print("="*70)
    
    print(f"\n{'Experiment':<20} {'Loss':<10} {'Obs Norm':<12} {'Obs Std':<12} {'Ablation %':<12} {'Emerged?':<10}")
    print("-" * 90)
    
    for r in results:
        emerged = "✓" if r['is_nontrivial'] and r['is_necessary'] else "✗"
        print(f"{r['name']:<20} {r['final_loss']:<10.4f} {r['observer_norm']:<12.4f} "
              f"{r['observer_std']:<12.4f} {r['ablation_impact']:<12.1f} {emerged:<10}")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    baseline = results[3]  # Baseline result
    
    print("\n✓ Control 1 (Short Delay):")
    if results[0]['observer_norm'] < baseline['observer_norm'] * 0.5:
        print("  PASS: Observer norm significantly lower than baseline")
    else:
        print("  FAIL: Observer norm not significantly different")
    
    if results[0]['ablation_impact'] < 50:  # Less than baseline
        print("  PASS: Ablation impact lower (observer less necessary)")
    else:
        print("  FAIL: Ablation impact not lower")
    
    print("\n✓ Control 2 (Fast Updates):")
    if results[1]['observer_std'] > baseline['observer_std'] * 2:
        print("  PASS: Observer std significantly higher (unstable)")
    else:
        print("  FAIL: Observer std not significantly higher")
    
    if results[1]['final_loss'] > baseline['final_loss'] * 2:
        print("  PASS: Final loss higher (poor performance)")
    else:
        print("  FAIL: Final loss not significantly higher")
    
    print("\n✓ Control 3 (No Observer):")
    if results[2]['final_loss'] > baseline['final_loss'] * 5:
        print("  PASS: Final loss much higher (task fails)")
    else:
        print("  FAIL: Final loss not significantly higher")
    
    print("\n" + "="*70)
    print("Negative controls complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
