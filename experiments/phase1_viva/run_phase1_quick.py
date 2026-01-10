"""
Phase 1 Quick Validation
Runs essential experiments only (~15 minutes)
"""

import torch
from rssa_v3 import RSSA, train_step, validate_observer_emergence, analyze_observer_dynamics
import numpy as np

def generate_copy_task(batch_size, seq_len, vocab_size, delay):
    """Generate copy task"""
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


def quick_train(config_name, vocab_size, delay, window, tau, n_steps=300):
    """Quick training run"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RSSA(
        vocab_size=vocab_size,
        d_model=256,
        d_observer=64,
        n_layers=4,
        processor_window=window,
        observer_tau=tau,
        chunk_size=64
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    batch_size = 8
    seq_len = 512
    
    for step in range(n_steps):
        input_ids, target_ids = generate_copy_task(batch_size, seq_len, vocab_size, delay)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        metrics = train_step(model, input_ids, target_ids, optimizer)
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: Loss {metrics['loss']:.4f}, Obs Norm {model.get_observer_stats()['norm']:.4f}")
    
    # Validate
    test_input, test_target = generate_copy_task(4, seq_len, vocab_size, delay)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    results = validate_observer_emergence(model, test_input, test_target)
    
    return model, results


def main():
    print("="*70)
    print("Phase 1 Quick Validation (~15 minutes)")
    print("="*70)
    
    vocab_size = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Baseline
    print("\n1. BASELINE (delay=200, window=128, tau=0.95)")
    model_baseline, res_baseline = quick_train("Baseline", vocab_size, 200, 128, 0.95, 300)
    print(f"   Loss: {res_baseline['loss_normal']:.4f}")
    print(f"   Ablation: {res_baseline['loss_increase_pct']:.1f}%")
    print(f"   Emerged: {res_baseline['is_nontrivial'] and res_baseline['is_necessary']}")
    
    # 2. Linearized Dynamics
    print("\n2. LINEARIZED DYNAMICS")
    test_input, _ = generate_copy_task(4, 512, vocab_size, 200)
    test_input = test_input.to(device)
    
    dynamics = analyze_observer_dynamics(model_baseline, test_input)
    
    # 3. Control: Short Delay
    print("\n3. CONTROL: Short Delay (delay=64 < window=128)")
    _, res_short = quick_train("Short Delay", vocab_size, 64, 128, 0.95, 300)
    print(f"   Loss: {res_short['loss_normal']:.4f}")
    print(f"   Ablation: {res_short['loss_increase_pct']:.1f}%")
    print(f"   Emerged: {res_short['is_nontrivial'] and res_short['is_necessary']}")
    
    # 4. Control: Fast Updates
    print("\n4. CONTROL: Fast Updates (tau=0.1)")
    _, res_fast = quick_train("Fast Updates", vocab_size, 200, 128, 0.1, 300)
    print(f"   Loss: {res_fast['loss_normal']:.4f}")
    print(f"   Obs Std: {res_fast['observer_stats']['std']:.4f}")
    print(f"   Stable: {res_fast['is_stable']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n✓ LINEARIZED DYNAMICS:")
    print(f"  Spectral radius: {dynamics['jacobian']['spectral_radius']:.4f} (< 1.0 = contractive)")
    print(f"  Contraction rate: {dynamics['contraction']['contraction_rate']:.4f}")
    print(f"  Manifold return time: {dynamics['manifold']['scale_0.1']['mean_return_time']:.1f} steps")
    
    print("\n✓ CONDITIONAL EMERGENCE:")
    print(f"  Baseline ablation: {res_baseline['loss_increase_pct']:.1f}% (should be high)")
    print(f"  Short delay ablation: {res_short['loss_increase_pct']:.1f}% (should be low)")
    print(f"  Fast updates stable: {res_fast['is_stable']} (should be False)")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    checks = []
    
    # Check 1: Baseline observer emerges
    if res_baseline['is_nontrivial'] and res_baseline['is_necessary']:
        print("✓ Baseline: Observer emerged")
        checks.append(True)
    else:
        print("✗ Baseline: Observer did not emerge")
        checks.append(False)
    
    # Check 2: Contractive dynamics
    if dynamics['jacobian']['is_contractive']:
        print("✓ Dynamics: System is contractive (spectral radius < 1)")
        checks.append(True)
    else:
        print("✗ Dynamics: System is not contractive")
        checks.append(False)
    
    # Check 3: Short delay doesn't need observer
    if res_short['ablation_impact'] < res_baseline['ablation_impact'] * 0.5:
        print("✓ Short delay: Observer less necessary (conditional emergence)")
        checks.append(True)
    else:
        print("✗ Short delay: Observer still necessary (unexpected)")
        checks.append(False)
    
    # Check 4: Fast updates unstable
    if not res_fast['is_stable']:
        print("✓ Fast updates: Observer unstable (validates tau importance)")
        checks.append(True)
    else:
        print("✗ Fast updates: Observer stable (unexpected)")
        checks.append(False)
    
    print("\n" + "="*70)
    if all(checks):
        print("🎉 ALL CHECKS PASSED - Observer emergence is AIRTIGHT!")
    else:
        print(f"⚠️  {sum(checks)}/{len(checks)} checks passed")
    print("="*70)


if __name__ == "__main__":
    main()
