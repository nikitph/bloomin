"""
Diagnostic: Identify Issues in RSSA Implementation
"""

import torch
from rssa_v3 import RSSA

def diagnose_observer_influence():
    """Test if observer actually influences processor"""
    print("="*70)
    print("DIAGNOSTIC 1: Observer Influence Test")
    print("="*70)
    
    vocab_size = 20
    device = 'cpu'
    
    model = RSSA(
        vocab_size=vocab_size,
        d_model=256,
        d_observer=64,
        n_layers=4,
        processor_window=128,
        observer_tau=0.95,
        chunk_size=64
    ).to(device)
    
    # Create test input
    test_input = torch.randint(0, vocab_size, (2, 128))
    
    # Test 1: Normal forward pass
    model.observer_state = torch.randn_like(model.observer_state)
    obs_before = model.observer_state.clone()
    
    logits1, _ = model(test_input)
    
    # Test 2: Zero observer
    model.observer_state = torch.zeros_like(model.observer_state)
    logits2, _ = model(test_input)
    
    # Test 3: Large observer
    model.observer_state = torch.randn_like(model.observer_state) * 10
    logits3, _ = model(test_input)
    
    # Compare outputs
    diff_zero = torch.norm(logits1 - logits2).item()
    diff_large = torch.norm(logits1 - logits3).item()
    
    print(f"\nLogit differences:")
    print(f"  Normal vs Zero observer: {diff_zero:.4f}")
    print(f"  Normal vs Large observer: {diff_large:.4f}")
    
    if diff_zero < 0.1:
        print("\n❌ PROBLEM: Observer has minimal influence on output!")
        print("   Processor is ignoring observer context")
    else:
        print("\n✓ Observer influences output")
    
    # Test observer update magnitude
    model.observer_state = torch.randn_like(model.observer_state) * 0.1
    obs_before = model.observer_state.clone()
    _, _ = model(test_input)
    obs_after = model.observer_state.clone()
    
    update_magnitude = torch.norm(obs_after - obs_before).item()
    print(f"\nObserver update magnitude: {update_magnitude:.4f}")
    
    if update_magnitude < 0.01:
        print("❌ PROBLEM: Observer barely updates!")
    else:
        print("✓ Observer updates significantly")


def diagnose_task_design():
    """Check if distractor task actually requires memory"""
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Task Design Analysis")
    print("="*70)
    
    from run_distractor_test import generate_needle_haystack_task
    
    # Generate sample task
    input_ids, target_ids = generate_needle_haystack_task(1, 20, n_distractors=10)
    
    print("\nSample sequence:")
    print(f"Input:  {input_ids[0].tolist()}")
    print(f"Target: {target_ids[0].tolist()}")
    
    # Find where answer appears
    answer_positions = []
    for i, (inp, tgt) in enumerate(zip(input_ids[0], target_ids[0])):
        if tgt != -100:
            answer_positions.append((i, inp.item(), tgt.item()))
    
    print(f"\nAnswer positions: {answer_positions}")
    
    # Check if answer is in input
    if len(answer_positions) > 0:
        ans_pos, ans_input, ans_target = answer_positions[0]
        if ans_input == ans_target:
            print("\n❌ PROBLEM: Answer is directly in input!")
            print("   Model can just copy, no memory needed")
        else:
            print("\n✓ Answer requires memory/computation")


def diagnose_window_constraint():
    """Check if window actually limits context"""
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Window Constraint Test")
    print("="*70)
    
    from rssa_v3 import WindowedAttention
    
    window_size = 128
    seq_len = 200
    
    attn = WindowedAttention(d_model=256, n_heads=8, window_size=window_size)
    
    # Create test input
    x = torch.randn(1, seq_len, 256)
    
    # Get attention mask
    mask = attn._create_window_mask(seq_len, x.device)
    
    # Check if tokens beyond window can attend
    for i in [50, 100, 150]:
        can_attend_to = mask[i].sum().item()
        max_lookback = min(i + 1, window_size)
        
        print(f"\nToken {i} can attend to {can_attend_to} tokens")
        print(f"  Expected (window={window_size}): {max_lookback}")
        
        if can_attend_to > max_lookback * 1.1:
            print(f"  ❌ PROBLEM: Attending beyond window!")


def main():
    print("="*70)
    print("RSSA IMPLEMENTATION DIAGNOSTICS")
    print("="*70)
    
    diagnose_observer_influence()
    diagnose_task_design()
    diagnose_window_constraint()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nRun diagnostics to identify implementation issues.")
    print("Look for ❌ PROBLEM markers above.")


if __name__ == "__main__":
    main()
