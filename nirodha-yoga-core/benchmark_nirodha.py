import torch
from nirodha.core import CognitiveState, YogaRegulator, Observer

def test_nirodha_convergence():
    print("Running Benchmark 1: Fluctuation Decay...")
    C0 = torch.randn(512)
    C = C0 + torch.randn(512) * 10

    state = CognitiveState(C)
    yoga = YogaRegulator(beta=0.5)

    for _ in range(100):
        noise = torch.randn(512) * 0.1
        state = yoga(state, noise)

    norm_diff = torch.norm(state.C - C0)
    print(f"Final deviance: {norm_diff:.4f}")
    assert norm_diff < 0.5
    print("Benchmark 1 PASSED")

def test_no_forgetting():
    print("\nRunning Benchmark 2: Memory Retention...")
    C0 = torch.randn(512)
    signal = torch.randn(512)

    state = CognitiveState(C0 + signal)
    yoga = YogaRegulator(beta=10.0)

    # Apply zero update - regulator should preserve the signal via the suppressed Vritti
    state = yoga(state, torch.zeros_like(signal))

    cosine = torch.nn.functional.cosine_similarity(
        signal, state.C - state.C0, dim=0
    )
    print(f"Cosine similarity: {cosine:.4f}")
    assert cosine > 0.95
    print("Benchmark 2 PASSED")

def test_attachment_decay():
    print("\nRunning Benchmark 3: Reward Hacking Resistance...")
    C0 = torch.zeros(512)
    shortcut = torch.ones(512) * 100

    state = CognitiveState(C0 + shortcut)
    yoga = YogaRegulator(beta=1.0)

    for _ in range(50):
        state = yoga(state, torch.zeros_like(shortcut))

    norm_c = torch.norm(state.C)
    norm_shortcut = torch.norm(shortcut)
    print(f"Regulated norm: {norm_c:.4f}, Shortcut norm: {norm_shortcut:.4f}")
    assert norm_c < norm_shortcut
    print("Benchmark 3 PASSED")

def test_observer_constant():
    print("\nRunning Benchmark 4: Observer Invariance...")
    C0 = torch.randn(512)
    state = CognitiveState(C0)
    observer = Observer()
    yoga = YogaRegulator()

    o0 = observer(state.C, state.C0)

    for _ in range(50):
        state = yoga(state, torch.randn(512))

    o_final = observer(state.C, state.C0)
    diff = abs(o_final - o0)
    print(f"Observer change: {diff:.8f}")
    assert diff < 1e-6
    print("Benchmark 4 PASSED")

if __name__ == "__main__":
    try:
        test_nirodha_convergence()
        test_no_forgetting()
        test_attachment_decay()
        test_observer_constant()
        print("\nAll benchmarks passed successfully.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        exit(1)
