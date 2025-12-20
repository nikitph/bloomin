import random
from witness_inversion_poc.bloom import rank_witnesses

def inject_noise(bits, fraction=0.1):
    """Randomly flips bits to simulate Bloom filter saturation/noise."""
    mutated = list(bits)
    m = len(bits)
    n_to_flip = int(m * fraction)
    indices = random.sample(range(m), n_to_flip)
    for idx in indices:
        mutated[idx] = 1 - mutated[idx]
    return mutated

def get_mean_confidence(bits, vocabulary, k=3):
    """Calculates average confidence of decoded witnesses."""
    ranked = rank_witnesses(bits, vocabulary, k)
    if not ranked: return 0.0
    return sum(c for w, c in ranked) / len(ranked)

def print_test_header(name):
    print(f"\n--- [STRESS_TEST] {name} ---")
