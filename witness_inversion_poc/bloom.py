from .hashes import get_hashes

def bloom_encode(witnesses: set[str], m=1024, k=3) -> list[int]:
    """
    Encodes witnesses into a bit vector of size m using k hashes.
    """
    bits = [0] * m
    hashes = get_hashes(k)
    for w in witnesses:
        for h in hashes:
            pos = h(w) % m
            bits[pos] = 1
    return bits

def bloom_decode(bits: list[int], vocabulary: list[str], k=3) -> set[str]:
    """Basic inversion: Any witness with all k-bits set is returned."""
    hashes = get_hashes(k)
    decoded = set()
    m = len(bits)
    for word in vocabulary:
        if all(bits[h(word) % m] == 1 for h in hashes):
            decoded.add(word)
    return decoded

def rank_witnesses(bits: list[int], vocabulary: list[str], k=3) -> list[tuple[str, float]]:
    """
    Ranks witnesses by confidence.
    For Witness Arithmetic, confidence is binary (1.0 or 0.0), but in 
    crowded filters, we can compute local density to adjust.
    Here we return (witness, confidence).
    """
    hashes = get_hashes(k)
    m = len(bits)
    results = []
    for word in vocabulary:
        if all(bits[h(word) % m] == 1 for h in hashes):
            results.append((word, 1.0))
        else:
            # Partial hit (might be useful for fuzzy searching)
            hits = sum(1 for h in hashes if bits[h(word) % m] == 1)
            if hits > 0:
                results.append((word, hits/k))
    return sorted(results, key=lambda x: x[1], reverse=True)

def sample_witness_sets(bits: list[int], vocabulary: list[str], k=3, num_samples=3):
    """
    Generates multi-sample witness sets from one sketch.
    Stochastic sampling: for each bit-flip or subset, we get a new 'creative' valid set.
    """
    base_witnesses = list(bloom_decode(bits, vocabulary, k))
    samples = []
    import random
    
    for _ in range(num_samples):
        # Sample a subset or slightly mutate for variety
        sample_size = random.randint(max(1, len(base_witnesses)-1), len(base_witnesses))
        sample = set(random.sample(base_witnesses, sample_size))
        samples.append(sample)
    return samples
