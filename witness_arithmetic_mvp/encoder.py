import mmh3
from bitarray import bitarray

M = 256   # bits
K = 3     # hashes

def encode(witnesses: set[str]) -> bitarray:
    """
    Encodes a set of witnesses into a bitarray (Bloom filter).
    """
    bits = bitarray(M)
    bits.setall(0)

    for w in witnesses:
        for i in range(K):
            # Seed mmh3 with i to simulate K independent hash functions
            # We use the string witness as the key
            h = mmh3.hash(w, seed=i) % M
            bits[h] = 1

    return bits
