import hashlib

def get_hashes(num_hashes=3):
    """
    Returns a list of lambda functions, each representing a hash function h_i.
    """
    def h_i(value, i):
        # Deterministic hashing with seed/index
        h = hashlib.sha256(f"seed_{i}_{value}".encode()).digest()
        return int.from_bytes(h, 'big')
    
    return [lambda v, i=i: h_i(v, i) for i in range(num_hashes)]
