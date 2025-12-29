import numpy as np
import hashlib
from collections import defaultdict
import time

class GlobalHasher:
    """Precomputes hashes for all node IDs to avoid MD5 overhead in loops."""
    def __init__(self, n, m, k):
        self.n = n
        self.m = m
        self.k = k
        self.hash_table = np.zeros((n, k), dtype=int)
        for i in range(n):
            h = hashlib.md5(str(i).encode()).hexdigest()
            for j in range(k):
                self.hash_table[i, j] = int(h[j*8:(j+1)*8], 16) % m

    def get_hashes(self, node_id):
        return self.hash_table[node_id]

class CountingBloomFilter:
    """Counting Bloom Filter to support true deletions."""
    def __init__(self, m, k, global_hasher=None):
        self.m = m
        self.k = k
        self.counters = np.zeros(m, dtype=int)
        self.global_hasher = global_hasher
    
    def _hashes(self, x):
        if self.global_hasher:
            return self.global_hasher.get_hashes(x)
        h = hashlib.md5(str(x).encode()).hexdigest()
        return [int(h[i:i+8], 16) % self.m for i in range(0, min(len(h), self.k*8), 8)]

    def add(self, x):
        for h in self._hashes(x):
            self.counters[h] += 1

    def remove(self, x):
        for h in self._hashes(x):
            if self.counters[h] > 0:
                self.counters[h] -= 1

    def contains(self, x):
        # Optimized contains: check if all precomputed hash positions in counters are > 0
        hashes = self._hashes(x)
        for h in hashes:
            if self.counters[h] == 0:
                return False
        return True

    def union(self, other):
        self.counters = np.maximum(self.counters, other.counters)

    def is_empty(self):
        return not np.any(self.counters > 0)

    def candidates(self, n_range):
        # We can optimize this further if n_range is full range(self.n)
        return [u for u in n_range if self.contains(u)]

class AFDR:
    """
    Approximate Fully-Dynamic Reachability (AFDR)
    Upgraded with Counting Bloom Filters and GlobalHasher for performance.
    """
    def __init__(self, n, bloom_m=256, bloom_k=3):
        self.n = n
        self.m = bloom_m
        self.k = bloom_k
        self.logN = int(np.ceil(np.log2(n))) + 1
        
        # Precompute hashes
        self.hasher = GlobalHasher(n, self.m, self.k)
        
        # Bloom[u][k] stores reachability for paths of length <= 2^k
        self.sketches = [[CountingBloomFilter(self.m, self.k, self.hasher) for _ in range(self.logN)] for _ in range(self.n)]
        
        self.active_edges = set()
        self.op_count = 0

    def insert_edge(self, u, v, propagate=True):
        if (u, v) in self.active_edges:
            return
        
        self.active_edges.add((u, v))
        self.sketches[u][0].add(v)
        if propagate:
            self._propagate(u)

    def delete_edge(self, u, v, propagate=True):
        if (u, v) in self.active_edges:
            self.active_edges.remove((u, v))
            self.sketches[u][0].remove(v)
            if propagate:
                self._propagate(u)

    def _propagate(self, u):
        n_range = range(self.n)
        for k in range(1, self.logN):
            self.sketches[u][k] = CountingBloomFilter(self.m, self.k, self.hasher)
            self.sketches[u][k].union(self.sketches[u][k-1])
            
            candidates = self.sketches[u][k-1].candidates(n_range)
            for x in candidates:
                self.sketches[u][k].union(self.sketches[x][k-1])

    def reachable(self, u, v):
        for k in range(self.logN):
            if self.sketches[u][k].contains(v):
                return True
        return False

    def epoch_rebuild(self):
        self.sketches = [[CountingBloomFilter(self.m, self.k, self.hasher) for _ in range(self.logN)] for _ in range(self.n)]
        for u, v in self.active_edges:
            self.sketches[u][0].add(v)
            
        n_range = range(self.n)
        for k in range(1, self.logN):
            for u in n_range:
                self.sketches[u][k].union(self.sketches[u][k-1])
                candidates = self.sketches[u][k-1].candidates(n_range)
                for x in candidates:
                    self.sketches[u][k].union(self.sketches[x][k-1])
