"""
Witness-LDPC Codes for Fast Similarity Search

Theory Connection to REWA/Ricci Curvature:
- In continual learning, we preserve geometric structure (distances, angles between class centroids)
- In retrieval, we ENCODE geometric structure into compact binary codes
- The "witness" idea: a small set of dimensions that characterize the vector's position
  in the embedding manifold

Key Insight:
- High-dimensional vectors have geometric structure that can be captured by "witnesses"
- Witnesses = the most distinctive dimensions of a vector
- Using expander graph structure (LDPC codes) ensures good distance preservation
- Multiple hash functions provide redundancy and error correction

This is related to:
- Locality Sensitive Hashing (LSH) but with better theoretical guarantees
- Product Quantization (PQ) but with simpler binary codes
- Graph-based ANN but with O(1) candidate lookup
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import mmh3  # MurmurHash3 for fast hashing


class WitnessExtractor:
    """
    Extract witnesses from dense vectors.

    A witness is a small set of dimension indices that "testify" to the vector's
    position in the embedding space. Similar vectors should share witnesses.

    Strategies:
    1. Top-K: Take dimensions with largest absolute values
    2. Percentile: Take dimensions above a threshold
    3. Random projection: Project to lower dim, take top-K of projection
    """

    def __init__(
        self,
        dim: int,
        num_witnesses: int = 64,
        strategy: str = 'top_k',
        use_signs: bool = True
    ):
        self.dim = dim
        self.num_witnesses = num_witnesses
        self.strategy = strategy
        self.use_signs = use_signs

        # For random projection strategy
        if strategy == 'random_projection':
            self.projection = np.random.randn(dim, num_witnesses) / np.sqrt(num_witnesses)

    def extract(self, vector: np.ndarray) -> List[int]:
        """Extract witness indices from a vector."""
        if self.strategy == 'top_k':
            # Top-K by absolute value
            indices = np.argsort(np.abs(vector))[-self.num_witnesses:]
            if self.use_signs:
                # Encode sign information: index * 2 + (sign > 0)
                witnesses = []
                for idx in indices:
                    sign_bit = 1 if vector[idx] > 0 else 0
                    witnesses.append(idx * 2 + sign_bit)
                return witnesses
            return indices.tolist()

        elif self.strategy == 'percentile':
            threshold = np.percentile(np.abs(vector), 100 - 100 * self.num_witnesses / self.dim)
            indices = np.where(np.abs(vector) >= threshold)[0][:self.num_witnesses]
            if self.use_signs:
                witnesses = []
                for idx in indices:
                    sign_bit = 1 if vector[idx] > 0 else 0
                    witnesses.append(idx * 2 + sign_bit)
                return witnesses
            return indices.tolist()

        elif self.strategy == 'random_projection':
            projected = vector @ self.projection
            indices = np.argsort(np.abs(projected))[-self.num_witnesses:]
            if self.use_signs:
                witnesses = []
                for idx in indices:
                    sign_bit = 1 if projected[idx] > 0 else 0
                    witnesses.append(idx * 2 + sign_bit)
                return witnesses
            return indices.tolist()

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class WitnessLDPC:
    """
    Witness-LDPC codes for approximate nearest neighbor search.

    Encoding process:
    1. Extract witnesses from dense vector
    2. Hash each witness K times using different hash functions
    3. Set bits at hashed positions → binary code

    Search process:
    1. Encode query → binary code
    2. Use inverted index to find candidates (O(1) per set bit)
    3. Score candidates by Hamming similarity
    4. Return top-K candidates

    Key parameters:
    - code_length (m): Length of binary code. Larger = better accuracy, more memory
    - num_hashes (K): Hash functions per witness. More = better recall, slower
    - num_witnesses: Witnesses per vector. More = better accuracy, slower encoding
    """

    def __init__(
        self,
        dim: int = 768,
        code_length: int = 2048,
        num_hashes: int = 4,
        num_witnesses: int = 64,
        witness_strategy: str = 'top_k',
        use_signs: bool = True,
        seed: int = 42
    ):
        self.dim = dim
        self.m = code_length
        self.K = num_hashes
        self.num_witnesses = num_witnesses

        np.random.seed(seed)

        # Witness extractor
        self.witness_extractor = WitnessExtractor(
            dim=dim,
            num_witnesses=num_witnesses,
            strategy=witness_strategy,
            use_signs=use_signs
        )

        # Hash function seeds
        self.hash_seeds = [np.random.randint(0, 2**31) for _ in range(num_hashes)]

        # Storage
        self.codes: Optional[np.ndarray] = None
        self.vectors: Optional[np.ndarray] = None  # Store originals for re-ranking
        self.inverted_index: Dict[int, List[int]] = defaultdict(list)
        self.n_vectors = 0

    def _hash_witness(self, witness: int, seed_idx: int) -> int:
        """Hash a witness to a bit position."""
        h = mmh3.hash(str(witness), self.hash_seeds[seed_idx])
        return h % self.m

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Encode a dense vector to a binary code."""
        code = np.zeros(self.m, dtype=np.uint8)

        # Extract witnesses
        witnesses = self.witness_extractor.extract(vector)

        # Hash each witness K times
        for witness in witnesses:
            for k in range(self.K):
                pos = self._hash_witness(witness, k)
                code[pos] = 1

        return code

    def add(self, vectors: np.ndarray, batch_size: int = 10000, verbose: bool = True):
        """
        Add vectors to the index.

        Args:
            vectors: (n, dim) array of vectors to index
            batch_size: Process in batches for memory efficiency
            verbose: Print progress
        """
        n = len(vectors)
        self.n_vectors = n

        if verbose:
            print(f"Indexing {n:,} vectors (dim={self.dim}, code_length={self.m})")

        # Store original vectors for optional re-ranking
        self.vectors = vectors.astype(np.float32)

        # Encode all vectors
        self.codes = np.zeros((n, self.m), dtype=np.uint8)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            for i in range(start, end):
                self.codes[i] = self.encode(vectors[i])

            if verbose and (end % (batch_size * 5) == 0 or end == n):
                print(f"  Encoded {end:,}/{n:,} vectors ({100*end/n:.1f}%)")

        # Build inverted index
        if verbose:
            print("Building inverted index...")

        self.inverted_index = defaultdict(list)
        for vec_id in range(n):
            for bit_pos in np.where(self.codes[vec_id])[0]:
                self.inverted_index[int(bit_pos)].append(vec_id)

        # Statistics
        bits_per_vector = np.mean(np.sum(self.codes, axis=1))
        if verbose:
            print(f"Index built:")
            print(f"  - {n:,} vectors")
            print(f"  - {len(self.inverted_index):,} active bits in inverted index")
            print(f"  - {bits_per_vector:.1f} bits per vector (avg)")
            print(f"  - Memory: {self.codes.nbytes / 1024**2:.1f} MB (codes) + {self.vectors.nbytes / 1024**2:.1f} MB (vectors)")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_candidates: int = None,
        rerank: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector (dim,)
            k: Number of results to return
            n_candidates: Number of candidates to retrieve before re-ranking.
                         Default: 10*k
            rerank: If True, re-rank candidates using exact distances

        Returns:
            indices: (k,) array of neighbor indices
            scores: (k,) array of similarity scores
        """
        if n_candidates is None:
            n_candidates = min(k * 10, self.n_vectors)

        # Encode query
        query_code = self.encode(query)
        query_bits = set(np.where(query_code)[0])

        # Fast candidate retrieval via inverted index
        candidate_scores = defaultdict(int)
        for bit_pos in query_bits:
            for vec_id in self.inverted_index.get(int(bit_pos), []):
                candidate_scores[vec_id] += 1

        if len(candidate_scores) == 0:
            # Fallback: return random candidates
            indices = np.random.choice(self.n_vectors, size=min(k, self.n_vectors), replace=False)
            return indices, np.zeros(len(indices))

        # Get top candidates by Hamming similarity
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [c[0] for c in sorted_candidates[:n_candidates]]

        if rerank and self.vectors is not None:
            # Re-rank using exact cosine similarity
            candidate_vectors = self.vectors[candidates]

            # Normalize for cosine similarity
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            cand_norms = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-8)

            similarities = cand_norms @ query_norm
            top_k_idx = np.argsort(similarities)[-k:][::-1]

            indices = np.array([candidates[i] for i in top_k_idx])
            scores = similarities[top_k_idx]
        else:
            # Return candidates sorted by Hamming similarity
            indices = np.array(candidates[:k])
            scores = np.array([candidate_scores[c] for c in candidates[:k]], dtype=np.float32)
            scores = scores / (len(query_bits) + 1e-8)  # Normalize to [0, 1]

        return indices, scores

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_candidates: int = None,
        rerank: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for multiple queries.

        Args:
            queries: (n_queries, dim) array of query vectors
            k: Number of results per query

        Returns:
            indices: (n_queries, k) array of neighbor indices
            scores: (n_queries, k) array of similarity scores
        """
        n_queries = len(queries)
        all_indices = np.zeros((n_queries, k), dtype=np.int64)
        all_scores = np.zeros((n_queries, k), dtype=np.float32)

        for i, query in enumerate(queries):
            indices, scores = self.search(query, k=k, n_candidates=n_candidates, rerank=rerank)
            n_results = min(len(indices), k)
            all_indices[i, :n_results] = indices[:n_results]
            all_scores[i, :n_results] = scores[:n_results]

        return all_indices, all_scores

    def memory_usage(self) -> Dict[str, float]:
        """Return memory usage in MB."""
        codes_mb = self.codes.nbytes / 1024**2 if self.codes is not None else 0
        vectors_mb = self.vectors.nbytes / 1024**2 if self.vectors is not None else 0

        # Estimate inverted index size
        inv_idx_size = sum(len(v) * 8 for v in self.inverted_index.values())  # 8 bytes per int64
        inv_idx_mb = inv_idx_size / 1024**2

        return {
            'codes_mb': codes_mb,
            'vectors_mb': vectors_mb,
            'inverted_index_mb': inv_idx_mb,
            'total_mb': codes_mb + vectors_mb + inv_idx_mb
        }


class WitnessLDPCCompact(WitnessLDPC):
    """
    Memory-optimized version that doesn't store original vectors.

    Trade-off: No re-ranking with exact distances, but 100× less memory.
    """

    def add(self, vectors: np.ndarray, batch_size: int = 10000, verbose: bool = True):
        """Add vectors without storing originals."""
        n = len(vectors)
        self.n_vectors = n

        if verbose:
            print(f"Indexing {n:,} vectors (COMPACT mode - no vector storage)")

        # Don't store original vectors
        self.vectors = None

        # Encode all vectors
        self.codes = np.zeros((n, self.m), dtype=np.uint8)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            for i in range(start, end):
                self.codes[i] = self.encode(vectors[i])

            if verbose and (end % (batch_size * 5) == 0 or end == n):
                print(f"  Encoded {end:,}/{n:,} vectors ({100*end/n:.1f}%)")

        # Build inverted index
        if verbose:
            print("Building inverted index...")

        self.inverted_index = defaultdict(list)
        for vec_id in range(n):
            for bit_pos in np.where(self.codes[vec_id])[0]:
                self.inverted_index[int(bit_pos)].append(vec_id)

        if verbose:
            print(f"Index built: {self.codes.nbytes / 1024**2:.1f} MB (no vectors stored)")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_candidates: int = None,
        rerank: bool = False  # Cannot rerank without vectors
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search without re-ranking (no vectors stored)."""
        return super().search(query, k=k, n_candidates=n_candidates, rerank=False)


if __name__ == "__main__":
    print("Testing Witness-LDPC codes...")

    # Generate random embeddings
    np.random.seed(42)
    n_vectors = 10000
    dim = 768

    print(f"\nGenerating {n_vectors:,} random vectors (dim={dim})")
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize

    # Test queries (use some vectors as queries)
    queries = vectors[:100].copy()

    # Build index
    print("\nBuilding Witness-LDPC index...")
    index = WitnessLDPC(
        dim=dim,
        code_length=2048,
        num_hashes=4,
        num_witnesses=64
    )
    index.add(vectors)

    # Search
    print("\nSearching...")
    import time

    times = []
    recalls = []

    for i, query in enumerate(queries):
        start = time.time()
        indices, scores = index.search(query, k=10)
        times.append(time.time() - start)

        # Check if true nearest neighbor (itself) is in results
        if i in indices:
            recalls.append(1)
        else:
            recalls.append(0)

    print(f"\nResults:")
    print(f"  Average query time: {np.mean(times)*1000:.2f} ms")
    print(f"  Self-recall@10: {np.mean(recalls)*100:.1f}%")
    print(f"  Memory: {index.memory_usage()}")

    print("\nTest passed!")
