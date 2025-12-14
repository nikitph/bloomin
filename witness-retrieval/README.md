# Witness-LDPC: Fast Similarity Search

Information-theoretically grounded approximate nearest neighbor search using Witness codes.

## Theory Connection

Building on the REWA (Ricci-Enhanced Witness Alignment) framework from continual learning:

- **Continual Learning**: Preserve geometric structure (class centroids, distances, angles) to prevent forgetting
- **Similarity Search**: Encode geometric structure into compact binary codes for fast retrieval

Key insight: High-dimensional vectors have geometric structure captured by "witnesses" - the most distinctive dimensions. Hash witnesses using LDPC-like codes for good distance preservation.

## Benchmark Results

### 50K Vectors (768-dim)

| Method | Speedup | Recall@10 | Memory |
|--------|---------|-----------|--------|
| Brute Force | 1.0x | 100% | 146 MB |
| Witness-LDPC (m=2048, K=4, w=64) | **4.4x** | **94.8%** | 205 MB |
| Witness-LDPC (m=4096, K=4, w=64) | **5.8x** | **96.5%** | 218 MB |

### 100K Vectors (768-dim)

| Method | Speedup | Recall@10 | Memory |
|--------|---------|-----------|--------|
| Brute Force | 1.0x | 100% | 293 MB |
| Witness-LDPC (m=2048, K=4, w=64) | 4.3x | 73.3% | 409 MB |
| Witness-LDPC (m=4096, K=4, w=64) | **6.1x** | **75.3%** | 437 MB |
| Witness-LDPC (m=2048, K=6, w=96) | 1.2x | 77.2% | 509 MB |

## Key Parameters

- `code_length (m)`: Binary code length. More bits = better recall, more memory
- `num_hashes (K)`: Hash functions per witness. More = better recall, slower encoding
- `num_witnesses (w)`: Witnesses per vector. More = better accuracy, slower encoding

Sweet spot: **m=4096, K=4, w=64** gives ~6x speedup with ~75-96% recall depending on dataset size.

## Building

```bash
cargo build --release
```

## Running Benchmarks

```bash
# 50K vectors
./target/release/benchmark 50000 768 300

# 100K vectors
./target/release/benchmark 100000 768 500
```

## Architecture

```
src/
├── lib.rs           # Core WitnessLDPC index
├── hierarchical.rs  # Two-level hierarchical index (coarse + fine)
└── benchmark.rs     # Benchmark binary
```

## Algorithm

1. **Encode**: Extract top-k dimensions by magnitude → hash each K times → set bits
2. **Index**: Build inverted index mapping bit positions → vector IDs
3. **Search**: Encode query → lookup candidates via inverted index → re-rank by exact similarity

## Future Work

- [ ] Better hierarchical filtering (current coarse level loses too many candidates)
- [ ] SIMD-optimized Hamming distance
- [ ] Information-theoretic witness selection (not just top-k by magnitude)
- [ ] Product quantization for memory compression
