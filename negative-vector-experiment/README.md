# Negative Vector Deletion Experiment

**Destructive Interference in Vector Retrieval: A Separation Result for Machine Unlearning**

This experiment provides a constructive proof that inserting negated vectors into ANN indices (like FAISS) does **not** implement deletion, while operator-based data structures (OBDS) with signed superposition achieve true O(1) unlearning via destructive interference.

## The Core Question

> Can we achieve deletion semantics through insertion alone?

Specifically: if we insert a "negative" or "anti-" vector `-v` into a vector database, does this cancel the original vector `v`'s contribution to retrieval?

**Answer:**
- **No** for all ANN indices with pointwise selection semantics (FAISS, ScaNN, HNSW)
- **Yes** for operator-based data structures (OBDS) with signed superposition

## Theoretical Foundation

### Selection vs. Field Semantics

The fundamental distinction is between two computational models:

| Model | Formula | Behavior |
|-------|---------|----------|
| **ANN Indices** | `f(q) = argmax_{x_i ∈ X} s(q, x_i)` | Selection over discrete elements |
| **OBDS** | `f(q) = argmax_x φ(x)` where `φ = Σ w_i K(·, c_i)` | Extremum of a continuous field |

### Key Insight

- **Selection operators** cannot implement cancellation because scores are computed independently
- **Field operators** implement cancellation naturally through linear superposition

### Main Theorems

**Theorem 5 (Impossibility of Insertion-Based Deletion in ANN Indices):**
For any pointwise ANN index with query semantics `I(q) = argmax_{x_i ∈ X} s(q, x_i)`, there exists no sequence of insertions that can make a stored element unretrievable without explicit deletion or query-time filtering.

**Theorem 7 (Constant-Time Unlearning via Signed Superposition):**
For an OBDS with field `φ(x) = Σ w_i K(x, c_i)` where K is a localized kernel, inserting a kernel with weight `-w_j` at center `c_j` removes the attractor in O(1) time.

## Experimental Setup

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D` | 128 | Embedding dimension |
| `N` | 100,000 | Dataset size |
| `K` | 10 | Top-k for retrieval |
| `σ` | 0.1 | Kernel bandwidth for OBDS |
| `η` | 0.01 | Learning rate for gradient ascent |
| `steps` | 100 | Maximum gradient ascent iterations |

### Protocol

1. **Baseline**: Query for target vector `v_42`, verify retrievability
2. **Attempted deletion**: Insert negated vector `-v_42`
3. **Post-deletion query**: Repeat query, measure retrievability

## Building and Running

### Prerequisites

- Rust toolchain (1.70+)

### Build

```bash
cd negative-vector-experiment
cargo build --release
```

### Run

```bash
./target/release/negative-vector-experiment
```

## Results

### FAISS-Style Index (Inner Product)

| Phase | Target Retrievable | Similarity Score |
|-------|-------------------|------------------|
| Before negative insertion | Yes (Rank 1) | 0.9939 |
| After negative insertion | **Yes (Rank 1)** | 0.9939 |

The negative vector is treated as an **independent point** with no effect on the original. The anti-vector doesn't even appear in top-10 results (its inner product with the query is negative).

### OBDS Field Model (Gaussian Kernels)

| Phase | Field Value at Target | Attractor Status |
|-------|----------------------|------------------|
| Before negative kernel | 1.0000 | Yes |
| After negative kernel | **0.0000** | **No** |

Exact cancellation achieved. Gradient ascent from the query no longer converges to the target.

### Comparison Summary

```
┌────────────────────────────────────┬──────────┬──────────┐
│ Metric                             │  FAISS   │   OBDS   │
├────────────────────────────────────┼──────────┼──────────┤
│ Target retrievable after deletion  │    ❌    │    ✅    │
│ Rebuild required for deletion      │    ✅    │    ❌    │
│ Deletion cost                      │  O(N)    │   O(1)   │
│ Provable guarantee                 │    ❌    │    ✅    │
│ GDPR/CCPA safe deletion            │    ❌    │    ✅    │
│ Supports destructive interference  │    ❌    │    ✅    │
└────────────────────────────────────┴──────────┴──────────┘
```

## Implementation Details

### FAISS-Like Index (`FaissLikeIndex`)

A flat index using inner product similarity:
- Vectors stored as independent points
- Search computes `argmax_i ⟨q, v_i⟩` over all stored vectors
- Parallel computation via `rayon`

### OBDS Field Model (`ObdsField`)

A Gaussian kernel field supporting signed weights:
- Field: `φ(x) = Σ w_i · exp(-||x - c_i||² / 2σ²)`
- Gradient: `∇φ(x) = Σ w_i · K(x, c_i) · (c_i - x) / σ²`
- Retrieval via gradient ascent to local maximum
- Deletion via insertion of negative kernel (w = -1)

## Implications

### GDPR/CCPA Compliance

Current vector databases cannot guarantee data erasure without index reconstruction. OBDS provides cryptographically verifiable deletion: the negative weight can be logged as proof that cancellation was applied.

### Federated Learning

Clients can withdraw their data without coordinator involvement by broadcasting deletion kernels.

### RAG Systems

Retrieval-augmented generation systems can implement "context amnesia" for sensitive queries by maintaining deletion masks as negative kernels.

### Hybrid Architecture

A practical deployment might combine both approaches:
1. Use FAISS for high-throughput approximate search
2. Maintain an OBDS "deletion mask" of negative kernels
3. At query time, filter FAISS results through the deletion mask

## Why This Matters

This is not merely an optimization but a **new computational primitive**:

| Deletion Type | Mechanism | Properties |
|---------------|-----------|------------|
| **Structural** | Modify data structure (remove pointers, rebuild indices) | Destructive, requires locks |
| **Algebraic** | Add inverse element (append-only) | Reversible, concurrent, auditable |

Algebraic deletion enables:
- **Versioned memory**: All operations logged; any state reconstructable
- **Reversible unlearning**: Deletion undone by removing negative kernel
- **Audit trails**: Compliance verification without raw data access
- **Concurrent operations**: Append-only is naturally lock-free

## Paper Reference

For full theoretical treatment, proofs, and extended discussion, see:

**[Destructive Interference in Vector Retrieval: A Separation Result for Machine Unlearning](paper/main.pdf)**

Nikit Phadke (nikitph@gmail.com)

### Citation

```bibtex
@article{phadke2024destructive,
  title={Destructive Interference in Vector Retrieval: A Separation Result for Machine Unlearning},
  author={Phadke, Nikit},
  year={2024}
}
```

## Dependencies

```toml
[dependencies]
rand = "0.8"
rand_distr = "0.4"
rayon = "1.8"
```

## License

See repository root for license information.
