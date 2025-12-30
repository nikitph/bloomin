# Minimal Viable Composer (MVC)

## Design Philosophy
- **Input:** 5 concrete algorithm implementations (Bloom, CountingBloom, MinHash, QuotientBloom, CuckooFilter).
- **Enumerate:** Try 4 composition operators mechanically.
- **Score:** Information theory + computational complexity.
- **Output:** Top-5 ranked hybrids with working code.

## Operators
- **⊗ (Tensor):** Independent parallel composition.
- **∘ (Sequential):** Pipeline composition with short-circuiting.
- **⊕ (Direct Sum):** Choice-at-query-time composition.
- **× (Pullback):** Constraint intersection semantics.
- **⊕_F (Fusion):** Merged components with structural/computational sharing.

## Scoring System (Refined)
- **Synergy:** Measures actual bit-level redundancy using information theory (-H_savings).
- **Novelty:** Penalizes trivial unions; rewards compositions with truly emergent properties.
- **Complexity:** Weights for computational speedups (e.g. early termination).

## Benchmarking Suite
The framework now includes a built-in benchmark that measures:
- **False Positive Rate (FPR)**
- **Query Latency (µs)**
- **Memory Consumption (Bytes)**

## Mechanistic Interpretability (Shadow Theory)
The MVC framework now supports "Shadow Theory" based mechanistic interpretability.
- **LIFT → DECOMPOSE → INTERPRET**: Map neural network weights to parent structures in Representation Theory, Differential Geometry, and RG Flow.
- **Interpretable by Design**: Build networks using compositional primitives that are interpretable by construction.
- **Circuit Discovery**: Automatically discover and visualize the irreducible circuits within a model.

See `mechanistic.py` and `interpretable_transformer.py` for implementation details.
Run `verify_mechanistic.py` for a full demonstration.
