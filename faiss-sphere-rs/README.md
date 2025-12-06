# FAISS-Sphere (Rust)

High-performance spherical vector search exploiting K=1 geometry.

## Features

- **2-3× faster search** via intrinsic-dimensional projection
- **2-3× memory reduction** 
- **95-99% recall** maintained
- **Zero-cost abstractions** with Rust
- **SIMD optimizations** via ndarray
- **Parallel processing** with Rayon

## Quick Start

```rust
use faiss_sphere::{IntrinsicProjector, SphericalIndex};

// Create projector
let mut projector = IntrinsicProjector::new(768, 320);
projector.train(&training_data)?;

// Project database
let data_projected = projector.project_parallel(&database)?;

// Build index
let mut index = SphericalIndex::new(320);
index.add(data_projected)?;

// Search
let queries_projected = projector.project(&queries)?;
let (distances, indices) = index.search_parallel(&queries_projected, 10)?;
```

## Benchmark

```bash
cargo run --release --example benchmark
```

Expected results:
- Speedup: 2-3×
- Memory: 2-3× smaller
- Recall: 95-99%

## Build

```bash
cargo build --release
cargo test
```

## Performance

Rust implementation achieves **true 2-3× speedup** through:
- Zero-cost abstractions
- SIMD vectorization
- Parallel processing
- Cache-friendly memory layout

## License

MIT
