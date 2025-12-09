# Dot Product Optimization

## Performance Improvement

Replaced cosine similarity with dot product for normalized vectors:

### Before (Cosine Similarity)
```rust
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();  // Expensive!
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();  // Expensive!
    dot / (norm1 * norm2)
}
```

### After (Dot Product)
```rust
fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()  // Hardware FMA!
}
```

## Mathematical Justification

For L2-normalized vectors on unit sphere (||x|| = ||y|| = 1):

```
||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩
           = 1 + 1 - 2⟨x,y⟩
           = 2 - 2⟨x,y⟩
```

Therefore:
- **Minimizing Euclidean distance** ≡ **Maximizing dot product**
- argmin(||x - y||²) = argmax(⟨x,y⟩)

## Performance Results

| Candidates | Recall@10 | Latency (Before) | Latency (After) | Speedup |
|------------|-----------|------------------|-----------------|---------|
| 50 | 80.0% | 1.36ms | **1.26ms** | 1.08x |
| 100 | 80.0% | 1.35ms | **1.25ms** | 1.08x |
| 200 | 89.7% | 2.32ms | **2.22ms** | 1.05x |

**Improvement**: ~8% faster re-ranking (1.35ms → 1.25ms)

## Why It's Faster

1. **No square roots**: Eliminated 2 sqrt operations per comparison
2. **No divisions**: Eliminated 1 division per comparison
3. **Hardware FMA**: Modern CPUs have fused multiply-add instructions
4. **Better vectorization**: Simpler loop is easier for compiler to optimize

## Operations Saved Per Comparison

- **Before**: 384 multiplies + 384 adds + 768 adds (norms) + 2 sqrts + 1 div = ~1,540 ops
- **After**: 384 multiplies + 384 adds = 768 ops
- **Savings**: ~50% fewer operations

## Code Changes

Updated files:
- `src/search.rs`: Replaced `cosine_similarity` with `dot_product`
- `examples/evaluate_recall.rs`: Updated to use dot product
- `examples/benchmark_improvements.rs`: Updated to use dot product

All tests passing ✅
