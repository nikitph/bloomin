# GPT-2 Compression Validation Results

## ❌ **HYPOTHESIS REJECTED**

**Finding**: Variance explained ≠ Retrieval quality preserved

## Results Summary

| Dimension | Variance | Recall@10 | Compression |
|-----------|----------|-----------|-------------|
| 4D | 86.0% | 19.7% | 192x |
| 8D | 93.3% | 21.9% | 96x |
| **12D** | **95.3%** | **25.2%** | **64x** |
| 16D | 96.2% | 24.0% | 48x |
| 24D | 97.3% | 24.2% | 32x |
| 32D | 97.9% | 27.1% | 24x |
| 48D | 98.5% | 24.0% | 16x |
| 64D | 98.8% | 25.0% | 12x |
| 96D | 99.3% | 24.1% | 8x |
| 128D | 99.5% | 25.3% | 6x |

## Critical Insights

### 1. **Variance ≠ Retrieval Quality**
- 12D captures 95% of variance
- But only preserves 25% of retrieval quality
- **Gap**: 95% variance → 25% recall (70% loss!)

### 2. **Plateau Effect**
- Recall stays ~20-27% across ALL compression levels
- Even 128D (99.5% variance) only gets 25% recall
- Suggests PCA is fundamentally wrong approach for GPT-2

### 3. **Why This Happens**
GPT-2 embeddings likely have:
- **High variance in irrelevant dimensions**: PCA captures variance, not semantic structure
- **Semantic information in low-variance dimensions**: The 5% "noise" may contain critical ranking information
- **Non-linear structure**: PCA assumes linear subspace, but semantic similarity may be non-linear

### 4. **Comparison to BERT**
- BERT: 361D intrinsic dimension (47% of 768D)
- GPT-2: 12D intrinsic dimension (1.6% of 768D)
- **But**: GPT-2's low intrinsic dimension is MISLEADING for retrieval!

## Implications

### ❌ **Don't Use**:
- PCA compression for GPT-2 embeddings
- Variance explained as proxy for retrieval quality
- Intrinsic dimension as compression target

### ✅ **Better Approaches**:
1. **Supervised dimensionality reduction**: Train on retrieval task
2. **Product quantization**: Preserve distances better than PCA
3. **Keep full 768D**: The "compression opportunity" doesn't exist
4. **Use BERT instead**: If compression is critical, BERT's 361D → ~200D might work better

## Conclusion

**The 64x compression opportunity for GPT-2 is NOT viable.**

While GPT-2 has a 12D intrinsic dimension (95% variance), this does NOT preserve retrieval quality. The hypothesis that "95% variance = 90% recall" is **fundamentally flawed**.

**Key Lesson**: Intrinsic dimension (variance-based) is useful for understanding representation structure, but NOT a reliable predictor of task performance after compression.

## Next Steps

1. Test supervised compression methods (e.g., contrastive learning)
2. Investigate why GPT-2 has such concentrated variance but poor compression
3. Compare with BERT compression (361D → 200D might preserve 90% recall)
4. Consider task-specific compression instead of generic PCA
