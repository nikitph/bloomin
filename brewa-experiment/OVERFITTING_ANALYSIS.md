# Overfitting Analysis and Fix

## Critical Finding: Severe Overfitting

**Results:**
- Training Recall: 100.0%
- Test Recall (same clusters): 8.4%
- Test Recall (new clusters): 10.8%

**Generalization gap: 91.6%** - This is catastrophic overfitting!

---

## Why This Happened

### Root Cause: Model Memorized Training Data

The model didn't learn "similar embeddings should stay similar" - it learned "these specific 5,000 embeddings map to these specific 256-dimensional points."

### Evidence

1. **Perfect training recall (100%)** - Model fit training data exactly
2. **Terrible test recall (8%)** - Model can't handle new data
3. **Similar performance on new clusters (10.8%)** - Not learning general structure

### Why Overfitting Occurred

1. **Too much capacity**: 2-layer MLP with 512 hidden units can memorize 5,000 samples
2. **Not enough regularization**: dropout=0.1 is too low
3. **No data augmentation**: Same samples every epoch
4. **Perfect separability**: 50 well-separated clusters are too easy

---

## Comparison with Random Hadamard

| Method | Training | Test (same) | Test (new) | Generalization |
|--------|----------|-------------|------------|----------------|
| Random Hadamard | 27% | 27% | 27% | ✅ Perfect (no overfitting) |
| Learned (current) | 100% | 8.4% | 10.8% | ❌ Terrible (severe overfitting) |

**Paradox**: Random projection generalizes perfectly but has low recall. Learned projection has high training recall but doesn't generalize!

---

## The Fix: Regularization + Augmentation

### 1. Stronger Regularization

```python
encoder = LearnedContinuousREWAEncoder(
    d_model=768,
    m_dim=256,
    dropout=0.5,  # Increase from 0.1 to 0.5
    noise_std=0.1,  # Increase from 0.01 to 0.1
)
```

### 2. Data Augmentation

```python
def augment_embedding(emb):
    # Random rotation
    angle = torch.rand(1) * 0.1
    rotation = create_rotation_matrix(angle)
    emb = rotation @ emb
    
    # Random noise
    emb = emb + torch.randn_like(emb) * 0.05
    
    # Re-normalize
    return emb / emb.norm()
```

### 3. Simpler Architecture

```python
# Use single linear layer instead of 2-layer MLP
encoder = LearnedContinuousREWAEncoder(
    d_model=768,
    m_dim=256,
    use_mlp=False,  # Single linear layer
)
```

### 4. Early Stopping

```python
# Stop when validation recall stops improving
best_val_recall = 0
patience = 5
for epoch in range(100):
    train_loss = train_epoch()
    val_recall = evaluate_on_validation()
    
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        break  # Stop training
```

### 5. More Diverse Training Data

```python
# Mix of cluster configurations
# - Different cluster sizes
# - Different intra-cluster variance
# - Different inter-cluster distances
```

---

## Expected Results After Fix

### Conservative Estimate

| Method | Training | Test (same) | Test (new) |
|--------|----------|-------------|------------|
| Fixed Learned REWA | 70-80% | 60-70% | 50-60% |

### Optimistic Estimate

| Method | Training | Test (same) | Test (new) |
|--------|----------|------------|------------|
| Fixed Learned REWA | 85-90% | 75-85% | 65-75% |

**Goal**: Achieve >70% test recall with <10% generalization gap

---

## Alternative Approach: Hybrid

Since random projection generalizes perfectly (27% everywhere), we can combine:

```python
class HybridREWAEncoder(nn.Module):
    def __init__(self, d_model, m_dim):
        # Part 1: Random projection (generalizes)
        self.random_proj = HadamardTransform(d_model, m_dim // 2)
        
        # Part 2: Learned projection (high capacity)
        self.learned_proj = nn.Linear(d_model, m_dim // 2)
    
    def forward(self, x):
        # Concatenate random + learned
        random_part = self.random_proj(x)  # [B, N, m_dim/2]
        learned_part = self.learned_proj(x)  # [B, N, m_dim/2]
        
        combined = torch.cat([random_part, learned_part], dim=-1)
        return F.normalize(combined, dim=-1)
```

**Expected**: 
- Random part ensures generalization (27% floor)
- Learned part boosts performance (up to 60-70%)
- Combined: 50-60% with good generalization

---

## Theoretical Insight

### Why Random Generalizes

Random projection is **data-independent**:
- Same transform for all inputs
- No parameters to overfit
- Johnson-Lindenstrauss guarantee

### Why Learned Overfits

Learned projection is **data-dependent**:
- Optimizes for specific training samples
- Can memorize instead of generalize
- Needs regularization to prevent overfitting

### The Trade-off

```
Generalization ←→ Capacity

Random:  Perfect generalization, low capacity (27%)
Learned: High capacity (100%), poor generalization (8%)
Hybrid:  Balance (50-60% with good generalization)
```

---

## Next Steps

1. **Immediate**: Re-train with stronger regularization
2. **Short-term**: Implement hybrid approach
3. **Long-term**: Test on real BERT embeddings (harder to overfit)

---

## Key Lesson

**High training accuracy ≠ Success**

The model achieving 100% training recall was a **red flag**, not a success. We should have:
1. Used validation set
2. Monitored generalization gap
3. Applied stronger regularization from the start

**This is a classic machine learning lesson**: Always test on held-out data!

---

## Status

- ❌ Learned REWA (current): Overfitted, not usable
- ⏳ Learned REWA (fixed): Need to re-train with regularization
- ✅ Random REWA: Generalizes but low recall (27%)
- 🎯 Hybrid REWA: Best of both worlds (target)

---

**Conclusion**: We need to fix the overfitting before claiming success. The 100% training recall was misleading - the model memorized rather than learned.
