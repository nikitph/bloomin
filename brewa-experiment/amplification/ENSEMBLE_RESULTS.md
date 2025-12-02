# Ensemble Evaluation Results

## Executive Summary

We evaluated the ensemble of 3 checkpoints (epochs 25, 30, 35) using three different methodologies to ensure correctness and clarity.

**Key Findings:**
1. **Ensemble Achieved 79.2% Recall**: Using the correct zero-shot clustering evaluation (retrieval within unseen set), the ensemble improved performance from 78.6% (single model) to **79.2%**.
2. **User's Proposed Method Yielded 0%**: Retrieving unseen queries from the *seen* database yielded 0% recall because the category sets are disjoint (no overlap between seen and unseen classes).
3. **Validation of 78.6% Baseline**: The single-model results (78.4%, 78.5%, 78.6%) were re-confirmed, validating our previous reporting.

---

## Detailed Results

### Method 1 & 2: Retrieval from SEEN Database (User's Request)
*Goal: Retrieve unseen category documents from the training set.*

| Model | Recall@10 |
|-------|-----------|
| Single Models | 0.0% |
| Ensemble | 0.0% |

**Reason**: The 20 Newsgroups split used for zero-shot learning has **disjoint label sets**:
- **Seen (15)**: `rec.autos`, `sci.space`, `comp.graphics`, etc.
- **Unseen (5)**: `sci.electronics`, `talk.politics.guns`, etc.
- **Intersection**: None.

Since the training set contains *no* documents with unseen labels, it is impossible to retrieve a "correct" document (same label) from the seen database.

### Method 3: Retrieval from UNSEEN Database (Clustering Quality)
*Goal: Retrieve relevant documents from the unseen set itself (Leave-One-Out).*
*This measures how well the model clusters unseen concepts.*

| Model | Recall@10 | Improvement |
|-------|-----------|-------------|
| Epoch 25 | 78.4% | - |
| Epoch 30 | 78.5% | - |
| Epoch 35 | 78.6% | - |
| **Ensemble (Voting)** | **79.2%** | **+0.6%** |

**Conclusion**: The ensemble successfully combines the strengths of different training checkpoints, pushing the recall from 78.6% to **79.2%**.

---

## Technical Analysis

### Why the Ensemble Works
The ensemble voting mechanism (top-20 votes from each model) helps smooth out noise in the retrieval process. While single models might miss a relevant neighbor due to random projection noise or training fluctuations, the consensus of 3 models (trained with different adversarial weights and mixup states) provides a more robust signal.

### 79.2% vs 80% Target
We are now only **0.8%** away from the 80% target.
- **Baseline (Hybrid REWA)**: 73.7%
- **Adversarial (Single)**: 78.6% (+4.9%)
- **Adversarial (Ensemble)**: 79.2% (+5.5%)

### Recommendation

1. **For Maximum Accuracy**: Deploy the **Ensemble (79.2%)**.
   - Pros: Highest recall, most robust.
   - Cons: 3x storage/compute cost (unless distilled).

2. **For Production Efficiency**: Deploy the **Epoch 35 Single Model (78.6%)**.
   - Pros: 3x faster, 3x smaller than ensemble.
   - Cons: Slightly lower recall (-0.6%).
   - **Verdict**: 78.6% is likely sufficient for most applications and maintains the "3x compression" value proposition perfectly.

## Files Generated
- `correct_ensemble_eval.py`: Script implementing all 3 evaluation methods.
- `correct_ensemble_v2.log`: Execution logs confirming the results.
