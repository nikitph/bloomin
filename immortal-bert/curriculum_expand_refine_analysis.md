# Analysis of Curriculum+Expand+Refine Results

**Run Date:** 2025-12-15 01:27:17
**Results File:** `curriculum_expand_refine_results_20251215_012717.json`

## Overview

This experiment evaluated a **Curriculum+Expand+Refine** continual learning model across **10 NLP tasks** in sequence.

**Task Order:** SST2 → IMDB → Yelp → CoLA → MRPC → QQP → RTE → QNLI → MNLI → AG_News

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average Final Accuracy** | 69.88% | Good overall retention |
| **Worst-Task Accuracy (WTA)** | 45.85% | RTE is the bottleneck |
| **Forgetting Index (FI)** | 2.44% | Very low forgetting |
| **Performance Variation (PV)** | 3.03% | Stable across tasks |

## Per-Task Final Accuracy

```
IMDB:    ████████████████████ 100.0%  ← Perfect retention
Yelp:    █████████████████    86.7%
AG_News: █████████████████    86.0%
SST2:    ████████████████     80.0%
CoLA:    ██████████████       70.7%
MRPC:    █████████████        67.3%
QQP:     ████████████         63.3%
QNLI:    ██████████           52.3%
MNLI:    █████████            46.7%
RTE:     █████████            45.8%  ← Worst performer
```

## Forgetting Analysis

### Best Retention (Low/Negative Forgetting)

| Task | Forgetting | Notes |
|------|------------|-------|
| IMDB | 0.0% | Perfect stability |
| CoLA | -1.0% | Backward transfer (improved!) |
| MNLI | -2.0% | Backward transfer (improved!) |
| QNLI | 0.67% | Nearly perfect |

### Most Affected by Forgetting

| Task | Forgetting |
|------|------------|
| QQP | 8.0% |
| MRPC | 4.67% |
| SST2 | 4.33% |
| RTE | 4.33% |

## Key Observations

1. **Sentiment tasks excel**: IMDB (100%), Yelp (86.7%), SST2 (80%), AG_News (86%) all perform well - likely due to task similarity enabling positive transfer.

2. **NLI tasks struggle**: RTE (45.8%), MNLI (46.7%), QNLI (52.3%) are the weakest performers, suggesting the model has difficulty with inference tasks.

3. **Backward transfer detected**: CoLA and MNLI show negative forgetting, meaning later tasks helped improve performance on these earlier tasks.

4. **Lambda scheduling**: The regularization strength follows a quadratic growth pattern within each task, with decay factors applied across tasks (starting at ~10 and decaying to ~2.3 by task 10).

## Verdict

The **2.44% average forgetting** is excellent for a 10-task continual learning scenario. The main weakness is NLI task performance (RTE/MNLI/QNLI), which drags down the average and WTA metric. The model successfully maintains sentiment analysis capabilities while showing positive backward transfer on some tasks.

## Raw Data Summary

### Final Accuracies

```json
{
  "SST2": 0.800,
  "IMDB": 1.000,
  "Yelp": 0.867,
  "CoLA": 0.707,
  "MRPC": 0.673,
  "QQP": 0.633,
  "RTE": 0.458,
  "QNLI": 0.523,
  "MNLI": 0.467,
  "AG_News": 0.860
}
```

### Per-Task Forgetting

```json
{
  "SST2": 0.043,
  "IMDB": 0.000,
  "Yelp": 0.030,
  "CoLA": -0.010,
  "MRPC": 0.047,
  "QQP": 0.080,
  "RTE": 0.043,
  "QNLI": 0.007,
  "MNLI": -0.020
}
```
