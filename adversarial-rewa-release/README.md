# Adversarial Hybrid REWA

**State-of-the-Art Compressed Semantic Retrieval**

This repository contains the official implementation of the **Adversarial Hybrid REWA (Random-Embedded Witness Autoencoder)**, which achieves **79.2% Zero-Shot Recall@10** on the 20 Newsgroups benchmark (5 unseen categories), outperforming standard learned baselines while maintaining a 3x compression ratio.

## Key Features

*   **Hybrid Architecture**: Combines frozen random projections (for generalization) with learned projections (for performance).
*   **Adversarial Training**: Uses a discriminator to force learned features to maintain the statistical properties of random projections, preventing overfitting.
*   **Advanced Training Recipe**: Includes smooth adversarial loss, mixup augmentation, and adaptive weighting.
*   **High Performance**: 
    *   **78.6%** Recall@10 with a single model (3x compression).
    *   **79.2%** Recall@10 with a 3-model ensemble.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
To train the model from scratch (approx. 50 epochs):

```bash
python train.py
```
This will save checkpoints to `checkpoints/`.

### 2. Evaluate
To reproduce the **79.2%** ensemble result using the provided checkpoints:

```bash
python evaluate.py
```

## Results

| Model | Recall@10 (Zero-Shot) | Compression |
|-------|----------------------|-------------|
| Random Projection (Baseline) | ~27% | 3x |
| Learned REWA (Standard) | ~55% | 3x |
| Hybrid REWA (Baseline) | 73.7% | 3x |
| **Adversarial Hybrid REWA** | **78.6%** | **3x** |
| **Ensemble (3 Models)** | **79.2%** | **1x** |

## Directory Structure

*   `src/`: Core source code
    *   `model.py`: The `AdversarialHybridREWAEncoder` architecture.
    *   `base.py`: The base `HybridREWAEncoder` class.
    *   `utils.py`: Data loading and evaluation utilities.
*   `checkpoints/`: Saved model weights.
*   `train.py`: Training script.
*   `evaluate.py`: Evaluation script.
