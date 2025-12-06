# REWA Text Diffusion Model

A minimal implementation of REWA Semantic Diffusion for Text Generation.

## Concept

Diffuse in continuous witness space → reconstruct witness structure → decode into text.

This bypasses discrete token issues, autoregressive error accumulation, and long-range inconsistency.

## Quick Start

### Training
```bash
python3 train.py
```

Trains for 100 epochs on dummy data. Model saved to `rewa_model.pt`.

### Generation

**With prompt:**
```bash
python3 sample.py "hello world"
```

**Without prompt (unconditional):**
```bash
python3 sample.py
```

## Architecture

- **WitnessEncoder**: Text → REWA witness vectors (via similarity overlap)
- **Diffusion**: Forward noise addition + Reverse denoising (DDPM-style)
- **ReverseDiffusionModel**: Predicts noise at each timestep
- **WitnessDecoder**: Witness vectors → Text (via Transformer decoder)

## Files

- `model.py` - Neural network modules
- `diffusion.py` - Diffusion processes
- `utils.py` - Tokenizer and dataset utilities
- `train.py` - Training script
- `sample.py` - Inference script

## Requirements

- PyTorch >= 2.0
- Python >= 3.8

## Example Output

```
Input: "hello"
Output: "helpo world"

Input: (none)
Output: "ginerative ai is the future"
```

## Next Steps

- Scale to larger datasets (WikiText, C4)
- Use BPE tokenization
- Implement Transformer-UNet
- Add classifier-free guidance
- Incorporate advanced REWA features (MI estimators, rank embeddings)
