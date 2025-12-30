import torch
from safetensors.torch import load_file
import numpy as np

def analyze_stats(path="/Users/truckx/PycharmProjects/bloomin/mvc/mvc/model.safetensors"):
    if not os.path.exists(path):
        print(f"Path {path} not found.")
        return

    weights = load_file(path)
    
    layer_idx = 5
    attn_key = f"transformer.h.{layer_idx}.attn.c_proj.weight"
    mlp_key = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
    
    for key, name in [(attn_key, "Attention"), (mlp_key, "MLP")]:
        W = weights[key]
        norms = torch.norm(W, dim=1).numpy()
        sparsity = (torch.abs(W) > 0.1).sum(dim=1).float() / W.shape[1]
        sparsity = sparsity.numpy()
        
        print(f"\n--- {name} Layer {layer_idx} Stats ---")
        print(f"Norms: Mean={norms.mean():.3f}, Std={norms.std():.3f}, Min={norms.min():.3f}, Max={norms.max():.3f}")
        print(f"Sparsity (>0.1): Mean={sparsity.mean():.3f}, Std={sparsity.std():.3f}, Min={sparsity.min():.3f}, Max={sparsity.max():.3f}")
        
        # Count current classes
        scalers = (norms > 1.8).sum()
        inhibitors = (norms < 0.6).sum()
        projectors = ((norms >= 0.6) & (norms <= 1.8) & (sparsity < 0.15)).sum()
        composers = ((norms >= 0.6) & (norms <= 1.8) & (sparsity > 0.4)).sum()
        rotators = ((norms >= 0.6) & (norms <= 1.8) & (sparsity >= 0.15) & (sparsity <= 0.4)).sum()
        
        total = len(norms)
        print(f"Class Counts: Scaler={scalers}, Inhibitor={inhibitors}, Projector={projectors}, Rotator={rotators}, Composer={composers}")

import os
if __name__ == "__main__":
    analyze_stats()
