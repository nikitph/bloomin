import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

class GPT2FullProfiler:
    """
    Sweeps every neuron in GPT2-Small and generates a comprehensive metadata map.
    """
    def __init__(self, model_id="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.n_layers = self.model.config.n_layer
        self.d_model = self.model.config.n_embd
        self.mlp_ratio = 4
        self.d_ff = self.d_model * self.mlp_ratio
        
        # Primitives for Role Classification
        self.roles = ["Projector", "Rotator", "Scaler", "Inhibitor", "Composer"]
        
    def classify_algebraic_role(self, norm, sparsity):
        if norm > 1.8: return "Scaler (Amplifier)"
        if norm < 0.6: return "Inhibitor"
        if sparsity < 0.15: return "Projector (Feature Detector)"
        if sparsity > 0.4: return "Composer (Global Linker)"
        return "Rotator (Context Tracker)"

    def profile_all(self, output_path="dashboard/neuron_metadata.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        metadata = {}
        
        print(f"Starting Full Sweep of {self.n_layers} layers...")
        
        for layer_idx in range(self.n_layers):
            layer_data = {"attn": [], "mlp": []}
            
            # Access weights for this layer
            with torch.no_grad():
                # Attn output projection weights (resid_stream contribution)
                W_attn = self.model.transformer.h[layer_idx].attn.c_proj.weight # (768, 768)
                norm_attn = torch.norm(W_attn, dim=1)
                sparsity_attn = (torch.abs(W_attn) > 0.1).sum(dim=1).float() / self.d_model
                
                # MLP output projection weights (resid_stream contribution)
                W_mlp = self.model.transformer.h[layer_idx].mlp.c_proj.weight # (3072, 768)
                norm_mlp = torch.norm(W_mlp, dim=1)
                sparsity_mlp = (torch.abs(W_mlp) > 0.1).sum(dim=1).float() / self.d_model

            # Profile Attention Neurons (768)
            for n in range(self.d_model):
                n_norm = norm_attn[n].item()
                n_sparsity = sparsity_attn[n].item()
                role = self.classify_algebraic_role(n_norm, n_sparsity)
                
                layer_data["attn"].append({
                    "id": n,
                    "role": role,
                    "norm": round(n_norm, 3),
                    "sparsity": round(n_sparsity, 3),
                    "semantic": self.mock_semantic(layer_idx, "attn", n)
                })
                
            # Profile MLP Neurons (3072) - Sampling every 4th for the demo dashboard to save size if needed,
            # but we'll try for all and see.
            for n in range(self.d_ff):
                n_norm = norm_mlp[n].item()
                n_sparsity = sparsity_mlp[n].item()
                role = self.classify_algebraic_role(n_norm, n_sparsity)
                
                layer_data["mlp"].append({
                    "id": n,
                    "role": role,
                    "norm": round(n_norm, 3),
                    "sparsity": round(n_sparsity, 3),
                    "semantic": self.mock_semantic(layer_idx, "mlp", n)
                })
            
            metadata[f"layer_{layer_idx}"] = layer_data
            print(f"  ✓ Layer {layer_idx} Complete")

        with open(output_path, "w") as f:
            json.dump(metadata, f)
        print(f"Finished! Metadata saved to {output_path}")

    def mock_semantic(self, layer, type, idx):
        # Deterministic 'meaning' based on indices for the demo
        meanings = [
            "Detects definite articles", "Tracks plural nouns", "Inhibits noise in context",
            "Anticipates verb phrases", "Links subjects to objects", "Encodes spatial relations",
            "Triggers factual recall", "Filters for animal names", "Maintains dialogue state",
            "Detects punctuation cues", "Amplifies important semantic tokens", "Routes dependency info"
        ]
        return meanings[(layer + idx) % len(meanings)]

if __name__ == "__main__":
    profiler = GPT2FullProfiler()
    profiler.profile_all()
