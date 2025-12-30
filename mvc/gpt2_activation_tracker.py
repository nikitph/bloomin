import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

class GPT2ActivationTracker:
    def __init__(self, model_id="gpt2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()
        
        self.n_layers = self.model.config.n_layer
        self.d_model = self.model.config.n_embd
        self.d_ff = self.d_model * 4
        
        # Storage for top activations per neuron
        # We store (max_score, context_text)
        self.top_contexts = {} # Key: "layer_X_type_idx", Val: (score, text)

    def get_hook(self, layer_idx, n_type):
        def hook(module, input, output):
            # output shape for MLP: (batch, seq, 3072)
            # output shape for Attn: (batch, seq, 768)
            
            # We assume batch size 1 for simplicity in tracking contexts
            activations = output[0] # (seq, neurons)
            
            for n_idx in range(activations.shape[-1]):
                max_val, max_pos = torch.max(activations[:, n_idx], dim=0)
                max_val = max_val.item()
                
                key = f"layer_{layer_idx}_{n_type}_{n_idx}"
                if key not in self.top_contexts or max_val > self.top_contexts[key][0]:
                    # Extract context (window of tokens)
                    start = max(0, max_pos.item() - 5)
                    end = min(activations.shape[0], max_pos.item() + 5)
                    context_ids = self.current_input_ids[0, start:end]
                    context_text = self.tokenizer.decode(context_ids)
                    self.top_contexts[key] = (max_val, context_text)
                    
        return hook

    def run_corpus(self, corpus_texts):
        handles = []
        for i in range(self.n_layers):
            # Track MLP activations (c_fc output is better for neuron features than c_proj output)
            # Actually, for the dashboard we mapped c_proj, so let's stick to c_proj output for consistency
            handles.append(self.model.transformer.h[i].mlp.c_proj.register_forward_hook(self.get_hook(i, "mlp")))
            handles.append(self.model.transformer.h[i].attn.c_proj.register_forward_hook(self.get_hook(i, "attn")))

        print(f"Tracking activations across {len(corpus_texts)} snippets...")
        for text in tqdm(corpus_texts):
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                self.current_input_ids = inputs["input_ids"]
                self.model(**inputs)

        for h in handles:
            h.remove()

    def update_metadata(self, input_path="dashboard/neuron_metadata.json", output_path="dashboard/neuron_metadata.json"):
        with open(input_path, "r") as f:
            metadata = json.load(f)

        print("Updating metadata with real semantic contexts...")
        for layer_key, layer_data in metadata.items():
            l_idx = int(layer_key.split('_')[1])
            
            for n_type in ["attn", "mlp"]:
                for n_meta in layer_data[n_type]:
                    key = f"layer_{l_idx}_{n_type}_{n_meta['id']}"
                    if key in self.top_contexts:
                        score, context = self.top_contexts[key]
                        # Trim context and clean up
                        clean_context = context.strip().replace("\n", " ")
                        if len(clean_context) > 100:
                            clean_context = clean_context[:97] + "..."
                        
                        n_meta["semantic"] = f"Top Context: \"{clean_context}\" (Act: {score:.2f})"
                    else:
                        n_meta["semantic"] = "No significant activation in corpus."

        with open(output_path, "w") as f:
            json.dump(metadata, f)
        print(f"✓ Metadata synchronized at {output_path}")

if __name__ == "__main__":
    # Sample corpus with diverse semantic basins
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists have discovered a new species of deep-sea jellyfish in the Mariana Trench.",
        "Paris is the capital of France, known for its iconic Eiffel Tower and world-class cuisine.",
        "The Python programming language is widely used for artificial intelligence and data science.",
        "In a surprising turn of events, the company announced a major merger with its primary competitor.",
        "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms.",
        "She decided to travel to the mountains for a peaceful retreat away from the city noise.",
        "The 19th amendment to the US constitution granted women the right to vote in 1920.",
        "A healthy diet consists of fruits, vegetables, whole grains, and lean proteins.",
        "The movie director praised the lead actor for his emotional performance in the final scene."
    ]
    
    tracker = GPT2ActivationTracker()
    tracker.run_corpus(corpus)
    tracker.update_metadata(input_path="/Users/truckx/PycharmProjects/bloomin/mvc/dashboard/neuron_metadata.json", 
                            output_path="/Users/truckx/PycharmProjects/bloomin/mvc/dashboard/neuron_metadata.json")
