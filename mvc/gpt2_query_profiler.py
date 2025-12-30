import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class GPT2QueryProfiler:
    """
    Generates dynamic activation maps for a set of archetypal queries.
    """
    def __init__(self, model_id="gpt2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()
        
        # Archetypal Queries
        self.queries = {
            "Factual Recall": "Paris is the capital of France.",
            "Logic & Math": "If all men are mortal and Socrates is a man, then Socrates is mortal.",
            "Python Coding": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "Grammar & Syntax": "The quick brown foxes are jumping over the lazy dog.",
            "Emotional Narrative": "She walked through the rain, feeling a sense of profound sadness and longing."
        }

    def get_activations(self, text):
        query_map = {} # Key: layer_X_type, Val: List of activations
        
        def hook_fn(layer_idx, n_type):
            def hook(module, input, output):
                # Using the last token's activations as the representative state for the query
                # shape: (batch, seq, neurons)
                activations = output[0, -1, :].cpu().numpy().tolist()
                query_map[f"layer_{layer_idx}_{n_type}"] = activations
            return hook

        handles = []
        for i in range(self.model.config.n_layer):
            handles.append(self.model.transformer.h[i].mlp.c_proj.register_forward_hook(hook_fn(i, "mlp")))
            handles.append(self.model.transformer.h[i].attn.c_proj.register_forward_hook(hook_fn(i, "attn")))

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            self.model(**inputs)

        for h in handles:
            h.remove()
            
        return query_map

    def profile_all(self, output_path="dashboard/live_activations.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        all_query_data = {}
        
        print("Profiling Archetypal Queries...")
        for name, text in tqdm(self.queries.items()):
            all_query_data[name] = self.get_activations(text)
            
        with open(output_path, "w") as f:
            json.dump(all_query_data, f)
            
        print(f"✓ Live activations exported to {output_path}")

if __name__ == "__main__":
    profiler = GPT2QueryProfiler()
    profiler.profile_all(output_path="/Users/truckx/PycharmProjects/bloomin/mvc/dashboard/live_activations.json")
