import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import os

def export_to_safetensors(model_id="gpt2", output_path="mvc/model.safetensors"):
    print(f"Loading {model_id} for export...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = model.state_dict()
    
    # We only need the weights for the decoder layers for profiling
    # But for simplicity, we export the whole state dict
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Handle shared tensors: lm_head and wte share memory in GPT-2
    clean_state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    if "lm_head.weight" in clean_state_dict and "transformer.wte.weight" in clean_state_dict:
        if clean_state_dict["lm_head.weight"].data_ptr() == clean_state_dict["transformer.wte.weight"].data_ptr():
            print("Shared memory detected between lm_head.weight and transformer.wte.weight. Handling...")
            # We'll keep both but ensure they are separate copies for safetensors if needed, 
            # though it's better to just save one if they are identical, 
            # but for our profiler we want the full dictionary.
            # safetensors doesn't like shared pointers.
            clean_state_dict["lm_head.weight"] = clean_state_dict["lm_head.weight"].clone()
    
    save_file(clean_state_dict, output_path)
    print("Export complete.")

if __name__ == "__main__":
    export_to_safetensors()
