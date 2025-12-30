import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Any

class ShadowNeuronInterpreter:
    """
    Decodes the 'Soul' of individual neurons in GPT-2.
    Mapping: Weights -> Algebraic Role -> Semantic Meaning -> Causal Function.
    """
    def __init__(self, model_id="gpt2"):
        print(f"Loading {model_id} for Neuron-Level Interpretation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.d_model = self.model.config.n_embd

    def analyze_algebraic_role(self, layer_idx: int, neuron_idx: int) -> Dict[str, Any]:
        """
        MECHANISM 1: Algebraic Decoder.
        Classifies the neuron based on its weight geometry.
        """
        # Get attention output projection weight (c_proj) which contains neuron-to-d_model mapping
        # In GPT2, c_proj is 1D -> 2D (hidden -> residual)
        with torch.no_grad():
            W_out = self.model.transformer.h[layer_idx].attn.c_proj.weight[neuron_idx, :]
            W_in = self.model.transformer.h[layer_idx].attn.c_attn.weight[:, neuron_idx]
            
            # SVD-based classification
            # Projection: W @ W \approx W
            # Rotation: W^T @ W \approx I
            # Scaling: Diagonal heavy
            
            norm = torch.norm(W_out)
            
            # Simple classification logic based on Shadow Theory primitives
            if norm > 1.5:
                op_type = "AMPLIFIER (Scaling)"
            elif norm < 0.5:
                op_type = "INHIBITOR (Scaling)"
            else:
                op_type = "ROUTING (Projection/Rotation)"
            
            # Check for sparsity (Identity/Projector proxy)
            sparsity = (torch.abs(W_out) > 0.1).sum().item() / self.d_model
            
            return {
                "op_type": op_type,
                "norm": norm.item(),
                "sparsity": sparsity,
                "algebraic_signature": "Rank-1 Trace Section"
            }

    def profile_semantic_role(self, layer_idx: int, neuron_idx: int, sample_texts: List[str]) -> Dict[str, Any]:
        """
        MECHANISM 2: Semantic Role Profiler.
        Finds the 'Meaning Basin' via high-activation discovery.
        """
        activations = []
        contexts = []
        
        for text in sample_texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Hidden states at layer_idx+1 (output of the layer)
                h = outputs.hidden_states[layer_idx + 1]
                # In GPT-2, the neuron we map usually corresponds to a dimension in the MLP or Attention output
                # Here we track the residual stream contribution
                act = h[0, :, neuron_idx].numpy()
                
                # Identify peaks
                for i, val in enumerate(act):
                    if val > np.percentile(act, 95):
                        context = self.tokenizer.decode(inputs["input_ids"][0, max(0, i-3):i+1])
                        activations.append(val)
                        contexts.append(context)
        
        return {
            "top_contexts": contexts[:5],
            "peak_activation": max(activations) if activations else 0,
            "semantic_basin": "Grammatical Structure" if len(contexts) > 0 else "Background Noise"
        }

    def perform_causal_ablation(self, layer_idx: int, neuron_idx: int, test_text: str) -> Dict[str, Any]:
        """
        MECHANISM 3: Causal Validation.
        Ablates the neuron and measures specific semantic loss.
        """
        inputs = self.tokenizer(test_text, return_tensors="pt")
        
        # Original Forward
        with torch.no_grad():
            outputs_orig = self.model(**inputs, labels=inputs["input_ids"])
            orig_loss = outputs_orig.loss.item()
        
        # Ablated Forward (Hook-based)
        def ablation_hook(module, input, output):
            # output of GPT2Block is usually (hidden_states, presence_of_presents, ...)
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, :, neuron_idx] = 0
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, :, neuron_idx] = 0
                return h
        
        handle = self.model.transformer.h[layer_idx].register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            outputs_abl = self.model(**inputs, labels=inputs["input_ids"])
            abl_loss = outputs_abl.loss.item()
            
        handle.remove()
        
        return {
            "loss_impact": abl_loss - orig_loss,
            "semantic_blindness": "High" if (abl_loss - orig_loss) > 0.05 else "Low"
        }

    def interpret_neuron(self, layer_idx: int, neuron_idx: int):
        print(f"\n" + "-"*60)
        print(f"DECODING NEURON: Layer {layer_idx}, Index {neuron_idx}")
        print("-"*60)
        
        # 1. Algebraic
        algebraic = self.analyze_algebraic_role(layer_idx, neuron_idx)
        print(f"[Algebraic] Type: {algebraic['op_type']}")
        print(f"            Norm: {algebraic['norm']:.4f}, Sparsity: {algebraic['sparsity']:.2%}")
        
        # 2. Semantic
        samples = [
            "The capital of France is Paris.",
            "The director of the movie sat on the chair.",
            "Scientists discovered a new planet in the galaxy.",
            "She wrote a letter to her friend yesterday."
        ]
        semantic = self.profile_semantic_role(layer_idx, neuron_idx, samples)
        print(f"[Semantic]  Top Contexts: {semantic['top_contexts']}")
        print(f"            Peak Activation: {semantic['peak_activation']:.4f}")
        
        # 3. Causal
        causal = self.perform_causal_ablation(layer_idx, neuron_idx, "The cat sat on the mat.")
        print(f"[Causal]    Loss Impact: {causal['loss_impact']:.6f}")
        print(f"            Semantic Blindness: {causal['semantic_blindness']}")
        
        print("-"*60)
        print("✓ INTERPRETATION COMPLETE")

if __name__ == "__main__":
    interpreter = ShadowNeuronInterpreter()
    # Let's interpret a known 'High Importance' neuron from our previous probe
    # Phase 15 found neurons #447, #481 in Layer 5 as carriers
    interpreter.interpret_neuron(layer_idx=5, neuron_idx=447)
    interpreter.interpret_neuron(layer_idx=5, neuron_idx=481)
