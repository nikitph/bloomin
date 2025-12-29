import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import (get_embeddings, calculate_semantic_variance, generate_equivalent_prompts, 
                   factual_match, contradiction_detected, unstable_under_reask)
import argparse
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def measure_execution_gap(model, tokenizer, prompt, K=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    entropies = []
    with torch.no_grad():
        for i in range(K):
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            entropies.append(entropy)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            # Update inputs dictionary for next step
            new_input_ids = torch.cat([inputs['input_ids'], next_token], dim=1)
            new_attention_mask = torch.cat([inputs['attention_mask'], torch.ones((1, 1), device=device)], dim=1)
            inputs = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
    if len(entropies) < 2: return 0.1
    # Delta_O = rate of internal confidence locking (faster in earlier tokens)
    delta_o = (entropies[0] - entropies[-1]) / (len(entropies) - 1)
    return max(delta_o, 1e-3)

def measure_semantic_gap(model, tokenizer, prompt):
    variants = generate_equivalent_prompts(prompt)
    answers = []
    for variant in variants:
        inputs = tokenizer(variant, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=15, num_return_sequences=1, do_sample=True, temperature=1.2, pad_token_id=tokenizer.eos_token_id)
            answer_text = tokenizer.decode(output[0], skip_special_tokens=True)
            gen_part = answer_text[len(variant):].strip()
            if gen_part: answers.append(gen_part)
            else: answers.append("placeholder")
    if not answers: return 1e-3
    embeddings = get_embeddings(answers, model)
    variance = calculate_semantic_variance(embeddings)
    delta_w = 1.0 / (variance + 1e-3)
    return delta_w

def hallucination_metric(delta_o, delta_w):
    return ((delta_o - delta_w)**2) / (delta_o * delta_w + 1e-9)

def baseline_inference(model, tokenizer, prompt, temp=0.7, max_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temp, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

def throttle_controller(model, tokenizer, prompt, h_threshold=1.0):
    delta_o = measure_execution_gap(model, tokenizer, prompt)
    delta_w = measure_semantic_gap(model, tokenizer, prompt)
    h = hallucination_metric(delta_o, delta_w)
    temp, max_tokens = 0.7, 20
    if h > h_threshold and delta_o > delta_w:
        temp, max_tokens = 0.2, 10
    ans = baseline_inference(model, tokenizer, prompt, temp=temp, max_tokens=max_tokens)
    return ans, h, delta_o, delta_w

def witness_lift_controller(model, tokenizer, prompt, h_threshold=1.0):
    delta_o = measure_execution_gap(model, tokenizer, prompt)
    delta_w = measure_semantic_gap(model, tokenizer, prompt)
    h = hallucination_metric(delta_o, delta_w)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if h > h_threshold:
        candidates = []
        for _ in range(3):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
                candidates.append(tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip())
        candidates.sort(key=len)
        return candidates[1], h, delta_o, delta_w
    return baseline_inference(model, tokenizer, prompt), h, delta_o, delta_w

def combined_controller(model, tokenizer, prompt, h_threshold=1.0):
    delta_o = measure_execution_gap(model, tokenizer, prompt)
    delta_w = measure_semantic_gap(model, tokenizer, prompt)
    h = hallucination_metric(delta_o, delta_w)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if h > h_threshold:
        # Combined: Throttled sampling + Witness-Lift
        candidates = []
        for _ in range(3):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.2, pad_token_id=tokenizer.eos_token_id)
                candidates.append(tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip())
        candidates.sort(key=len)
        return candidates[1], h, delta_o, delta_w
    return baseline_inference(model, tokenizer, prompt), h, delta_o, delta_w

def is_hallucination(answer, reference):
    if not answer: return True
    return not factual_match(answer, reference)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    args = parser.parse_args()

    print(f"Loading model {args.model_name} on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    model.eval()

    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    results = {"baseline": [], "throttle": [], "witness_lift": [], "combined": []}
    all_metrics = []

    for i, item in enumerate(dataset[:args.num_samples]):
        prompt, reference = item["prompt"], item["reference"]
        print(f"\n[{i+1}/{args.num_samples}] Prompt: {prompt}")

        # Baseline & Metrics
        ans_base = baseline_inference(model, tokenizer, prompt)
        is_h_base = is_hallucination(ans_base, reference)
        results["baseline"].append(is_h_base)
        
        dO = measure_execution_gap(model, tokenizer, prompt)
        dW = measure_semantic_gap(model, tokenizer, prompt)
        H = hallucination_metric(dO, dW)
        ratio = dO / (dW + 1e-9)
        all_metrics.append({"H": H, "ratio": ratio, "dO": dO, "dW": dW, "is_h": is_h_base})
        print(f"  [Sample {i+1}] Prompt: {prompt} | Ref: {reference}")
        print(f"  [Metrics] dO (Execution): {dO:.4f}, dW (Semantic): {dW:.4f}, H (Impedance): {H:.4f}, Ratio: {ratio:.4f}")
        print(f"  [Answer] {ans_base}")
        print(f"  [Result] Base: {'HALLUCINATION' if is_h_base else 'PASS'}")

        # Throttle
        ans_thr, _, _, _ = throttle_controller(model, tokenizer, prompt)
        results["throttle"].append(is_hallucination(ans_thr, reference))
        
        # Witness Lift
        ans_wit, _, _, _ = witness_lift_controller(model, tokenizer, prompt)
        results["witness_lift"].append(is_hallucination(ans_wit, reference))

        # Combined
        ans_comb, _, _, _ = combined_controller(model, tokenizer, prompt)
        results["combined"].append(is_hallucination(ans_comb, reference))

        print(f"  H={H:.4f} | Base: {'H' if is_h_base else 'P'} | Thr: {'H' if results['throttle'][-1] else 'P'} | Wit: {'H' if results['witness_lift'][-1] else 'P'} | Comb: {'H' if results['combined'][-1] else 'P'}")

    print("\n" + "="*30 + "\nFINAL RESULTS")
    for key, val in results.items():
        print(f"{key.capitalize()}: {sum(val)/len(val):.2%} Hallucination Rate")

    os.makedirs("plots", exist_ok=True)
    suffix = f"_{args.model_name.replace('/', '_')}"
    
    # Stratified Plotting with wider bins
    bins = [(0, 10), (10, 100), (100, float('inf'))]
    bin_labels = ["H < 10", "10 < H < 100", "H > 100"]
    binned_h_rates = []
    for lo, hi in bins:
        bin_samples = [m["is_h"] for m in all_metrics if lo <= m["H"] < hi]
        binned_h_rates.append(sum(bin_samples)/len(bin_samples) if bin_samples else 0)

    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, binned_h_rates, color='orange')
    plt.title(f"Hallucination Rate Stratified by H ({args.model_name})")
    plt.ylabel("Baseline Hallucination Rate")
    plt.grid(axis='y', linestyle='--')
    plt.savefig(f"plots/stratified_h{suffix}.png")

    # System Comparison
    plt.figure(figsize=(10, 6))
    systems = ['Baseline', 'Throttle', 'Witness-Lift', 'Combined']
    rates = [sum(results[s.lower().replace('-', '_')]) / len(results[s.lower().replace('-', '_')]) for s in systems]
    plt.bar(systems, rates, color=['gray', 'blue', 'green', 'purple'])
    plt.title(f"System Comparison ({args.model_name})")
    plt.ylabel("Hallucination Rate")
    plt.savefig(f"plots/system_comparison{suffix}.png")

    print(f"\nPlots saved to 'plots/' with suffix '{suffix}'.")

if __name__ == "__main__":
    main()
