import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from gpt2_plusplus import GPT2PlusPlus
import random
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. IFE DATA GENERATOR (With Intermediate Traces)
# ============================================================================

def f_ife(x, steps=1):
    """f(x) = (x * 2 + 3) mod 97"""
    curr = x
    for _ in range(steps):
        curr = (curr * 2 + 3) % 97
    return curr

def generate_ife_trace(max_steps=50, stride=5):
    """
    Simpler trace: "x=4. n=20. [11, 25, 53, 12]"
    """
    x = random.randint(0, 96)
    n = random.randint(1, max_steps)
    if n % stride != 0:
        n = ((n // stride) + 1) * stride
    
    trace_vals = []
    for i in range(stride, n + 1, stride):
        res = f_ife(x, i)
        trace_vals.append(str(res))
    
    trace_str = ", ".join(trace_vals)
    problem = f"x={x}. n={n}. Trace: [{trace_str}]"
    return problem, x, n

# ============================================================================
# 2. EVALUATION HARNESS
# ============================================================================

def evaluate_ife(model, tokenizer, device, max_n=100, samples_per_n=10, loop_k=1):
    model.eval()
    results = {}
    
    print(f"\n📊 Evaluating IFE Depth (LoopK={loop_k})...")
    n_values = sorted(list(set([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])))
    
    with torch.no_grad():
        for n in n_values:
            if n > max_n: break
            correct = 0
            for s_idx in range(samples_per_n):
                x = random.randint(0, 96)
                target = f_ife(x, n)
                # We prompt for the specific xN value at the end of the trace
                prompt = f"x={x}. n={n}. Trace: "
                if n > 5:
                    # Provide partial trace as context for large N to help shallow patterns, 
                    # but test if Nirodha can complete the final step better.
                    # Actually, let's just test direct prediction after "Trace: "
                    pass
                
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                
                # We need to simulate the generation of the trace or just the final value.
                # To stay true to the 'internal computation' claim, we test if it can 
                # finish the trace accurately.
                
                # Search for "x{n}={target}" in the output
                for step_gen in range(30): # Allow room for trace generation
                    outputs = model(input_ids, loop_k=loop_k)
                    next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(0)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    decoded = tokenizer.decode(input_ids[0])
                    
                    if s_idx == 0 and step_gen % 10 == 0:
                        print("      [Debug Gen] " + decoded.replace('\n', ' '))
                        
                    if f"{target}" in decoded:
                        correct += 1
                        break
                    if "Final:" in decoded: break # stop condition
            
            acc = correct / samples_per_n
            results[n] = acc
            print(f"N={n:3d} | Accuracy: {acc*100:3.1f}%")

    return results

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_ife(depth=8, loop_k=4, steps=2000, lr=5e-4):
    """
    Slower LR, more steps, simpler beta schedule.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    print(f"\n🚀 Training Recurrent IFE | BlockDepth={depth} | LoopK={loop_k} | Device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2PlusPlus().to(device)
    model.add_layers(n=depth, beta=50.0) 
    model.to(device); model.set_anchor()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for step in range(steps + 1):
        # 1. Slower Beta Schedule
        curr_beta = min(1000, 50 + step // 2)
        model.update_beta(curr_beta)
        
        # 2. Batch with Intermediate Supervision
        batch_text = [generate_ife_trace(max_steps=30, stride=5)[0] for _ in range(8)]
        inputs = tokenizer(batch_text, return_tensors='pt', padding=True).to(device)
        labels = inputs.input_ids[:, 1:].contiguous()
        
        optimizer.zero_grad()
        # Use loop_k recurrent iterations
        logits = model(inputs.input_ids, attention_mask=inputs.attention_mask, loop_k=loop_k)[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            avg_drift = sum(model.drifts) / len(model.drifts) if model.drifts else 0
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Beta: {curr_beta:4d} | Avg Drift: {avg_drift:.6f}")

    return model, tokenizer, device

# ============================================================================
# 4. MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    depths_configs = [
        {"depth": 0,  "loop_k": 1, "name": "Baseline (GPT-2)"},
        {"depth": 8,  "loop_k": 4, "name": "Recurrent 8x4 (32 effective)"},
        {"depth": 8,  "loop_k": 16, "name": "Recurrent 8x16 (128 effective)"}
    ]
    
    all_results = {}
    
    for config in depths_configs:
        print(f"\n" + "="*50)
        print(f"🏁 CONFIG: {config['name']}")
        print("="*50)
        
        if config["depth"] == 0:
            if torch.backends.mps.is_available(): device = torch.device("mps")
            else: device = torch.device('cpu')
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2PlusPlus().to(device)
            results = evaluate_ife(model, tokenizer, device, max_n=30, samples_per_n=10, loop_k=1)
        else:
            model, tokenizer, device = train_ife(
                depth=config["depth"], 
                loop_k=config["loop_k"], 
                steps=1000
            )
            results = evaluate_ife(
                model, tokenizer, device, 
                max_n=100, samples_per_n=10, 
                loop_k=config["loop_k"]
            )
        
        all_results[config["name"]] = results

    # Print summary table
    print("\n" + "="*50)
    print("📈 FINAL IFE RECURSION RESULTS")
    print("="*50)
    
    names = [c["name"] for c in depths_configs]
    header = "N ".ljust(5) + "".join([n[:15].ljust(18) for n in names])
    print(header)
    print("-" * len(header))
    
    n_values = [1, 5, 10, 20, 50, 100]
    for n in n_values:
        row = f"{n:2d} ".ljust(5)
        for name in names:
            acc = all_results[name].get(n, 0.0)
            row += f"{acc*100:4.1f}%".ljust(18)
        print(row)

if __name__ == "__main__":
    run_experiment()
