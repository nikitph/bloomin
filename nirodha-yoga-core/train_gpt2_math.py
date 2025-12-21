import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from gpt2_plusplus import GPT2PlusPlus
import random

# ============================================================================
# 1. SYNTHETIC MATH DATA GENERATOR (Chain-of-Thought Addition)
# ============================================================================

def generate_math_problem():
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    res = a + b
    # CoT: "Q: 7 + 8. A: 7+8=15. Final: 15"
    problem = f"Q: {a} + {b}. A: {a}+{b}={res}. Final: {res}"
    return problem

# ============================================================================
# 2. EVALUATION UTILS
# ============================================================================

def evaluate_base_knowledge(model, tokenizer, device, label="Target"):
    """Checks if the model still knows basic facts."""
    prompts = ["The capital of France is", "To be or not to", "The sun rises in the"]
    model.eval()
    print(f"\n--- Base Knowledge Check ({label}) ---")
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            # Simple greedy generation
            for _ in range(5):
                outputs = model(input_ids)
                next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            print(f"Prompt: '{prompt}' -> '{tokenizer.decode(input_ids[0])}'")

def evaluate_math(model, tokenizer, device, num_samples=10, label="Target"):
    model.eval()
    correct = 0
    print(f"\n--- Math Reasoning Check ({label}) ---")
    with torch.no_grad():
        for _ in range(num_samples):
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            prompt = f"Q: {a} + {b}. A:"
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            target_str = str(a + b)
            
            # Generate until 'Final:' or max 20 tokens
            for _ in range(15):
                outputs = model(input_ids)
                next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                decoded = tokenizer.decode(input_ids[0])
                if "Final:" in decoded and target_str in decoded:
                    correct += 1
                    break
                if len(input_ids[0]) > 20: break
            
            print(f"Prompt: '{prompt}' -> Generated: '{tokenizer.decode(input_ids[0])}'")
    print(f"Math Accuracy: {100 * correct / num_samples:.2f}%")

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_specialization():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Model
    print("[1/4] Initializing Nirodha-D Model...")
    model = GPT2PlusPlus().to(device)
    
    # 2. Benchmarking Base
    print("\n[Baseline Evaluation]")
    evaluate_base_knowledge(model, tokenizer, device, "Original")
    
    # 3. Expansion
    print("\n[2/4] Expanding Depth (Balanced Beta=500)...")
    model.add_layers(n=3, beta=500.0) 
    model.to(device); model.set_anchor()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    print("\n[3/4] Training on Single-Digit Addition (2000 Steps)...")
    model.train()
    for step in range(2001):
        text_batch = [generate_math_problem() for _ in range(4)]
        inputs = tokenizer(text_batch, return_tensors='pt', padding=True).to(device)
        labels = inputs.input_ids[:, 1:].contiguous()
        
        optimizer.zero_grad()
        logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward(); optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            if step == 1000:
                evaluate_math(model, tokenizer, device, num_samples=3, label="Mid-Train")
            
    # 4. Final Comparison
    print("\n[4/4] Final Results Comparison")
    evaluate_math(model, tokenizer, device, num_samples=10, label="Nirodha-D (Final)")
    evaluate_base_knowledge(model, tokenizer, device, label="Nirodha-D (Final)")

if __name__ == "__main__":
    train_specialization()
