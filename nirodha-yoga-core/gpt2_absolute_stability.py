import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from gpt2_plusplus import GPT2PlusPlus
import random

def generate_math_problem():
    a, b = random.randint(1, 9), random.randint(1, 9)
    return f"Q: {a} + {b}. A: {a}+{b}={a+b}. Final: {a+b}"

def eval_base(model, tokenizer, device, label):
    prompt = "The capital of France is"
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        for _ in range(5):
            outputs = model(input_ids)
            next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        res = tokenizer.decode(input_ids[0])
        print(f"[{label}] Prompt: '{prompt}' -> Result: '{res}'")
    return res

def run_stability_check():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test three configurations
    betas = [0, 1000, 1000000]
    labels = ["Standard FT (beta=0)", "Nirodha-D (beta=1e3)", "Absolute (beta=1e6)"]
    
    models = []
    optimizers = []
    
    print("--- Initializing Models ---")
    for beta in betas:
        m = GPT2PlusPlus().to(device)
        m.add_layers(n=2, beta=beta) # 2 layers for speed
        m.to(device); m.set_anchor()
        models.append(m)
        optimizers.append(torch.optim.AdamW(m.parameters(), lr=1e-3))
    
    # Baseline for all
    eval_base(models[0], tokenizer, device, "Original GPT-2")
    
    print("\n--- Training Aggressively on Math (50 steps) ---")
    for step in range(50):
        text_batch = [generate_math_problem() for _ in range(2)]
        inputs = tokenizer(text_batch, return_tensors='pt', padding=True).to(device)
        labels_ids = inputs.input_ids[:, 1:].contiguous()
        
        for m, opt in zip(models, optimizers):
            m.train(); opt.zero_grad()
            logits = m(inputs.input_ids, attention_mask=inputs.attention_mask)[:, :-1, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels_ids.reshape(-1))
            loss.backward(); opt.step()
            
    print("\n--- Final Stability Results ---")
    for m, l in zip(models, labels):
        eval_base(m, tokenizer, device, l)

if __name__ == "__main__":
    run_stability_check()
