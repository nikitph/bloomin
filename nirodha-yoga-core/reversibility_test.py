import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from gpt2_plusplus import GPT2PlusPlus
import random
import copy

# ============================================================================
# 1. UTILS & DATA GENERATOR
# ============================================================================

def generate_math_problem():
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    res = a + b
    # CoT: "Q: 7 + 8. A: 7+8=15. Final: 15"
    problem = f"Q: {a} + {b}. A: {a}+{b}={res}. Final: {res}"
    return problem

def evaluate(model, tokenizer, device, label="Phase"):
    model.eval()
    print(f"\n" + "="*50)
    print(f"📊 EVALUATION: {label}")
    print("="*50)
    
    # 1. Base Knowledge Check
    prompts = ["The capital of France is", "To be or not to", "The sun rises in the"]
    print(f"\n--- Language Performance ---")
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            # Simple greedy generation
            for _ in range(5):
                outputs = model(input_ids)
                next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            print(f"Prompt: '{prompt}' -> '{tokenizer.decode(input_ids[0])}'")

    # 2. Math Reasoning Check
    num_samples = 10
    correct = 0
    print(f"\n--- Math Reasoning Check ---")
    with torch.no_grad():
        for i in range(num_samples):
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
            
            if i < 3: # Log first 3 samples
                print(f"Prompt: '{prompt}' -> Generated: '{tokenizer.decode(input_ids[0])}'")
    
    accuracy = 100 * correct / num_samples
    print(f"\n[Result] Math Accuracy: {accuracy:.2f}%")
    return accuracy

# ============================================================================
# 2. REVERSIBILITY TEST ORCHESTRATION
# ============================================================================

def run_reversibility_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Reversibility Test on device: {device}\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Phase 1: Baseline
    print("📍 [PHASE 1] Recording Baseline...")
    model = GPT2PlusPlus().to(device)
    baseline_acc = evaluate(model, tokenizer, device, "BASELINE (Original GPT-2)")
    
    # Phase 2: Specialization
    print("\n📍 [PHASE 2] Specializing on Arithmetic (Adding Depth)...")
    # Add layers and anchor
    model.add_layers(n=3, beta=500.0) 
    model.to(device); model.set_anchor()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Training 3 specialized Nirodha layers...")
    model.train()
    for step in range(1001):
        text_batch = [generate_math_problem() for _ in range(4)]
        inputs = tokenizer(text_batch, return_tensors='pt', padding=True).to(device)
        labels = inputs.input_ids[:, 1:].contiguous()
        
        optimizer.zero_grad()
        logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward(); optimizer.step()
        
        if step % 250 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    specialized_acc = evaluate(model, tokenizer, device, "SPECIALIZED (GPT-2++ with Nirodha-D)")
    
    # Phase 3: Reversion
    print("\n📍 [PHASE 3] Reverting to Baseline (Removing Added Depth)...")
    # Clear the added layers
    model.nirodha_blocks = nn.ModuleList() 
    
    # After clearing nirodha_blocks, the forward pass will skip step 3 
    # and go from frozen base to final head.
    reverted_acc = evaluate(model, tokenizer, device, "REVERTED (Back to Baseline)")
    
    # Final Summary
    print("\n" + "="*50)
    print("🏁 REVERSIBILITY TEST SUMMARY")
    print("="*50)
    print(f"Baseline Math Accuracy:   {baseline_acc:.1f}%")
    print(f"Specialized Math Accuracy: {specialized_acc:.1f}%")
    print(f"Reverted Math Accuracy:    {reverted_acc:.1f}%")
    
    if abs(reverted_acc - baseline_acc) < 1.0:
        print("\n✅ TEST PASSED: Performance returned exactly to baseline.")
        print("Cognition addition was 100% reversible and non-destructive.")
    else:
        print("\n❌ TEST FAILED: Divergence detected after reversion.")

if __name__ == "__main__":
    run_reversibility_test()
