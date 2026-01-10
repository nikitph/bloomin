"""
Experiment 2: Amnesiac Processor Test
Definitively prove observer necessity by disabling SSM memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rssa_v3 import RSSA, train_step
import os

class AmnesiacRSSA(RSSA):
    """RSSA with the ability to disable SSM memory at initialization"""
    def __init__(self, *args, use_ssm=True, **kwargs):
        self.use_ssm = use_ssm # Carry flag for easy identification
        super().__init__(*args, **kwargs)
        
        # Override layers with use_ssm flag
        from rssa_v3 import ProcessorLayer
        self.layers = nn.ModuleList([
            ProcessorLayer(
                kwargs.get('d_model', 256),
                n_heads=kwargs.get('n_heads', 8),
                window_size=kwargs.get('processor_window', 256),
                dropout=kwargs.get('dropout', 0.1),
                use_ssm=use_ssm
            ) for _ in range(kwargs.get('n_layers', 4))
        ])

def generate_strict_copy_task(batch_size, vocab_size, delay=64):
    """
    HEAVY NEEDLE: [Token x 32] [64 Pad] [Query] -> [Token x 32]
    Increasing signal mass to bypass chunk-mean dilution.
    """
    input_len = 32
    query_token = vocab_size - 2
    pad_token = vocab_size - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Random token (repeated to make heavy needle)
        token_val = torch.randint(1, vocab_size-2, (1,))
        val = token_val.repeat(input_len)
        
        # Input: [VAL] [PAD x delay] [QUERY] [PAD x input_len]
        in_seq = torch.cat([
            val,
            torch.full((delay,), pad_token),
            torch.tensor([query_token]),
            torch.full((input_len,), pad_token)
        ])
        
        # Target: [IGNORE x 32 + delay + 1] [VAL]
        target = torch.cat([
            torch.full((input_len + delay + 1,), -100),
            val
        ])
        
        inputs.append(in_seq)
        targets.append(target)
        
    return torch.stack(inputs), torch.stack(targets)

def run_test_case(name, use_ssm, ablate_observer, n_steps=1000):
    print(f"\n--- Running Test: {name} ---")
    print(f"Config: use_ssm={use_ssm}, ablate_observer={ablate_observer}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 10
    d_model = 256
    
    model = AmnesiacRSSA(
        vocab_size=vocab_size,
        d_model=d_model,
        d_observer=64,
        n_layers=4,
        processor_window=32, # window < delay
        observer_tau=0.95,
        use_ssm=use_ssm
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(n_steps):
        x, y = generate_strict_copy_task(8, vocab_size, delay=64)
        x, y = x.to(device), y.to(device)
        
        if ablate_observer:
            # Manual train step for ablation
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x, reset_observer=True, ablate_observer=True)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
            metrics = {'loss': loss.item()}
        else:
            metrics = train_step(model, x, y, optimizer)
            
        losses.append(metrics['loss'])
        
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            print(f"  Step {step+1:3d} | Loss: {avg_loss:.4f}")
            
    return sum(losses[-50:]) / 50

def main():
    print("Setting window=32, delay=64 (Attention cannot bridge gap)")
    
    # Case 1: Standard RSSA (SSM + Observer) - Expected: SUCCESS
    loss_rssa = run_test_case("Standard RSSA (SSM+Obs)", use_ssm=True, ablate_observer=False, n_steps=500)
    
    # Case 2: Amnesiac No-Observer (Identity) - Expected: STUCK AT LOSS 2.0+
    loss_fail = run_test_case("Amnesiac No-Obs (Identity)", use_ssm=False, ablate_observer=True, n_steps=500)
    
    # Case 3: Amnesiac RSSA (Identity + Observer) - Expected: NEAR ZERO
    loss_amnesiac_obs = run_test_case("Amnesiac RSSA (Identity+Obs)", use_ssm=False, ablate_observer=False, n_steps=1000)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Standard RSSA (SSM+Obs):     Loss {loss_rssa:.4f}")
    print(f"Amnesiac No-Obs (Identity):  Loss {loss_fail:.4f}")
    print(f"Amnesiac RSSA (Identity+Obs): Loss {loss_amnesiac_obs:.4f}")
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if loss_fail > 1.5 and loss_amnesiac_obs < 0.2:
        print("✅ SUCCESS: Observer necessity proven!")
        print("   The processor became amnesiac without SSM, but the Observer")
        print("   successfully bridged the gap alone.")
    else:
        print("❌ FAILED to prove necessity.")

if __name__ == "__main__":
    main()
