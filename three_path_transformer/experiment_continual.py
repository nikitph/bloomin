import torch
import numpy as np
from utils import set_seed, load_config
from model import ThreePathTransformer
from memory import ConceptMemory
from data_generation import generate_color_dataset
import copy

# Mock Shape Dataset Generator
def generate_shape_dataset(n_samples=1000, dim=128):
    from utils import random_peaked_distribution, geometric_mean
    concepts = {}
    concepts['Circle'] = random_peaked_distribution(dim, peak_loc=70)
    concepts['Square'] = random_peaked_distribution(dim, peak_loc=90)
    concepts['Triangle'] = random_peaked_distribution(dim, peak_loc=110)
    
    concepts['Cylinder'] = geometric_mean(concepts['Circle'], concepts['Square'])
    concepts['Cone'] = geometric_mean(concepts['Circle'], concepts['Triangle'])
    concepts['Pyramid'] = geometric_mean(concepts['Square'], concepts['Triangle'])
    
    examples = []
    mix_triples = [
        ('Circle', 'Square', 'Cylinder'),
        ('Circle', 'Triangle', 'Cone'),
        ('Square', 'Triangle', 'Pyramid')
    ]
    
    for _ in range(n_samples):
        c1, c2, tgt = mix_triples[np.random.choice(len(mix_triples))]
        examples.append({
            'input1': c1,
            'input2': c2,
            'target': concepts[tgt],
            'target_name': tgt
        })
    return examples, concepts

def evaluate_task(model, examples, vocab):
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for ex in examples:
            # Tokenize: [ID1, ID2]
            id1 = vocab[ex['input1']]
            id2 = vocab[ex['input2']]
            inputs = torch.tensor([[id1, id2]], device=device)
            
            # Encode (Mix via Attention)
            out = model.encode(inputs, path='fast')
            
            loss = torch.nn.functional.mse_loss(out, ex['target'].unsqueeze(0))
            total_loss += loss.item()
            
    return total_loss / len(examples)

def run_continual_learning():
    print("="*60)
    print("Experiment 3: Continual Learning (Colors -> Shapes)")
    print("="*60)
    
    set_seed(42)
    config = load_config()
    dim = config.get('embedding_dim', 128)
    
    # 1. Generate Data
    print("Generating Datasets...")
    # Pass max_depth=1 to get primaries+secondaries
    color_ex_raw, color_concepts, _, _ = generate_color_dataset(n_samples=200, dim=dim, max_depth=1)
    # Filter only 'mix' tasks for simplicity
    color_examples = [ex for ex in color_ex_raw if ex['type'] == 'mix']
    
    shape_examples, shape_concepts = generate_shape_dataset(n_samples=200, dim=dim)
    
    # Build Vocabulary
    all_names = list(color_concepts.keys()) + list(shape_concepts.keys())
    vocab = {name: i for i, name in enumerate(set(all_names))}
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size}")
    
    # Test Sets
    color_test = color_examples[:30]
    color_train = color_examples[30:]
    shape_test = shape_examples[:30]
    shape_train = shape_examples[30:]
    
    # 2. Initialize Models
    # Using 'encode' uses the Transformer weights (shared capacity)
    baseline = ThreePathTransformer(vocab_size=vocab_size, dim=dim)
    threepath = ThreePathTransformer(vocab_size=vocab_size, dim=dim)
    memory = ConceptMemory()
    
    opt_base = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    opt_tp = torch.optim.Adam(threepath.parameters(), lr=1e-3)
    
    # ==========================
    # PHASE 1: COLORS
    # ==========================
    print("\nPhase 1: Training on Colors (10 Epochs)...")
    for epoch in range(10):
        # Baseline
        loss_b = 0
        for ex in color_train:
            opt_base.zero_grad()
            ids = torch.tensor([[vocab[ex['input1']], vocab[ex['input2']]]])
            out = baseline.encode(ids)
            loss = torch.nn.functional.mse_loss(out, ex['target'].unsqueeze(0))
            loss.backward()
            opt_base.step()
            loss_b += loss.item()
            
        # Three-Path
        loss_tp = 0
        for ex in color_train:
            opt_tp.zero_grad()
            ids = torch.tensor([[vocab[ex['input1']], vocab[ex['input2']]]])
            out = threepath.encode(ids)
            loss = torch.nn.functional.mse_loss(out, ex['target'].unsqueeze(0))
            loss.backward()
            opt_tp.step()
            loss_tp += loss.item()
            
    print(f"  P1 Final Loss: Base={loss_b/len(color_train):.4f}, TP={loss_tp/len(color_train):.4f}")
    
    # Sleep / Store Memories
    print("  Sleeping: Consolidating Colors...")
    for name in color_concepts:
        # For experimental simplicity, we store the Ground Truth vector
        # In deployment, we would store self-generated vectors
        memory.store(name, color_concepts[name])
        
    # Evaluate P1
    acc_c_base = evaluate_task(baseline, color_test, vocab)
    acc_c_tp = evaluate_task(threepath, color_test, vocab)
    print(f"  Eval P1 (Color): Base={acc_c_base:.4f}, TP={acc_c_tp:.4f}")
    
    # ==========================
    # PHASE 2: SHAPES
    # ==========================
    print("\nPhase 2: Training on Shapes (10 Epochs)...")
    
    # Baseline: Naive training on Shape stream
    # Three-Path: Interleaved Replay
    
    for epoch in range(10):
        # Baseline
        for ex in shape_train:
            opt_base.zero_grad()
            ids = torch.tensor([[vocab[ex['input1']], vocab[ex['input2']]]])
            out = baseline.encode(ids)
            loss = torch.nn.functional.mse_loss(out, ex['target'].unsqueeze(0))
            loss.backward()
            opt_base.step()
            
        # Three-Path (Replay)
        # Mix 50/50 Shape and Replay
        batch_mix = []
        # Add shapes
        for ex in shape_train:
            batch_mix.append(('shape', ex))
        
        # Add replays (Color Maintenance)
        # Task: Recover Red from Red+Red (Identity) or just maintain embeddings?
        # Let's say we replay Mixing tasks for Colors if we remember them?
        # Or we replay "Concept Consistency": Input "Red" -> Output Red_Vec
        # Our model takes 2 tokens. We can do "Red Red" -> Red.
        replay_names = list(color_concepts.keys())
        for _ in range(len(shape_train)):
            nm = np.random.choice(replay_names)
            vec = memory.retrieve(nm)
            batch_mix.append(('replay', {'input1': nm, 'input2': nm, 'target': vec}))
            
        np.random.shuffle(batch_mix)
        
        loss_tp_shape = 0
        loss_tp_replay = 0
        
        for kind, ex in batch_mix:
            opt_tp.zero_grad()
            ids = torch.tensor([[vocab[ex['input1']], vocab[ex['input2']]]])
            out = threepath.encode(ids) # fast path training
            loss = torch.nn.functional.mse_loss(out, ex['target'].unsqueeze(0))
            loss.backward()
            opt_tp.step()
            
            if kind == 'shape': loss_tp_shape += loss.item()
            else: loss_tp_replay += loss.item()
            
    # ==========================
    # FINAL RESULTS
    # ==========================
    print("\nFinal Results:")
    
    final_c_base = evaluate_task(baseline, color_test, vocab)
    final_c_tp = evaluate_task(threepath, color_test, vocab)
    
    final_s_base = evaluate_task(baseline, shape_test, vocab)
    final_s_tp = evaluate_task(threepath, shape_test, vocab)
    
    print(f"Color Retention (Lower is better):")
    print(f"  Baseline:   {final_c_base:.4f} (drifted from {acc_c_base:.4f})")
    print(f"  Three-Path: {final_c_tp:.4f} (retained from {acc_c_tp:.4f})")
    
    ratio = final_c_base / (final_c_tp + 1e-9)
    print(f"  Forgetting Factor: Baseline error is {ratio:.1f}x higher")
    
    with open("CONTINUAL_RESULTS.md", "w") as f:
        f.write("# Continual Learning Results\n\n")
        f.write("| Task | Baseline Error | Three-Path Error |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Colors (Old)** | {final_c_base:.4f} | **{final_c_tp:.4f}** |\n")
        f.write(f"| **Shapes (New)** | {final_s_base:.4f} | {final_s_tp:.4f} |\n")
        f.write(f"\n**Impact**: Sleep Replay reduced forgetting by **{ratio:.1f}x**.\n")

if __name__ == "__main__":
    run_continual_learning()

