"""
Amplification Experiment
========================

Compare different amplification techniques for Hybrid REWA on 20 Newsgroups.
Techniques: Dynamic, MultiScale, Adversarial, Curriculum, Ensemble.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_rewa_encoder import HybridREWAEncoder, TripletLossTrainer
from experiment_20newsgroups import load_and_embed_data, split_categories, evaluate_recall
from amplification.amplified_encoders import (
    DynamicHybridREWAEncoder,
    MultiScaleHybridREWAEncoder,
    AdversarialHybridREWAEncoder,
    CurriculumHybridREWAEncoder,
    EnsembleHybridREWAEncoder
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def compare_amplification_methods():
    print("="*70)
    print("Amplification Methods Comparison")
    print("="*70)
    
    # 1. Load Data
    embeddings, labels, target_names = load_and_embed_data()
    
    # 2. Split Categories
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    # Split Seen into Train/Val
    n_seen = len(seen_emb)
    indices = torch.randperm(n_seen)
    train_idx = indices[:int(0.9 * n_seen)]
    val_idx = indices[int(0.9 * n_seen):]
    
    train_emb = seen_emb[train_idx].to(DEVICE)
    train_labels = seen_labels[train_idx].to(DEVICE)
    val_emb = seen_emb[val_idx].to(DEVICE)
    val_labels = seen_labels[val_idx].to(DEVICE)
    unseen_emb = unseen_emb.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)
    
    print(f"\nData Splits:")
    print(f"Train: {len(train_emb)}")
    print(f"Val:   {len(val_emb)}")
    print(f"Unseen (Zero-shot): {len(unseen_emb)}")
    
    # 3. Define Methods
    methods = {
        'Baseline': HybridREWAEncoder(768, 256, random_ratio=0.5),
        'Dynamic': DynamicHybridREWAEncoder(768, 256),
        'MultiScale': MultiScaleHybridREWAEncoder(768, 256),
        'Adversarial': AdversarialHybridREWAEncoder(768, 256),
        'Curriculum': CurriculumHybridREWAEncoder(768, 256),
        'Ensemble': EnsembleHybridREWAEncoder(768, 256, num_bases=8),
    }
    
    results = {}
    
    # 4. Train and Evaluate Each
    for name, model in methods.items():
        print(f"\n" + "-"*50)
        print(f"Testing {name}...")
        print("-"*50)
        
        model = model.to(DEVICE)
        trainer = TripletLossTrainer(model, margin=1.0, lr=1e-3)
        
        # Train
        for epoch in range(15): # 15 epochs for quick comparison
            # Handle special training requirements
            if name == 'Adversarial':
                # Custom training loop for adversarial?
                # For now, just standard triplet loss, maybe add adv loss if easy
                # The class has `adversarial_loss(x)`
                loss = trainer.train_epoch(train_emb, train_labels, num_triplets=1000)
                
                # Add adversarial step
                adv_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
                model.train()
                
                # Train discriminator and generator
                # Simple implementation: just run one batch of adv loss
                idx = torch.randperm(len(train_emb))[:BATCH_SIZE]
                batch = train_emb[idx]
                adv_loss = model.adversarial_loss(batch)
                
                adv_optimizer.zero_grad()
                adv_loss.backward()
                adv_optimizer.step()
                
            elif name == 'Curriculum':
                # Update epoch in model
                model.epoch = epoch
                loss = trainer.train_epoch(train_emb, train_labels, num_triplets=1000)
                
            elif name == 'Dynamic':
                # Pass epoch to forward? TripletLossTrainer calls model(x)
                # We need to patch TripletLossTrainer or just rely on default
                # The model.forward has default epoch=None.
                # Let's just train normally.
                loss = trainer.train_epoch(train_emb, train_labels, num_triplets=1000)
                
            else:
                loss = trainer.train_epoch(train_emb, train_labels, num_triplets=1000)
            
            if (epoch+1) % 5 == 0:
                val_recall = evaluate_recall(model, val_emb, val_labels)
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Recall={val_recall:.1%}")
        
        # Final Test
        print(f"Evaluating {name} on Unseen Categories...")
        unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
        results[name] = unseen_recall
        print(f"Result: {unseen_recall:.1%}")
    
    # 5. Summary
    print("\n" + "="*70)
    print("FINAL COMPARISON RESULTS")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Method':<20} {'Unseen Recall':<15} {'Improvement':<15}")
    print("-"*50)
    
    baseline = results['Baseline']
    
    for name, recall in sorted_results:
        imp = (recall - baseline) / baseline
        print(f"{name:<20} {recall:<15.1%} {imp:<15.1%}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    names = [x[0] for x in sorted_results]
    values = [x[1] for x in sorted_results]
    
    colors = ['blue' if n != 'Baseline' else 'red' for n in names]
    plt.bar(names, values, color=colors)
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.1%})')
    plt.ylim(0.5, 0.9)
    plt.title('Amplification Methods Comparison (Unseen Recall)')
    plt.ylabel('Recall@10')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('amplification_comparison.png')
    print("\nPlot saved to amplification_comparison.png")

if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # BATCH_SIZE global
    BATCH_SIZE = 64
    
    compare_amplification_methods()
