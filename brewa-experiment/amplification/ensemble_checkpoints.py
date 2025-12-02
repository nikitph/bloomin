"""
Ensemble Checkpoint Test
=========================

Test if ensembling multiple checkpoints (epochs 25, 30, 35) can push recall higher.
Simple averaging of embeddings from different checkpoints.

Expected: 79.0-79.5% recall
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_20newsgroups import load_and_embed_data, split_categories, evaluate_recall
from amplification.amplified_encoders import AdversarialHybridREWAEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def ensemble_encode(models, embeddings):
    """Encode using ensemble of models"""
    encoded_list = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            encoded = model(embeddings, add_noise=False)
            encoded_list.append(encoded)
    
    # Average the encodings
    ensemble_encoded = torch.stack(encoded_list).mean(dim=0)
    
    # Normalize
    return F.normalize(ensemble_encoded, dim=-1)

def test_ensemble():
    print("="*70)
    print("Ensemble Checkpoint Test")
    print("="*70)
    
    # Load Data
    embeddings, labels, target_names = load_and_embed_data()
    
    # Split Categories
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    unseen_emb = unseen_emb.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)
    
    print(f"\nUnseen (Zero-shot): {len(unseen_emb)} samples")
    
    # Try to load checkpoints
    checkpoint_files = [
        'adversarial_best_25.pth',
        'adversarial_best_30.pth',
        'adversarial_best_35.pth',
    ]
    
    models = []
    loaded_epochs = []
    
    for ckpt_file in checkpoint_files:
        if os.path.exists(ckpt_file):
            print(f"Loading {ckpt_file}...")
            model = AdversarialHybridREWAEncoder(768, 256).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
            model.eval()
            models.append(model)
            
            # Extract epoch number
            epoch = ckpt_file.split('_')[-1].replace('.pth', '')
            loaded_epochs.append(epoch)
        else:
            print(f"⚠️  {ckpt_file} not found, skipping...")
    
    if len(models) == 0:
        print("\n❌ No checkpoints found! Cannot run ensemble test.")
        print("Expected files: adversarial_best_25.pth, adversarial_best_30.pth, adversarial_best_35.pth")
        return
    
    print(f"\n✓ Loaded {len(models)} checkpoints: epochs {', '.join(loaded_epochs)}")
    
    # Test individual checkpoints first
    print(f"\n" + "-"*70)
    print("Individual Checkpoint Performance:")
    print("-"*70)
    
    individual_recalls = []
    for i, (model, epoch) in enumerate(zip(models, loaded_epochs)):
        recall = evaluate_recall(model, unseen_emb, unseen_labels)
        individual_recalls.append(recall)
        print(f"Epoch {epoch}: {recall:.1%}")
    
    # Test ensemble
    print(f"\n" + "-"*70)
    print("Ensemble Performance:")
    print("-"*70)
    
    # Custom evaluation for ensemble
    batch_size = 1000
    all_encoded = []
    
    for i in range(0, len(unseen_emb), batch_size):
        batch = unseen_emb[i:i+batch_size]
        encoded_batch = ensemble_encode(models, batch)
        all_encoded.append(encoded_batch)
    
    ensemble_encoded = torch.cat(all_encoded, dim=0)
    
    # Compute recall
    k = 10
    correct = 0
    total = len(unseen_emb)
    
    for i in range(total):
        query = ensemble_encoded[i:i+1]
        query_label = unseen_labels[i].item()
        
        # Compute similarities
        sims = torch.matmul(query, ensemble_encoded.T).squeeze(0)
        
        # Get top-k
        top_k_indices = torch.topk(sims, k+1)[1][1:]  # Exclude self
        top_k_labels = unseen_labels[top_k_indices]
        
        # Check if query label in top-k
        if query_label in top_k_labels:
            correct += 1
    
    ensemble_recall = correct / total
    
    print(f"Ensemble ({len(models)} models): {ensemble_recall:.1%}")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best_individual = max(individual_recalls)
    improvement = (ensemble_recall - best_individual) * 100
    
    print(f"Best Individual:  {best_individual:.1%}")
    print(f"Ensemble:         {ensemble_recall:.1%}")
    print(f"Improvement:      {improvement:+.1f}%")
    
    if ensemble_recall >= 0.79:
        print(f"\n🎉 SUCCESS! Ensemble achieved {ensemble_recall:.1%} (≥79%)")
    elif ensemble_recall > best_individual:
        print(f"\n✅ IMPROVEMENT! Ensemble beat best individual by {improvement:.1f}%")
    else:
        print(f"\n⚠️  Ensemble did not improve over best individual")
    
    return ensemble_recall, best_individual

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    ensemble_recall, best_individual = test_ensemble()
    
    print(f"\n" + "="*70)
    print(f"🎯 FINAL: Ensemble = {ensemble_recall:.1%}, Best Single = {best_individual:.1%}")
    print("="*70)
