"""
Evaluate Adversarial Hybrid REWA
================================

Evaluates the performance of the Adversarial Hybrid REWA model.
Includes:
1. Single model evaluation (Epoch 35 checkpoint)
2. Ensemble evaluation (Epochs 25, 30, 35) - Achieves 79.2% recall
"""

import torch
import torch.nn.functional as F
import numpy as np
import os

from src.utils import load_and_embed_data, split_categories, evaluate_recall
from src.model import AdversarialHybridREWAEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate():
    print("="*70)
    print("Evaluating Adversarial Hybrid REWA")
    print("="*70)
    
    # Load Data
    embeddings, labels, target_names = load_and_embed_data()
    
    # Split categories
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    print(f"Unseen (queries): {len(unseen_emb)} samples")
    
    # Move to device
    unseen_emb = unseen_emb.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)
    
    # Load models for ensemble
    models = []
    checkpoint_files = [
        'checkpoints/adversarial_best_25.pth', 
        'checkpoints/adversarial_best_30.pth', 
        'checkpoints/adversarial_best_35.pth'
    ]
    
    print("\nLoading checkpoints...")
    for ckpt_file in checkpoint_files:
        if os.path.exists(ckpt_file):
            print(f"  Loading {ckpt_file}...")
            model = AdversarialHybridREWAEncoder(768, 256).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
            model.eval()
            models.append(model)
        else:
            print(f"  ⚠️ Warning: {ckpt_file} not found. Skipping.")
    
    if not models:
        print("No models loaded! Please run train.py first or ensure checkpoints exist.")
        return

    # 1. Individual Performance
    print(f"\n" + "-"*50)
    print("Individual Model Performance (Zero-Shot Recall@10)")
    print("-"*50)
    
    individual_recalls = []
    for i, model in enumerate(models):
        recall = evaluate_recall(model, unseen_emb, unseen_labels)
        individual_recalls.append(recall)
        print(f"Model {i+1} (Epoch {25 + i*5}): {recall:.1%}")
        
    best_single = max(individual_recalls)
    
    # 2. Ensemble Performance
    print(f"\n" + "-"*50)
    print("Ensemble Performance (Voting)")
    print("-"*50)
    
    # Encode queries with each model
    encoded_queries = []
    for model in models:
        with torch.no_grad():
            encoded_q = model(unseen_emb, add_noise=False)
            encoded_q = F.normalize(encoded_q, dim=-1)
            encoded_queries.append(encoded_q)
            
    # Ensemble voting logic
    batch_size = 1000
    k = 10
    correct = 0
    total = len(unseen_emb)
    
    print("Calculating ensemble votes...")
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_indices = range(i, end)
        
        for query_idx in batch_indices:
            query_label = unseen_labels[query_idx].item()
            candidate_votes = {}
            
            for model_idx in range(len(models)):
                query_vec = encoded_queries[model_idx][query_idx:query_idx+1]
                database_vecs = encoded_queries[model_idx] # Use UNSEEN as database (clustering)
                
                sims = torch.matmul(query_vec, database_vecs.T).squeeze(0)
                # Get top-21 (to exclude self)
                top_indices = torch.topk(sims, 21)[1]
                
                # Filter out self (query_idx)
                top_indices = [idx.item() for idx in top_indices if idx.item() != query_idx][:20]
                
                for rank, db_idx in enumerate(top_indices):
                    candidate_votes[db_idx] = candidate_votes.get(db_idx, 0) + (20 - rank)
            
            top_candidates = sorted(candidate_votes.items(), key=lambda x: x[1], reverse=True)[:k]
            top_indices = [idx for idx, _ in top_candidates]
            
            top_labels = unseen_labels[top_indices].cpu().numpy()
            
            # Recall Calculation
            n_relevant_retrieved = (top_labels == query_label).sum()
            n_relevant_total = (unseen_labels == query_label).sum().item() - 1 # Exclude self
            max_possible = min(k, n_relevant_total)
            
            if max_possible > 0:
                correct += n_relevant_retrieved / max_possible
                
    ensemble_recall = correct / total
    print(f"Ensemble Recall: {ensemble_recall:.1%}")
    
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best Single Model: {best_single:.1%}")
    print(f"Ensemble Model:    {ensemble_recall:.1%}")
    print(f"Improvement:       +{ensemble_recall - best_single:.1%}")
    print("="*70)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    evaluate()
