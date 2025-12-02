"""
Correct Ensemble Evaluation
============================

Evaluating ensemble performance with correct methodology:
1. Method 1: Retrieve from Seen Database (User's Request) - Expected 0% due to disjoint labels
2. Method 2: Ensemble Voting from Seen Database (User's Request) - Expected 0% due to disjoint labels
3. Method 3: Retrieve from Unseen Database (Leave-One-Out) - Expected ~79% (Validates clustering)
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

def correct_ensemble_eval():
    print("="*70)
    print("CORRECT Ensemble Evaluation")
    print("="*70)
    
    # Load ALL data
    embeddings, labels, target_names = load_and_embed_data()
    
    # Split categories
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    print(f"Seen (database): {len(seen_emb)} samples")
    print(f"Unseen (queries): {len(unseen_emb)} samples")
    
    # Check intersection
    seen_classes = torch.unique(seen_labels).cpu().numpy()
    unseen_classes = torch.unique(unseen_labels).cpu().numpy()
    intersection = np.intersect1d(seen_classes, unseen_classes)
    print(f"Class Intersection: {intersection}")
    if len(intersection) == 0:
        print("⚠️  WARNING: Seen and Unseen classes are DISJOINT.")
        print("    Retrieving from Seen database for Unseen queries will yield 0% recall.")
    
    # Move to device
    seen_emb = seen_emb.to(DEVICE)
    unseen_emb = unseen_emb.to(DEVICE)
    seen_labels = seen_labels.to(DEVICE)
    unseen_labels = unseen_labels.to(DEVICE)
    
    # Load models
    models = []
    checkpoint_files = ['adversarial_best_25.pth', 'adversarial_best_30.pth', 'adversarial_best_35.pth']
    
    for ckpt_file in checkpoint_files:
        if os.path.exists(ckpt_file):
            print(f"Loading {ckpt_file}...")
            model = AdversarialHybridREWAEncoder(768, 256).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
            model.eval()
            models.append(model)
    
    if not models:
        print("No models loaded!")
        return

    # Method 1: Average Recalls (Retrieving from SEEN)
    print(f"\n" + "-"*70)
    print("Method 1: Retrieve from SEEN Database (User's Request)")
    print("-"*70)
    
    for i, model in enumerate(models):
        recall = evaluate_recall(model, unseen_emb, unseen_labels, 
                                 database_emb=seen_emb, database_labels=seen_labels)
        print(f"Model {i+1}: {recall:.1%}")
    
    # Method 2: Ensemble Voting (Retrieving from SEEN)
    print(f"\n" + "-"*70)
    print("Method 2: Ensemble Voting from SEEN Database (User's Request)")
    print("-"*70)
    
    # Encode database with each model
    encoded_databases = []
    for model in models:
        with torch.no_grad():
            encoded_db = model(seen_emb, add_noise=False)
            encoded_db = F.normalize(encoded_db, dim=-1)
            encoded_databases.append(encoded_db)
    
    # Encode queries with each model
    encoded_queries = []
    for model in models:
        with torch.no_grad():
            encoded_q = model(unseen_emb, add_noise=False)
            encoded_q = F.normalize(encoded_q, dim=-1)
            encoded_queries.append(encoded_q)
            
    # Ensemble voting logic (from user)
    batch_size = 1000
    k = 10
    correct = 0
    total = len(unseen_emb)
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_indices = range(i, end)
        
        for query_idx in batch_indices:
            query_label = unseen_labels[query_idx].item()
            candidate_votes = {}
            
            for model_idx in range(len(models)):
                query_vec = encoded_queries[model_idx][query_idx:query_idx+1]
                database_vecs = encoded_databases[model_idx]
                
                sims = torch.matmul(query_vec, database_vecs.T).squeeze(0)
                top_indices = torch.topk(sims, 20)[1]
                
                for rank, db_idx in enumerate(top_indices):
                    candidate_votes[db_idx.item()] = candidate_votes.get(db_idx.item(), 0) + (20 - rank)
            
            top_candidates = sorted(candidate_votes.items(), key=lambda x: x[1], reverse=True)[:k]
            top_indices = [idx for idx, _ in top_candidates]
            
            top_labels = seen_labels[top_indices].cpu().numpy()
            if query_label in top_labels:
                correct += 1
        
        if i % 1000 == 0:
            print(f"Processed {i}/{total} queries...")

    ensemble_recall_seen = correct / total
    print(f"Ensemble Voting Recall (from SEEN): {ensemble_recall_seen:.1%}")

    # Method 3: Retrieve from UNSEEN Database (Leave-One-Out)
    print(f"\n" + "-"*70)
    print("Method 3: Retrieve from UNSEEN Database (Leave-One-Out)")
    print("This evaluates clustering quality within the unseen set.")
    print("-"*70)
    
    # Individual
    individual_recalls_unseen = []
    for i, model in enumerate(models):
        recall = evaluate_recall(model, unseen_emb, unseen_labels) # Default: database=unseen (exclude self)
        individual_recalls_unseen.append(recall)
        print(f"Model {i+1}: {recall:.1%}")
        
    # Ensemble Voting (from UNSEEN)
    correct = 0
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_indices = range(i, end)
        
        for query_idx in batch_indices:
            query_label = unseen_labels[query_idx].item()
            candidate_votes = {}
            
            for model_idx in range(len(models)):
                query_vec = encoded_queries[model_idx][query_idx:query_idx+1]
                database_vecs = encoded_queries[model_idx] # Use UNSEEN as database
                
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
            
            # Recall Calculation (Corrected from Hit Rate)
            n_relevant_retrieved = (top_labels == query_label).sum()
            n_relevant_total = (unseen_labels == query_label).sum().item() - 1 # Exclude self
            max_possible = min(k, n_relevant_total)
            
            if max_possible > 0:
                correct += n_relevant_retrieved / max_possible
                
    ensemble_recall_unseen = correct / total
    print(f"Ensemble Voting Recall (from UNSEEN): {ensemble_recall_unseen:.1%}")
    
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Retrieval from SEEN (User's Method):   {ensemble_recall_seen:.1%}")
    print(f"Retrieval from UNSEEN (Cluster Qual):  {ensemble_recall_unseen:.1%}")
    print(f"Best Single Model (Unseen):            {max(individual_recalls_unseen):.1%}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    correct_ensemble_eval()
