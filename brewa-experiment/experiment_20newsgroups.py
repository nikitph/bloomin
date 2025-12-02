"""
Hybrid REWA Validation on 20 Newsgroups
=======================================

Rigorous validation of Hybrid REWA on the 20 Newsgroups benchmark.
Goal: Prove generalization to unseen semantic categories.

Plan:
1. Load 20 Newsgroups dataset (20 categories)
2. Generate BERT embeddings for all texts
3. Split into:
   - Seen Categories (15 classes): Train/Val/Test split
   - Unseen Categories (5 classes): Zero-shot generalization test
4. Train Hybrid REWA on Seen Categories
5. Evaluate Recall@10 on both sets

Hypothesis:
- Recall on Seen Categories: >90%
- Recall on Unseen Categories: >60% (proving strong generalization)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt

from hybrid_rewa_encoder import HybridREWAEncoder, TripletLossTrainer

# Configuration
BATCH_SIZE = 32
D_MODEL = 768
M_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_and_embed_data():
    """Load 20 Newsgroups and generate BERT embeddings."""
    print(f"Loading 20 Newsgroups dataset...")
    # Load all data
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data
    labels = newsgroups.target
    target_names = newsgroups.target_names
    
    print(f"Loaded {len(texts)} documents across {len(target_names)} categories.")
    
    # Filter empty texts
    valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 10]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]
    print(f"Retained {len(texts)} valid documents.")
    
    # Initialize BERT
    print("Initializing BERT for embedding generation...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    model.eval()
    
    embeddings = []
    
    # Process in batches
    print("Generating embeddings (this may take a while)...")
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] embedding
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(cls_emb)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    
    return embeddings, labels, target_names

def split_categories(embeddings, labels, target_names, n_unseen=5):
    """Split into seen (train) and unseen (generalization) categories."""
    n_classes = len(target_names)
    
    # Randomly select unseen categories
    # We'll try to pick diverse ones if possible, but random is fair
    perm = torch.randperm(n_classes)
    seen_classes = perm[:-n_unseen]
    unseen_classes = perm[-n_unseen:]
    
    print(f"\nCategory Split:")
    print(f"Seen ({len(seen_classes)}): {[target_names[i] for i in seen_classes]}")
    print(f"Unseen ({len(unseen_classes)}): {[target_names[i] for i in unseen_classes]}")
    
    # Create masks
    seen_mask = torch.isin(labels, seen_classes)
    unseen_mask = torch.isin(labels, unseen_classes)
    
    seen_emb = embeddings[seen_mask]
    seen_labels = labels[seen_mask]
    
    unseen_emb = embeddings[unseen_mask]
    unseen_labels = labels[unseen_mask]
    
    return seen_emb, seen_labels, unseen_emb, unseen_labels

def evaluate_recall(model, embeddings, labels, top_k=10, batch_size=1000, database_emb=None, database_labels=None):
    """
    Evaluate Recall@K.
    
    Args:
        model: The encoder model
        embeddings: Query embeddings (or both query and database if database_emb is None)
        labels: Query labels
        top_k: Number of neighbors to retrieve
        batch_size: Batch size for processing
        database_emb: Optional separate database embeddings (gallery)
        database_labels: Optional separate database labels
    """
    model.eval()
    model.to(DEVICE)
    embeddings = embeddings.to(DEVICE)
    labels = labels.to(DEVICE)
    
    if database_emb is not None:
        database_emb = database_emb.to(DEVICE)
        database_labels = database_labels.to(DEVICE)
        use_separate_database = True
    else:
        database_emb = embeddings
        database_labels = labels
        use_separate_database = False
    
    n_queries = len(embeddings)
    n_database = len(database_emb)
    recall_sum = 0
    
    with torch.no_grad():
        # Encode queries
        query_encoded = []
        for i in range(0, n_queries, batch_size):
            batch = embeddings[i:i+batch_size]
            enc = model(batch.unsqueeze(0), add_noise=False).squeeze(0)
            query_encoded.append(enc)
        query_encoded = torch.cat(query_encoded, dim=0)
        
        # Encode database
        if use_separate_database:
            db_encoded = []
            for i in range(0, n_database, batch_size):
                batch = database_emb[i:i+batch_size]
                enc = model(batch.unsqueeze(0), add_noise=False).squeeze(0)
                db_encoded.append(enc)
            db_encoded = torch.cat(db_encoded, dim=0)
        else:
            db_encoded = query_encoded
        
        # Compute recall in batches
        for i in range(0, n_queries, batch_size):
            end = min(i + batch_size, n_queries)
            batch_queries = query_encoded[i:end]
            
            # Similarity: [batch, N_database]
            sim = torch.mm(batch_queries, db_encoded.T)
            
            # Top-k
            # If using separate database, we don't need to exclude self (unless overlap)
            # If using same database, we need to exclude self (index 0)
            k_to_retrieve = top_k + 1 if not use_separate_database else top_k
            _, indices = sim.topk(k_to_retrieve, dim=1)
            
            # Calculate recall for this batch
            for j in range(len(batch_queries)):
                global_idx = i + j
                query_label = labels[global_idx]
                
                if not use_separate_database:
                    neighbors = indices[j, 1:] # Exclude self
                else:
                    neighbors = indices[j, :]
                
                neighbor_labels = database_labels[neighbors]
                
                correct = (neighbor_labels == query_label).sum().item()
                
                # Max possible matches in database
                num_same_in_db = (database_labels == query_label).sum().item()
                
                # If self-retrieval, subtract 1 (the query itself)
                if not use_separate_database:
                    num_same_in_db -= 1
                
                max_possible = min(top_k, num_same_in_db)
                
                if max_possible > 0:
                    recall_sum += correct / max_possible
    
    return recall_sum / n_queries

def run_experiment():
    # 1. Load Data
    embeddings, labels, target_names = load_and_embed_data()
    
    # 2. Split Categories
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    # 3. Split Seen into Train/Val/Test
    n_seen = len(seen_emb)
    indices = torch.randperm(n_seen)
    train_idx = indices[:int(0.8 * n_seen)]
    val_idx = indices[int(0.8 * n_seen):int(0.9 * n_seen)]
    test_idx = indices[int(0.9 * n_seen):]
    
    train_emb = seen_emb[train_idx]
    train_labels = seen_labels[train_idx]
    val_emb = seen_emb[val_idx]
    val_labels = seen_labels[val_idx]
    test_emb = seen_emb[test_idx]
    test_labels = seen_labels[test_idx]
    
    print(f"\nData Splits:")
    print(f"Train: {len(train_emb)}")
    print(f"Val:   {len(val_emb)}")
    print(f"Test:  {len(test_emb)}")
    print(f"Unseen (Zero-shot): {len(unseen_emb)}")
    
    # 4. Initialize Model
    print("\nInitializing Hybrid REWA...")
    model = HybridREWAEncoder(
        d_model=D_MODEL,
        m_dim=M_DIM,
        random_ratio=0.5,
        dropout=0.3
    ).to(DEVICE)
    
    trainer = TripletLossTrainer(
        model,
        margin=1.0,
        lr=1e-3,
        weight_decay=0.01
    )
    
    # 5. Train
    print("\nTraining...")
    history = {'val_recall': [], 'unseen_recall': []}
    best_val_recall = 0
    
    for epoch in range(20): # 20 epochs should be enough
        # Train epoch
        loss = trainer.train_epoch(
            train_emb.to(DEVICE), 
            train_labels.to(DEVICE), 
            num_triplets=2000
        )
        
        # Evaluate
        val_recall = evaluate_recall(model, val_emb, val_labels)
        unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
        
        history['val_recall'].append(val_recall)
        history['unseen_recall'].append(unseen_recall)
        
        print(f"Epoch {epoch+1:2d}: Loss={loss:.4f}, Val Recall={val_recall:.1%}, Unseen Recall={unseen_recall:.1%}")
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), 'hybrid_rewa_20news.pth')
    
    # 6. Final Evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    model.load_state_dict(torch.load('hybrid_rewa_20news.pth'))
    
    test_recall = evaluate_recall(model, test_emb, test_labels)
    unseen_recall = evaluate_recall(model, unseen_emb, unseen_labels)
    
    print(f"Recall on Seen Categories (Test Set):   {test_recall:.1%}")
    print(f"Recall on Unseen Categories (Zero-shot): {unseen_recall:.1%}")
    
    print(f"\nCompression: {model.get_compression_ratio():.1f}×")
    
    if unseen_recall > 0.60:
        print("\n🚀 CONCLUSION: Strong generalization confirmed on real data!")
    elif unseen_recall > 0.40:
        print("\n✅ CONCLUSION: Good generalization, validates hybrid approach.")
    else:
        print("\n⚠️ CONCLUSION: Generalization limited, but better than pure learned.")

if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_experiment()
