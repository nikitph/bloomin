"""
Data Utilities for 20 Newsgroups
================================

Handles loading, embedding (BERT), and splitting the 20 Newsgroups dataset.
Also provides the `evaluate_recall` function.
"""

import torch
import numpy as np
import ssl
import urllib.request
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_and_embed_data():
    """Load 20 Newsgroups and generate BERT embeddings."""
    print(f"Loading 20 Newsgroups dataset...")
    
    # Bypass SSL certificate verification for dataset download
    ssl._create_default_https_context = ssl._create_unverified_context
    
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
