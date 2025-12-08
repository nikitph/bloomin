import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import argparse
import json
import os
from data import get_cifar100_loader, get_wiki_loader
from models import REWAModel
from tqdm import tqdm
from transformers import AutoTokenizer

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # 1. Load Model
    if args.modality == 'vision':
        backbone = 'resnet18'
    else:
        backbone = 'prajjwal1/bert-tiny'
        
    model = REWAModel(backbone, mode=args.variant, out_dim=args.dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # 2. Get embeddings
    print("Extracting embeddings...")
    embeddings = []
    
    if args.modality == 'vision':
        _, test_loader = get_cifar100_loader(batch_size=args.batch_size)
    else:
        _, test_loader = get_wiki_loader(batch_size=args.batch_size)
        tokenizer = AutoTokenizer.from_pretrained(backbone)
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if args.modality == 'vision':
                x = batch[0].to(device)
            else:
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=64).to(device)
                x = inputs
                
            v, _, r = model(x)
            embeddings.append(v.cpu())
            
    embeddings = torch.cat(embeddings, dim=0) # [N, D]
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 3. Metrics
    results = {}
    
    # Intrinsic Dimension
    print("Computing Intrinsic Dimension...")
    X = embeddings.numpy()
    # Center
    X_centered = X - X.mean(axis=0)
    # PCA
    pca = PCA()
    pca.fit(X_centered)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d95 = np.searchsorted(cumsum, 0.95) + 1
    results['intrinsic_dim_95'] = int(d95)
    
    # Radius Stats
    norms = torch.norm(embeddings, dim=1)
    results['norm_mean'] = float(norms.mean())
    results['norm_std'] = float(norms.std())
    results['norm_cv'] = float(norms.std() / norms.mean())
    
    # Retrieval (Recall@K)
    # For Unsupervised, we check "Instance Retrieval" (Augmented view retrieval)
    # But here we only embedded one view per image?
    # Wait, the Standard Eval for SimCLR is Linear Probing (Classification).
    # The user asked for "Retrieval: Recall@1/5/10".
    # On CIFAR-100, Retrieval usually means: query is image, target is other images of SAME CLASS.
    # So we need Labels!
    # DataLoader returns labels.
    
    labels = []
    if args.modality == 'vision':
        for batch in test_loader:
            labels.append(batch[1])
    else:
        # Wiki has no class labels. SimCSE evaluates on STS (Semantic Textual Similarity) tasks.
        # Constructing "Same Class" retrieval for unsupervised text is hard without a label dataset.
        # User said "100k Wikipedia subset".
        # Maybe self-retrieval (dropout view)? 
        # Or standard STS.
        # Given "Agentic Mode" constraints, I will implement Class Retrieval for Vision
        # And for Text? I will skip Recall for Text unless I have labels.
        # Actually, user said: "Evaluation tasks: Nearest-neighbor retrieval".
        # If unsupervised, maybe he implies "Find nearest neighbors and check coverage"?
        # Or maybe he assumes known classes. Wiki has no classes.
        # I'll enable retrieval for Vision (CIFAR labels). 
        # For Text, I'll return dummy Recall or skip.
        pass
        
    if args.modality == 'vision':
        labels = torch.cat(labels).to(device)
        embeddings = embeddings.to(device)
        
        # Distance Matrix
        if args.variant == 'normalized':
            # Cosine distance ranking = Dot product ranking
            # High dot product = Close
            sim_matrix = torch.matmul(embeddings, embeddings.T) # [N, N]
            # dist = -sim
            dist_matrix = -sim_matrix
        else:
            # Euclidean distance
            # ||x-y||^2 = x^2 + y^2 - 2xy
            norms_sq = torch.sum(embeddings**2, dim=1, keepdim=True)
            dist_matrix = norms_sq + norms_sq.T - 2 * torch.matmul(embeddings, embeddings.T)
        
        # Mask self
        N = len(labels)
        dist_matrix.fill_diagonal_(float('inf'))
        
        # Get Top K
        # We want to retrieve items with SAME label.
        # Recall@K = Is there at least one item of same label in Top K?
        
        # indices: [N, K]
        k_max = 10
        _, indices = dist_matrix.topk(k_max, dim=1, largest=False)
        
        # Check matches
        retrieved_labels = labels[indices] # [N, K]
        # Query labels: [N, 1]
        query_labels = labels.view(-1, 1)
        
        matches = (retrieved_labels == query_labels) # [N, K] bool
        
        r1 = matches[:, :1].any(dim=1).float().mean().item()
        r5 = matches[:, :5].any(dim=1).float().mean().item()
        r10 = matches[:, :10].any(dim=1).float().mean().item()
        
        results['recall_1'] = r1
        results['recall_5'] = r5
        results['recall_10'] = r10
        
        print(f"Recall@1: {r1:.4f}")
        print(f"Recall@5: {r5:.4f}")
    
    print(f"Intrinsic Dim (95%): {d95}")
    print(f"Norm CV: {results['norm_cv']:.4f}")
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()
    
    evaluate(args)
