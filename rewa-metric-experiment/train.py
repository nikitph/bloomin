import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from data import get_cifar100_loader, get_wiki_loader
from models import REWAModel
from losses import InfoNCELoss
from tqdm import tqdm
import time

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # 1. Data
    if args.modality == 'vision':
        train_loader, _ = get_cifar100_loader(batch_size=args.batch_size)
    else:
        train_loader, _ = get_wiki_loader(batch_size=args.batch_size)
        if train_loader is None:
            print("Failed to load text data")
            return

    # 2. Model
    # Backbone mapping
    if args.modality == 'vision':
        backbone = 'resnet18' 
    else:
        backbone = 'prajjwal1/bert-tiny' # Use tiny BERT for speed in demo/dev
        # backbone = 'bert-base-uncased' # Full scale
        
    model = REWAModel(backbone, mode=args.variant, out_dim=args.dim).to(device)
    
    # 3. Loss
    # Normalized -> Cosine
    # REWA/Unnorm -> Euclidean
    loss_mode = 'cosine' if args.variant == 'normalized' else 'euclidean'
    criterion = InfoNCELoss(mode=loss_mode, temperature=args.tau).to(device)
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Logging
    run_name = f"{args.modality}_{args.variant}_s{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Loop
    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            if args.modality == 'vision':
                # batch is [ [view1], [view2], label ]
                # But DataLoader collates [ [v1_batch], [v2_batch] ] from TwoCropTransform
                # Actually TwoCropTransform returns list [q, k].
                # Collate fn might stack them? 
                # Standard PyTorch DataLoader with list items:
                # batch is [ [B, C, H, W], [B, C, H, W], [B] labels ]
                # Wait, TwoCropTransform returns [q, k]. Visual dataset returns (img, target).
                # So dataset[i] is ([q, k], target).
                # DataLoader collate: 
                # batch[0] is list of [q_Tensor_batch, k_Tensor_batch]
                # batch[1] is target
                
                # Check structure
                views = batch[0] # List of 2 tensors
                x1, x2 = views[0].to(device), views[1].to(device)
            else:
                # Text: batch is list of strings? No, WikiDataset returns string.
                # Problem: We need Multi-view.
                # SimCSE usually does dropout twice on same input.
                # So we pass x twice to modle.
                # But model expects tokenized input.
                # We need a collate_fn for text that tokenizes.
                # Or tokenize inside loop (inefficient).
                # For this demo, let's assume get_wiki_loader handles it? 
                # Implementation of get_wiki_loader in data.py was naive. 
                # Let's fix text handling on the fly:
                # We get raw sentences. Tokenize here.
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(backbone)
                texts = batch
                inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=64).to(device)
                
                # For SimCSE: Pass same input twice. Dropout in BERT will generate mismatched views.
                x1 = inputs
                x2 = inputs
                
                # REWAModel forward for text handles dict {'input_ids': ...}
                # So x1, x2 are dicts.

            optimizer.zero_grad()
            
            # Forward
            # v1, u1, r1 = model(x1)
            # v2, u2, r2 = model(x2)
            
            if args.modality == 'vision':
                v1, _, r1 = model(x1)
                v2, _, r2 = model(x2)
            else:
                v1, _, r1 = model(x1)
                # Need to run forward again to get different dropout mask? 
                # Yes, calling model(x1) twice works for dropout.
                v2, _, r2 = model(x2)
                
            loss = criterion(v1, v2)
            
            # Optional: Radius regularizer (keep r near 1, avoid explosion)
            if args.variant == 'rewa':
                # KL(log r || N(0, 1)) or just penalize (mean(r) - 1)^2
                # Simple: mean squared log radius
                reg = args.reg * torch.mean((torch.log(r1) - 0.0)**2) 
                loss += reg
            
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train', loss.item(), step)
            if args.variant in ['rewa', 'unnormalized']:
                writer.add_scalar('Radius/mean', r1.mean().item(), step)
            
            step += 1
            pbar.set_postfix({'loss': loss.item()})
            
    # Save
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f"checkpoints/{run_name}.pt"
    model.save(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True, choices=['vision', 'text'])
    parser.add_argument('--variant', type=str, required=True, choices=['normalized', 'unnormalized', 'rewa'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=0.1, help="Radial regularization strength")
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    train(args)
