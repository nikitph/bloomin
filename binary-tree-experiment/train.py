"""
Training script for Binary Tree Path Prediction

Trains both standard and hyperbolic branch Transformers on the same task.
Compares their performance at different model sizes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm

from tree_dataset import BinaryTreePathDataset, collate_batch
from standard_transformer import StandardTransformer
from hyperbolic_transformer import HierarchicalBranchTransformer


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, targets)
            
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nCreating datasets (depth={args.depth})...")
    train_dataset = BinaryTreePathDataset(
        num_samples=args.train_samples,
        depth=args.depth,
        value_range=args.value_range,
        seed=args.seed
    )
    
    val_dataset = BinaryTreePathDataset(
        num_samples=args.val_samples,
        depth=args.depth,
        value_range=args.value_range,
        seed=args.seed + 1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    vocab_size = train_dataset.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {train_dataset.get_max_seq_length()}")
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    if args.model_type == 'standard':
        model = StandardTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_classes=args.value_range,
            dropout=args.dropout
        )
    elif args.model_type == 'hyperbolic':
        model = HierarchicalBranchTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            bucket_sizes=tuple(args.bucket_sizes),
            num_classes=args.value_range,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(
                    args.save_dir,
                    f"{args.model_type}_d{args.d_model}_best.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'args': vars(args)
                }, save_path)
                print(f"  → Saved best model to {save_path}")
    
    # Save final results
    if args.save_dir:
        results = {
            'model_type': args.model_type,
            'd_model': args.d_model,
            'n_params': model.count_parameters(),
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'history': history,
            'args': vars(args)
        }
        
        results_path = os.path.join(
            args.save_dir,
            f"{args.model_type}_d{args.d_model}_results.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {results_path}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train binary tree path prediction models')
    
    # Data
    parser.add_argument('--depth', type=int, default=20, help='Tree depth')
    parser.add_argument('--value-range', type=int, default=1000, help='Range of leaf values')
    parser.add_argument('--train-samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=2000, help='Number of validation samples')
    
    # Model
    parser.add_argument('--model-type', type=str, required=True, choices=['standard', 'hyperbolic'],
                        help='Model type')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--bucket-sizes', type=int, nargs='+', default=[256, 64, 16],
                        help='Bucket sizes for hyperbolic model')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Saving
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    main(args)
