import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random

# --- Vision: CIFAR-100 (SimCLR) ---

class TwoCropTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def get_cifar100_loader(batch_size=128, num_workers=2):
    print("Loading CIFAR-100 Data...")
    
    # SimCLR Augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    # Test transform (Resize + CenterCrop? No, CIFAR is 32x32. Just Normalize)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, 
        transform=TwoCropTransform(train_transform), 
        download=True
    )
    
    # For Eval (Retrieval), we need a query set and a gallery set.
    # We can use the Test set. 
    # Or split Test into Query/Gallery. 
    # Standard: Use Test set as both (Retrieval within test set, mask self).
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, 
        transform=test_transform, 
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader

# --- Text: Wikipedia (SimCSE) ---

class WikiDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        return self.sentences[idx]

def get_wiki_loader(batch_size=64, max_samples=100000):
    print(f"Loading Wikipedia Data (Top {max_samples})...")
    # Download WikiText-2 raw
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    try:
        if not os.path.exists('./data/wiki.txt'):
            os.makedirs('./data', exist_ok=True)
            urllib.request.urlretrieve(url, './data/wiki.txt')
        
        with open('./data/wiki.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Split into sentences (naive)
        sentences = [s.strip() for s in text.split('\n') if len(s.split()) > 5] # Min length 5 words
        
        if len(sentences) > max_samples:
            sentences = sentences[:max_samples]
            
        print(f"  ✓ Loaded {len(sentences)} sentences")
        
        # Split Train/Test
        split_idx = int(len(sentences) * 0.9)
        train_sentences = sentences[:split_idx]
        test_sentences = sentences[split_idx:]
        
        train_ds = WikiDataset(train_sentences)
        test_ds = WikiDataset(test_sentences)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error loading Wiki: {e}")
        return None, None

def check_data():
    c_train, c_test = get_cifar100_loader()
    print(f"CIFAR Train batches: {len(c_train)}")
    
    w_train, w_test = get_wiki_loader()
    if w_train:
        print(f"Wiki Train batches: {len(w_train)}")

if __name__ == "__main__":
    check_data()
