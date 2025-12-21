import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import ssl

# Bypass SSL certificate verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# 1. NIRODHA CONVOLUTIONAL & LINEAR WRAPPERS
# ============================================================================

class NirodhaWrapper(nn.Module):
    """Wraps any layer (Linear, Conv2d) with Nirodha stabilization"""
    def __init__(self, layer, beta=0.1):
        super().__init__()
        self.layer = layer
        self.beta = beta
        self.anchor = None
        
    def set_anchor(self, state_dict):
        self.anchor = {k: v.clone() for k, v in state_dict.items()}
        
    def nirodha_operator(self, x):
        return x / (1 + self.beta * torch.abs(x))
    
    def forward(self, x, residual=False):
        weight = self.layer.weight
        bias = self.layer.bias
        
        if self.anchor is not None:
            anchor_weight = self.anchor['layer.weight']
            weight = anchor_weight + self.nirodha_operator(weight - anchor_weight)
            if self.layer.bias is not None:
                anchor_bias = self.anchor['layer.bias']
                bias = anchor_bias + self.nirodha_operator(bias - anchor_bias)
        
        # Use underlying layer for forward, but override weights
        if isinstance(self.layer, nn.Conv2d):
            out = F.conv2d(x, weight, bias, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups)
        else:
            out = F.linear(x, weight, bias)
            
        if residual:
            return x + F.relu(out)
        return out

# ============================================================================
# 2. CONVOLUTIONAL NIRODHA-D MODEL
# ============================================================================

class ConvNirodhaNet(nn.Module):
    def __init__(self, initial_depth=2):
        super().__init__()
        # Basic convolutional feature extractor
        self.conv1 = NirodhaWrapper(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        self.conv2 = NirodhaWrapper(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        
        # Dynamic layers (starts with Linear for simplicity after pool)
        self.fc_layers = nn.ModuleList([
            NirodhaWrapper(nn.Linear(64 * 8 * 8, 256)),
            NirodhaWrapper(nn.Linear(256, 256))
        ])
        self.initial_depth = initial_depth
        
        # Split heads (0-4 and 5-9)
        self.head_a = NirodhaWrapper(nn.Linear(256, 5))
        self.head_b = NirodhaWrapper(nn.Linear(256, 5))
        
    def set_anchor(self, beta=1000.0):
        self.conv1.set_anchor(self.conv1.state_dict())
        self.conv1.beta = beta
        self.conv2.set_anchor(self.conv2.state_dict())
        self.conv2.beta = beta
        for layer in self.fc_layers:
            layer.set_anchor(layer.state_dict())
            layer.beta = beta
        self.head_a.set_anchor(self.head_a.state_dict())
        self.head_a.beta = beta
        
    def add_layers(self, n=2):
        new_layers = [NirodhaWrapper(nn.Linear(256, 256), beta=0.1) for _ in range(n)]
        self.fc_layers.extend(new_layers)
        
    def forward(self, x, task='a'):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        
        for i, layer in enumerate(self.fc_layers):
            is_res = (i >= self.initial_depth)
            x = layer(x, residual=is_res)
            if not is_res: x = F.relu(x)
            
            if task == 'a' and i == self.initial_depth - 1:
                return self.head_a(x)
        return self.head_b(x)

class SimpleConvNet(nn.Module):
    """Baseline for CIFAR-10"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head_a = nn.Linear(256, 5)
        self.head_b = nn.Linear(256, 5)
        
    def forward(self, x, task='a'):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head_a(x) if task == 'a' else self.head_b(x)

# ============================================================================
# 3. DATA LOADING
# ============================================================================

def get_split_cifar(train=True, digits_subset=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    dataset = datasets.CIFAR10('./data', train=train, download=True, transform=transform)
    if digits_subset is not None:
        mapping = {d: i for i, d in enumerate(digits_subset)}
        indices = [i for i, (_, label) in enumerate(dataset) if label in digits_subset]
        subset = Subset(dataset, indices)
        class RemappedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, mapping):
                self.subset = subset
                self.mapping = mapping
            def __getitem__(self, index):
                data, label = self.subset[index]
                return data, self.mapping[label]
            def __len__(self):
                return len(self.subset)
        return DataLoader(RemappedSubset(subset, mapping), batch_size=128, shuffle=True)
    return DataLoader(dataset, batch_size=128, shuffle=True)

# ============================================================================
# 4. TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, loader, optimizer, device, task='a'):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, task=task)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, task='a'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data, task=task)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

# ============================================================================
# 5. EXPERIMENT RUNNER
# ============================================================================

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    task_a_train = get_split_cifar(train=True, digits_subset=[0,1,2,3,4])
    task_a_test = get_split_cifar(train=False, digits_subset=[0,1,2,3,4])
    task_b_train = get_split_cifar(train=True, digits_subset=[5,6,7,8,9])
    task_b_test = get_split_cifar(train=False, digits_subset=[5,6,7,8,9])
    
    results = {}
    
    # BASELINE
    print("=" * 60)
    print("BASELINE: CIFAR-10 Sequential Multi-Head")
    print("=" * 60)
    baseline = SimpleConvNet().to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
    
    print("\nPhase 1: Training on classes 0-4 (5 epochs)...")
    for epoch in range(5):
        loss = train_epoch(baseline, task_a_train, optimizer, device, task='a')
        acc = evaluate(baseline, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Task A Acc={acc:.2f}%")
    
    a_before = evaluate(baseline, task_a_test, device, task='a')
    
    print("\nPhase 2: Fine-tuning on classes 5-9 (10 epochs)...")
    for epoch in range(10):
        train_epoch(baseline, task_b_train, optimizer, device, task='b')
        acc_b = evaluate(baseline, task_b_test, device, task='b')
        acc_a = evaluate(baseline, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Task B Acc={acc_b:.2f}%, Task A Acc={acc_a:.2f}%")
    
    results['baseline'] = {'before': a_before, 'after': evaluate(baseline, task_a_test, device, task='a')}
    
    # NIRODHA-D
    print("\n" + "=" * 60)
    print("NIRODHA-D: CIFAR-10 Residual Stability")
    print("=" * 60)
    nirodha = ConvNirodhaNet().to(device)
    optimizer = torch.optim.Adam(nirodha.parameters(), lr=0.001)
    
    print("\nPhase 1: Training on classes 0-4 (5 epochs)...")
    for epoch in range(5):
        loss = train_epoch(nirodha, task_a_train, optimizer, device, task='a')
        acc = evaluate(nirodha, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Task A Acc={acc:.2f}%")
        
    a_before = evaluate(nirodha, task_a_test, device, task='a')
    
    print("\n🔒 Anchoring Base Knowledge (beta=1000)...")
    nirodha.set_anchor(beta=1000.0)
    print("📈 Adding 2 residual layers for expansion...")
    nirodha.add_layers(n=2)
    nirodha.to(device)
    
    optimizer = torch.optim.Adam(nirodha.parameters(), lr=0.0005)
    
    print("\nPhase 2: Training on classes 5-9 (10 epochs)...")
    for epoch in range(10):
        train_epoch(nirodha, task_b_train, optimizer, device, task='b')
        acc_b = evaluate(nirodha, task_b_test, device, task='b')
        acc_a = evaluate(nirodha, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Task B Acc={acc_b:.2f}%, Task A Acc={acc_a:.2f}%")
        
    results['nirodha'] = {'before': a_before, 'after': evaluate(nirodha, task_a_test, device, task='a')}
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Baseline Forgetting: {results['baseline']['before'] - results['baseline']['after']:.2f}%")
    print(f"   Nirodha-D Forgetting: {results['nirodha']['before'] - results['nirodha']['after']:.2f}%")
    
    # Plotting
    plt.figure(figsize=(8, 4))
    methods = ['Baseline', 'Nirodha-D']
    forgetting = [results['baseline']['before'] - results['baseline']['after'], 
                  results['nirodha']['before'] - results['nirodha']['after']]
    plt.bar(methods, forgetting, color=['red', 'green'])
    plt.ylabel('Forgetting Amount (%)')
    plt.title('Catastrophic Forgetting: CIFAR-10')
    plt.savefig('nirodha_cifar10_results.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
