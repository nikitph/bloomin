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

# Bypass SSL certificate verification for MNIST download
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# 1. NIRODHA OPERATOR
# ============================================================================

class NirodhaLayer(nn.Module):
    """Universal layer with Nirodha regulation for weights"""
    def __init__(self, in_dim, out_dim, beta=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.beta = beta
        self.anchor = None
        
    def set_anchor(self, state_dict):
        self.anchor = {k: v.clone() for k, v in state_dict.items()}
        
    def nirodha_operator(self, x):
        return x / (1 + self.beta * torch.abs(x))
    
    def forward(self, x, residual=False):
        weight = self.linear.weight
        bias = self.linear.bias
        
        # Apply Nirodha suppression relative to anchor (Always, not just training)
        if self.anchor is not None:
            anchor_weight = self.anchor['linear.weight']
            weight = anchor_weight + self.nirodha_operator(weight - anchor_weight)
            if 'linear.bias' in self.anchor:
                anchor_bias = self.anchor['linear.bias']
                bias = anchor_bias + self.nirodha_operator(bias - anchor_bias)
        
        out = F.linear(x, weight, bias)
        if residual:
            return x + F.relu(out)
        return out

# ============================================================================
# 2. MODELS
# ============================================================================

class SimpleNet(nn.Module):
    """Baseline: Sharing layers, separate heads, NO Nirodha"""
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head_a = nn.Linear(hidden_dim, 5) 
        self.head_b = nn.Linear(hidden_dim, 5)
        
    def forward(self, x, task='a'):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head_a(x) if task == 'a' else self.head_b(x)

class ProgressiveDepthNet(nn.Module):
    """Nirodha-D: Residual extension with Anchor Short-Circuiting"""
    def __init__(self, input_dim=784, hidden_dim=128, initial_layers=2):
        super().__init__()
        self.input_proj = NirodhaLayer(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            NirodhaLayer(hidden_dim, hidden_dim) for _ in range(initial_layers)
        ])
        self.initial_depth = initial_layers
        self.head_a = NirodhaLayer(hidden_dim, 5)
        self.head_b = NirodhaLayer(hidden_dim, 5)
        
    def set_anchor(self, beta=1000.0):
        """Freeze the base with extremely high beta suppression"""
        self.input_proj.set_anchor(self.input_proj.state_dict())
        self.input_proj.beta = beta
        for layer in self.layers:
            layer.set_anchor(layer.state_dict())
            layer.beta = beta
        self.head_a.set_anchor(self.head_a.state_dict())
        self.head_a.beta = beta
        
    def add_layers(self, n=2):
        new_layers = [NirodhaLayer(128, 128, beta=0.1) for _ in range(n)]
        self.layers.extend(new_layers)
        
    def forward(self, x, task='a'):
        x = x.view(-1, 784)
        x = F.relu(self.input_proj(x))
        for i, layer in enumerate(self.layers):
            is_res = (i >= self.initial_depth)
            x = layer(x, residual=is_res)
            if not is_res: x = F.relu(x)
            
            # Key: Task A head only uses the features it was trained on
            if task == 'a' and i == self.initial_depth - 1:
                return self.head_a(x)
        return self.head_b(x)

# ============================================================================
# 3. DATA LOADING (Relative Labels 0-4)
# ============================================================================

def get_split_mnist(train=True, digits_subset=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = './data'
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
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
    """Get MNIST with optional digit filtering and relative labels"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_dir = './data'
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    
    if digits_subset is not None:
        # Create a mapping to 0-4
        mapping = {d: i for i, d in enumerate(digits_subset)}
        
        indices = []
        # filtered_labels = [] # This variable was unused
        for i, (_, label) in enumerate(dataset):
            if label in digits_subset:
                indices.append(i)
                # We'll need to override labels in the dataset or handle it in training
        
        subset = Subset(dataset, indices)
        # Decorator to remap labels
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
# 5. THE EXPERIMENT
# ============================================================================

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Task A: digits 0-4
    # Task B: digits 5-9
    task_a_train = get_split_mnist(train=True, digits_subset=[0,1,2,3,4])
    task_a_test = get_split_mnist(train=False, digits_subset=[0,1,2,3,4])
    task_b_train = get_split_mnist(train=True, digits_subset=[5,6,7,8,9])
    task_b_test = get_split_mnist(train=False, digits_subset=[5,6,7,8,9])
    
    results = {}
    
    # ========================================================================
    # BASELINE: Sequential with Separate Heads
    # ========================================================================
    print("=" * 60)
    print("BASELINE: Multi-Head Sequential (Feature Drift Test)")
    print("=" * 60)
    
    baseline = SimpleNet().to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
    
    print("\nPhase 1: Training on digits 0-4...")
    for epoch in range(3):
        loss = train_epoch(baseline, task_a_train, optimizer, device, task='a')
        acc = evaluate(baseline, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Task A Acc={acc:.2f}%")
    
    task_a_before = evaluate(baseline, task_a_test, device, task='a')
    
    print("\nPhase 2: Fine-tuning on digits 5-9 (10 epochs)...")
    for epoch in range(10):
        loss = train_epoch(baseline, task_b_train, optimizer, device, task='b')
        acc_b = evaluate(baseline, task_b_test, device, task='b')
        acc_a = evaluate(baseline, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Task B Acc={acc_b:.2f}%, Task A Acc={acc_a:.2f}%")
    
    task_a_after = evaluate(baseline, task_a_test, device, task='a')
    task_b_final = evaluate(baseline, task_b_test, device, task='b')
    results['baseline'] = {
        'task_a_before': task_a_before, 'task_a_after': task_a_after,
        'task_b_final': task_b_final, 'forgetting': task_a_before - task_a_after
    }
    
    # ========================================================================
    # NIRODHA-D: Anchoring + Added Depth
    # ========================================================================
    print("\n" + "=" * 60)
    print("NIRODHA-D: Multi-Head Residual (Nirodha-D Stability)")
    print("=" * 60)
    
    nirodha = ProgressiveDepthNet().to(device)
    optimizer = torch.optim.Adam(nirodha.parameters(), lr=0.001)
    
    print("\nPhase 1: Training on digits 0-4...")
    for epoch in range(3):
        loss = train_epoch(nirodha, task_a_train, optimizer, device, task='a')
        acc = evaluate(nirodha, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Task A Acc={acc:.2f}%")
    
    task_a_before = evaluate(nirodha, task_a_test, device, task='a')
    
    print("\n🔒 Setting anchor to preserve Task A features (beta=1000)...")
    nirodha.set_anchor(beta=1000.0)
    
    print("📈 Adding 3 new residual layers for Task B extension...")
    nirodha.add_layers(n=3)
    
    optimizer = torch.optim.Adam(nirodha.parameters(), lr=0.0005)
    
    print("\nPhase 2: Training on digits 5-9 (10 epochs)...")
    for epoch in range(10):
        loss = train_epoch(nirodha, task_b_train, optimizer, device, task='b')
        acc_b = evaluate(nirodha, task_b_test, device, task='b')
        acc_a = evaluate(nirodha, task_a_test, device, task='a')
        print(f"Epoch {epoch+1}: Task B Acc={acc_b:.2f}%, Task A Acc={acc_a:.2f}%")
    
    task_a_after = evaluate(nirodha, task_a_test, device, task='a')
    task_b_final = evaluate(nirodha, task_b_test, device, task='b')
    results['nirodha'] = {
        'task_a_before': task_a_before, 'task_a_after': task_a_after,
        'task_b_final': task_b_final, 'forgetting': task_a_before - task_a_after
    }
    
    # Logging & Plotting
    print(f"\n📊 FINAL COMPARISON:")
    print(f"   Baseline Forgetting: {results['baseline']['forgetting']:.2f}%")
    print(f"   Nirodha-D Forgetting: {results['nirodha']['forgetting']:.2f}%")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    methods = ['Baseline', 'Nirodha-D']
    before = [results['baseline']['task_a_before'], results['nirodha']['task_a_before']]
    after = [results['baseline']['task_a_after'], results['nirodha']['task_a_after']]
    x = np.arange(len(methods))
    plt.bar(x - 0.2, before, 0.4, label='Before Task B', color='green', alpha=0.7)
    plt.bar(x + 0.2, after, 0.4, label='After Task B', color='red', alpha=0.7)
    plt.ylabel('Accuracy (%)')
    plt.title('Task A Retention')
    plt.xticks(x, methods)
    plt.legend()
    plt.ylim([0, 100])
    plt.subplot(1, 2, 2)
    forgetting = [results['baseline']['forgetting'], results['nirodha']['forgetting']]
    plt.bar(methods, forgetting, color=['red', 'green'], alpha=0.7)
    plt.ylabel('Forgetting (%)')
    plt.title('Catastrophic Forgetting Amount')
    plt.ylim([0, 100])
    for i, v in enumerate(forgetting):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('nirodha_proof_of_concept.png', dpi=150, bbox_inches='tight')
    return results

if __name__ == "__main__":
    import time
    start_time = time.time()
    run_experiment()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
