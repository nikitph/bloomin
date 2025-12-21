import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# 1. NIRODHA LAYER
# ============================================================================

class NirodhaLayer(nn.Module):
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
        if self.anchor is not None:
            anchor_w = self.anchor['linear.weight']
            weight = anchor_w + self.nirodha_operator(weight - anchor_w)
            if self.linear.bias is not None:
                anchor_b = self.anchor['linear.bias']
                bias = anchor_b + self.nirodha_operator(bias - anchor_b)
        out = F.linear(x, weight, bias)
        if residual: return x + F.relu(out)
        return out

# ============================================================================
# 2. PERMUTED MNIST DATA LOADER
# ============================================================================

def get_permuted_mnist(permutation):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(-1)[permutation].view(1, 28, 28)
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=128, shuffle=True), DataLoader(test, batch_size=128, shuffle=False)

# ============================================================================
# 3. NIRODHA-D PERMUTATION MODEL
# ============================================================================

class PermutedNirodhaNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, initial_depth=2):
        super().__init__()
        self.fc1 = NirodhaLayer(input_dim, hidden_dim)
        self.initial_depth = initial_depth
        self.layers = nn.ModuleList([
            NirodhaLayer(hidden_dim, hidden_dim) for _ in range(initial_depth)
        ])
        # Multi-Head setup: one head per potential task (max 20)
        self.heads = nn.ModuleList([
            NirodhaLayer(hidden_dim, 10) for _ in range(20)
        ])
        
    def set_anchor(self, beta=1000.0, task_id=0):
        self.fc1.set_anchor(self.fc1.state_dict())
        self.fc1.beta = beta
        for layer in self.layers:
            layer.set_anchor(layer.state_dict())
            layer.beta = beta
        # Also anchor the head for the task we just learned
        self.heads[task_id].set_anchor(self.heads[task_id].state_dict())
        self.heads[task_id].beta = beta
        
    def add_layers(self, n=1):
        self.layers.append(NirodhaLayer(256, 256, beta=0.1))
        
    def forward(self, x, task_id=0):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        for i, layer in enumerate(self.layers):
            is_res = (i >= self.initial_depth)
            x = layer(x, residual=is_res)
            if not is_res: x = F.relu(x)
        return self.heads[task_id](x)

# ============================================================================
# 4. EXPERIMENT
# ============================================================================

def run_experiment(num_tasks=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Pre-generate permutations
    permutations = [np.random.permutation(784) for _ in range(num_tasks)]
    
    model = PermutedNirodhaNet().to(device)
    accuracies = [] # Store accuracies of Task 1 after each subsequent task
    
    for t in range(num_tasks):
        print(f"\n--- TASK {t+1} START ---")
        train_loader, test_loader = get_permuted_mnist(permutations[t])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001 if t==0 else 0.0005)
        
        # Train
        for epoch in range(3):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data, task_id=t)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            acc = evaluate(model, test_loader, device, task_id=t)
            print(f"Task {t+1} Epoch {epoch+1}: Acc={acc:.2f}%")
            
        # Re-evaluate Task 1 (First Permutation)
        t1_train, t1_test = get_permuted_mnist(permutations[0])
        t1_acc = evaluate(model, t1_test, device, task_id=0)
        accuracies.append(t1_acc)
        print(f"Task 1 Current Accuracy: {t1_acc:.2f}%")
        
        # Prepare for next task: Anchor and Add Depth
        if t < num_tasks - 1:
            print("🔒 Anchoring Current Knowledge...")
            model.set_anchor(beta=1000.0, task_id=t)
            print("📈 Increasing Depth...")
            model.add_layers(1)
            model.to(device)
            
    # Final Summary
    print("\n📊 RETENTION SUMMARY (Task 1 Accuracy across tasks):")
    for i, acc in enumerate(accuracies):
        print(f"After Task {i+1}: {acc:.2f}%")
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_tasks + 1), accuracies, marker='o', color='blue')
    plt.xlabel('Number of Tasks Learned')
    plt.ylabel('Task 1 Accuracy (%)')
    plt.title('Permuted MNIST: Cumulative Retention (Nirodha-D)')
    plt.grid(True)
    plt.savefig('nirodha_permuted_mnist.png')
    plt.close()

def evaluate(model, loader, device, task_id=0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data, task_id=task_id)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

if __name__ == "__main__":
    run_experiment(num_tasks=5) # Start with 5 for the PoC
