# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# --- ACI Core with Seed Management ---

class NirodhaLinear(nn.Module):
    def __init__(self, in_features, out_features, beta=10.0):
        super().__init__()
        self.beta = beta
        self.weight_core = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_core = nn.Parameter(torch.Tensor(out_features))
        self.weight_v = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_v = nn.Parameter(torch.zeros(out_features))
        
        nn.init.kaiming_uniform_(self.weight_core, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_core)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_core, -bound, bound)

    def nirodha(self, V):
        return V / (1.0 + self.beta * torch.abs(V) + 1e-12)

    def forward(self, x):
        w_eff = self.weight_core + self.nirodha(self.weight_v)
        b_eff = self.bias_core + self.nirodha(self.bias_v)
        return F.linear(x, w_eff, b_eff)

    def get_seed(self, compress=False, threshold=0.01):
        """Extract current fluctuation as a seed."""
        w_v = self.weight_v.detach().clone()
        b_v = self.bias_v.detach().clone()
        if compress:
            # Simple pruning for Q2
            w_v[torch.abs(w_v) < threshold] = 0
            b_v[torch.abs(b_v) < threshold] = 0
        return (w_v, b_v)

    def set_seed(self, seed):
        """Restore a seed into the fluctuation parameters."""
        w_v, b_v = seed
        self.weight_v.data.copy_(w_v)
        self.bias_v.data.copy_(b_v)

    def commit_to_core(self):
        with torch.no_grad():
            self.weight_core.copy_(self.weight_core + self.nirodha(self.weight_v))
            self.bias_core.copy_(self.bias_core + self.nirodha(self.bias_v))
            self.weight_v.zero_()
            self.bias_v.zero_()

    def retrogress(self):
        """Erasure: V -> 0. Returns complexity cost (sum of absolute reset)."""
        with torch.no_grad():
            cost = torch.sum(torch.abs(self.weight_v)).item() + torch.sum(torch.abs(self.bias_v)).item()
            self.weight_v.zero_()
            self.bias_v.zero_()
            return cost

class ACIMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10, beta=10.0):
        super().__init__()
        self.fc1 = NirodhaLinear(input_dim, hidden_dim, beta=beta)
        self.fc2 = NirodhaLinear(hidden_dim, output_dim, beta=beta)
        self.seed_stack = []

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_state_seed(self, compress=False, threshold=0.01):
        return (self.fc1.get_seed(compress, threshold), self.fc2.get_seed(compress, threshold))

    def set_state_seed(self, state_seed):
        self.fc1.set_seed(state_seed[0])
        self.fc2.set_seed(state_seed[1])

    def push_task(self):
        """Save current core to stack and commit fluctuations to core."""
        # Save current core state
        core_snapshot = (
            (self.fc1.weight_core.detach().clone(), self.fc1.bias_core.detach().clone()),
            (self.fc2.weight_core.detach().clone(), self.fc2.bias_core.detach().clone())
        )
        self.seed_stack.append(core_snapshot)
        self.fc1.commit_to_core()
        self.fc2.commit_to_core()

    def pop_task(self):
        """Retrogress to previous core state."""
        if not self.seed_stack:
            return self.retrogress()
        
        # Discard current fluctuations
        self.fc1.retrogress()
        self.fc2.retrogress()
        
        # Restore previous core
        prev_core = self.seed_stack.pop()
        with torch.no_grad():
            self.fc1.weight_core.copy_(prev_core[0][0])
            self.fc1.bias_core.copy_(prev_core[0][1])
            self.fc2.weight_core.copy_(prev_core[1][0])
            self.fc2.bias_core.copy_(prev_core[1][1])
        return 0 # Cost not tracked for core pop in this demo

    def retrogress(self):
        return self.fc1.retrogress() + self.fc2.retrogress()

# --- Data Helpers ---

def get_mnist_subset(digits, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    def filter_digits(dataset):
        mask = torch.tensor([y in digits for y in dataset.targets])
        dataset.targets = dataset.targets[mask]
        dataset.data = dataset.data[mask]
        return dataset
    train_loader = torch.utils.data.DataLoader(filter_digits(train_dataset), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(filter_digits(test_dataset), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# --- Experiment Execution Functions ---

def train_epoch(model, loader, optimizer):
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(loader.dataset)

# --- Q1: ABCBA Sequence ---

def run_q1_abcba():
    print("\n--- Q1: Arbitrary Task Sequence (ABCBA) ---")
    model = ACIMLP(beta=10.0)
    
    # Task A: 0-1, Task B: 2-3, Task C: 4-5
    loaders = {
        'A': get_mnist_subset([0, 1]),
        'B': get_mnist_subset([2, 3]),
        'C': get_mnist_subset([4, 5])
    }
    
    # Learning Path A -> B -> C
    results = []
    
    # Learn A
    print("Learning Task A...")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    train_epoch(model, loaders['A'][0], opt)
    results.append(('A_after_A', evaluate(model, loaders['A'][1])))
    print(f"  Acc A: {results[-1][1]:.4f}")
    
    # Commit A, Learn B
    model.push_task()
    print("Learning Task B...")
    opt = optim.Adam([p for n, p in model.named_parameters() if '_v' in n], lr=1e-3) # Learn on fluctuation
    train_epoch(model, loaders['B'][0], opt)
    results.append(('B_after_B', evaluate(model, loaders['B'][1])))
    print(f"  Acc B: {results[-1][1]:.4f}")
    
    # Commit B, Learn C
    model.push_task()
    print("Learning Task C...")
    opt = optim.Adam([p for n, p in model.named_parameters() if '_v' in n], lr=1e-3)
    train_epoch(model, loaders['C'][0], opt)
    results.append(('C_after_C', evaluate(model, loaders['C'][1])))
    print(f"  Acc C: {results[-1][1]:.4f}")
    
    # Retrogression Path C -> B -> A
    print("Retrogressing C -> B...")
    model.retrogress() # Returns to Core (which is B state)
    results.append(('B_recon', evaluate(model, loaders['B'][1])))
    print(f"  Acc B Recov: {results[-1][1]:.4f}")
    
    print("Retrogressing B -> A...")
    model.pop_task() # Restores Core A, clears B fluctuations
    results.append(('A_recon', evaluate(model, loaders['A'][1])))
    print(f"  Acc A Recov: {results[-1][1]:.4f}")
    
    return results

# --- Q3: Branching ---

def run_q3_branching():
    print("\n--- Q3: Branching Timelines ---")
    model = ACIMLP(beta=10.0)
    
    # Root A: 0-1
    # Branch 1 (B): 2-3
    # Branch 2 (C): 8-9
    la, lb, lc = get_mnist_subset([0, 1]), get_mnist_subset([2, 3]), get_mnist_subset([8, 9])
    
    # Learn Root A
    print("Learning Root A...")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    train_epoch(model, la[0], opt)
    model.push_task() # A is now Core
    
    # Develop Branch B
    print("Developing Branch B...")
    opt = optim.Adam([p for n, p in model.named_parameters() if '_v' in n], lr=1e-3)
    for _ in range(2): train_epoch(model, lb[0], opt)
    seed_b = model.get_state_seed()
    acc_b = evaluate(model, lb[1])
    print(f"  Branch B Acc: {acc_b:.4f}")
    
    # Reset to Root A
    print("Resetting to Root A...")
    model.retrogress()
    acc_a_reset = evaluate(model, la[1])
    
    # Develop Branch C
    print("Developing Branch C...")
    opt = optim.Adam([p for n, p in model.named_parameters() if '_v' in n], lr=1e-3)
    for _ in range(2): train_epoch(model, lc[0], opt)
    acc_c = evaluate(model, lc[1])
    print(f"  Branch C Acc: {acc_c:.4f}")
    
    # Switch back to Branch B
    print("Switching back to Branch B via Seed Injection...")
    model.set_state_seed(seed_b)
    acc_b_recovered = evaluate(model, lb[1])
    print(f"  Branch B Recovered: {acc_b_recovered:.4f}")
    
    return acc_b, acc_c, acc_b_recovered

# --- Q2: Compression ---

def run_q2_compression():
    print("\n--- Q2: Seed Compression ---")
    model = ACIMLP(beta=10.0)
    l = get_mnist_subset([0, 1, 2, 3, 4, 5])
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    train_epoch(model, l[0], opt)
    
    acc_orig = evaluate(model, l[1])
    
    # Prune seed at different thresholds
    thresholds = [0.0, 0.001, 0.01, 0.05, 0.1]
    comp_results = []
    
    orig_seed = model.get_state_seed()
    
    for t in thresholds:
        pruned_seed = model.get_state_seed(compress=True, threshold=t)
        model.set_state_seed(pruned_seed)
        acc = evaluate(model, l[1])
        
        # Calculate sparseness
        w1_v, b1_v = pruned_seed[0]
        w2_v, b2_v = pruned_seed[1]
        sparsity = (torch.sum(w1_v == 0) + torch.sum(w2_v == 0)).item() / \
                   (w1_v.numel() + w2_v.numel())
        
        print(f"  Threshold {t}: Acc={acc:.4f}, Sparsity={sparsity*100:.1f}%")
        comp_results.append((sparsity, acc))
        
    return comp_results

# --- Q4: Thermodynamic Cost ---

def run_q4_thermo():
    print("\n--- Q4: Thermodynamic Cost of Retrogression ---")
    model = ACIMLP(beta=10.0)
    l = get_mnist_subset([0, 1])
    
    # 1. Cost of Learning (Task Energy)
    print("Learning Task...")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    # Track update distance
    total_learning_dist = 0
    for data, target in l[0]:
        old_params = [p.detach().clone() for p in model.parameters()]
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # Simplified: just one step
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        for i, p in enumerate(model.parameters()):
            total_learning_dist += torch.sum(torch.abs(p - old_params[i])).item()
        break # Just one batch for demo
        
    # 2. Cost of Retrogression (Erasure Energy)
    retro_cost = model.retrogress()
    print(f"  Learning Effort (Update Sum): {total_learning_dist:.2f}")
    print(f"  Retrogression Cost (State Erasure): {retro_cost:.2f}")
    
    return total_learning_dist, retro_cost

# --- Main ---

if __name__ == "__main__":
    q1 = run_q1_abcba()
    q3 = run_q3_branching()
    q2 = run_q2_compression()
    q4 = run_q4_thermo()
    
    # Plot Q2
    plt.figure(figsize=(10, 6))
    x, y = zip(*q2)
    plt.plot(x, y, marker='o', linewidth=2, color='green')
    plt.title("Constraint-Preserving Compression: Accuracy vs. Seed Sparsity")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.savefig('advanced_q2_compression.png')
    
    # Final Summary Table
    print("\nAdvanced Research Results Summary")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Q1: Sequential Reversion (B) | {q1[-1][1]*100:.1f}% |")
    print(f"| Q3: Branch Recovery (B) | {q3[2]*100:.1f}% |")
    print(f"| Q2: Max Sparsity at >95% Acc | {max([s for s, a in q2 if a > 0.95])*100:.1f}% |")
    print(f"| Q4: Retrogression/Learning Ratio | {q4[1]/q4[0]:.2f} |")
