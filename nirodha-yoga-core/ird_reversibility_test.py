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

# --- ACI Components ---

class NirodhaOperator(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, V):
        return V / (1.0 + self.beta * torch.abs(V) + 1e-12)

class NirodhaLinear(nn.Module):
    """
    Axiom 12: Kaivalyam (Modular Retrogression) enabled Linear Layer.
    Maintains a Core State (Anchor) and a task-specific Fluctuation (V).
    """
    def __init__(self, in_features, out_features, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Core Parameters (Anchor)
        self.weight_core = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_core = nn.Parameter(torch.Tensor(out_features))
        
        # Task Fluctuations (V)
        self.weight_v = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_v = nn.Parameter(torch.zeros(out_features))
        
        self.nirodha = NirodhaOperator(beta)
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight_core, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_core)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_core, -bound, bound)

    def forward(self, x):
        # Axiom 1: Nirodha suppression of fluctuations
        w_eff = self.weight_core + self.nirodha(self.weight_v)
        b_eff = self.bias_core + self.nirodha(self.bias_v)
        return F.linear(x, w_eff, b_eff)

    def commit_to_core(self):
        """Phase 1 -> 2 transition: Task A results become the new Core Anchor."""
        with torch.no_grad():
            self.weight_core.copy_(self.weight_core + self.nirodha(self.weight_v))
            self.bias_core.copy_(self.bias_core + self.nirodha(self.bias_v))
            self.weight_v.zero_()
            self.bias_v.zero_()

    def retrogress(self):
        """Axiom 10: Kaivalyam / Pratiprasava. Discard Task B fluctuations."""
        with torch.no_grad():
            self.weight_v.zero_()
            self.bias_v.zero_()

# --- Models ---

class StandardMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ACIMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10, beta=100.0):
        super().__init__()
        self.fc1 = NirodhaLinear(input_dim, hidden_dim, beta=beta)
        self.fc2 = NirodhaLinear(hidden_dim, output_dim, beta=beta)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def commit_to_core(self):
        self.fc1.commit_to_core()
        self.fc2.commit_to_core()

    def retrogress(self):
        self.fc1.retrogress()
        self.fc2.retrogress()

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

# --- Experiment Execution ---

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(loader.dataset)

def run_ird_test():
    print("--- Starting IRD (Irreversible -> Reversible Depth) Test ---")
    
    # Configuration
    batch_size = 64
    epochs_a = 5
    epochs_b = 5
    
    # Task A: Digits 0-4
    # Task B: Digits 5-9
    loader_a_train, loader_a_test = get_mnist_subset(range(5), batch_size)
    loader_b_train, loader_b_test = get_mnist_subset(range(5, 10), batch_size)
    
    # Initialize Models
    vanilla = StandardMLP()
    aci = ACIMLP(beta=10.0) # Moderate beta for balance
    
    opt_v = optim.Adam(vanilla.parameters(), lr=1e-3)
    opt_a = optim.Adam(aci.parameters(), lr=1e-3)
    
    # --- PHASE 1: Learn Task A ---
    print("\nPhase 1: Learning Task A (Digits 0-4)")
    for epoch in range(1, epochs_a + 1):
        loss_v = train_one_epoch(vanilla, loader_a_train, opt_v, epoch)
        loss_a = train_one_epoch(aci, loader_a_train, opt_a, epoch)
        acc_v = evaluate(vanilla, loader_a_test)
        acc_a = evaluate(aci, loader_a_test)
        print(f"  Epoch {epoch}: Vanilla Acc={acc_v:.4f}, ACI Acc={acc_a:.4f}")
    
    acc_a1_vanilla = evaluate(vanilla, loader_a_test)
    acc_a1_aci = evaluate(aci, loader_a_test)
    
    # Record snapshots
    theta_a_vanilla = copy.deepcopy(vanilla.state_dict())
    theta_a_aci = copy.deepcopy(aci.state_dict())
    
    # Commit Task A to Core in ACI
    aci.commit_to_core()
    
    # --- PHASE 2: Learn Task B (Adversarial) ---
    print("\nPhase 2: Learning Task B (Digits 5-9)")
    
    # PROTECT THE CORE: In ACI, we only learn the fluctuation for Task B.
    # Standard models have no "Core" concept, so they just update everything.
    params_a_phase2 = [
        {'params': [p for n, p in aci.named_parameters() if 'weight_v' in n or 'bias_v' in n], 'lr': 1e-3}
        # weight_core and bias_core are NOT included here -> frozen
    ]
    opt_a_task_b = optim.Adam(params_a_phase2)
    
    for epoch in range(1, epochs_b + 1):
        loss_v = train_one_epoch(vanilla, loader_b_train, opt_v, epoch)
        loss_a = train_one_epoch(aci, loader_b_train, opt_a_task_b, epoch)
        acc_b_v = evaluate(vanilla, loader_b_test)
        acc_b_a = evaluate(aci, loader_b_test)
        acc_a_v = evaluate(vanilla, loader_a_test) # Forgetfulness check
        acc_a_a = evaluate(aci, loader_a_test)
        print(f"  Epoch {epoch}: Task B Acc (V/A): {acc_b_v:.4f}/{acc_b_a:.4f} | Task A Acc (V/A): {acc_a_v:.4f}/{acc_a_a:.4f}")

    # --- PHASE 3: Reversion Test (The Kill Shot) ---
    print("\nPhase 3: Kaivalyam Reversion")
    
    # Vanilla: No inherent reversion. Just measure current state.
    acc_rev_vanilla = evaluate(vanilla, loader_a_test)
    
    # ACI: Apply Retrogression
    aci.retrogress()
    acc_rev_aci = evaluate(aci, loader_a_test)
    
    print("\n--- Final Metrics ---")
    print(f"Functional Reversibility (Task A Accuracy):")
    print(f"  Vanilla: {acc_a1_vanilla:.4f} -> {acc_rev_vanilla:.4f} (Forgetting: {acc_a1_vanilla - acc_rev_vanilla:.4f})")
    print(f"  ACI:     {acc_a1_aci:.4f} -> {acc_rev_aci:.4f} (Forgetting: {acc_a1_aci - acc_rev_aci:.4f})")
    
    # State-Space Reversibility
    # Check if aci weights are identical to theta_a_aci after commit+retrogress
    aci_params = aci.state_dict()
    sims = []
    for k in aci_params:
        if 'weight_core' in k or 'bias_core' in k:
            p1 = theta_a_aci[k.replace('weight_core', 'weight_core').replace('bias_core', 'bias_core')] # committed
            p2 = aci_params[k]
            # Since we committed Task A and then retrogressed Task B, ACI should be EXACTLY at State A.
            cos_sim = F.cosine_similarity(p1.view(-1), p2.view(-1), dim=0).item()
            sims.append(cos_sim)
    
    avg_sim_aci = np.mean(sims)
    
    # Vanilla state similarity (how far did it move?)
    sims_v = []
    v_params = vanilla.state_dict()
    for k in v_params:
        p1 = theta_a_vanilla[k]
        p2 = v_params[k]
        cos_sim = F.cosine_similarity(p1.view(-1), p2.view(-1), dim=0).item()
        sims_v.append(cos_sim)
    avg_sim_vanilla = np.mean(sims_v)
    
    print(f"\nState-Space Reversibility (Cosine Similarity to State A):")
    print(f"  Vanilla: {avg_sim_vanilla:.6f}")
    print(f"  ACI:     {avg_sim_aci:.6f}")
    
    # Visualization
    methods = ['Vanilla', 'ACI']
    forgetting = [acc_a1_vanilla - acc_rev_vanilla, acc_a1_aci - acc_rev_aci]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, [acc_rev_vanilla, acc_rev_aci], color=['red', 'blue'])
    plt.axhline(y=acc_a1_aci, color='black', linestyle='--', label='Initial Task A Acc')
    plt.title("Functional Reversibility (Task A Recovery)")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, [avg_sim_vanilla, avg_sim_aci], color=['red', 'blue'])
    plt.title("State-Space Reversibility (Topology)")
    plt.ylabel("Cosine Similarity to $\theta_A$")
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('ird_reversibility_results.png')
    print("\nResults saved to ird_reversibility_results.png")

if __name__ == "__main__":
    run_ird_test()
