import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist_proof import NirodhaLayer, SimpleNet, ProgressiveDepthNet, get_split_mnist, train_epoch, evaluate
import matplotlib.pyplot as plt
import numpy as np

def run_ablation_beta_sweep(betas=[0.1, 1.0, 10.0, 100.0, 1000.0]):
    """Vary Beta and measure Task A retention vs Task B accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Beta Sweep on {device}...")
    
    task_a_train = get_split_mnist(train=True, digits_subset=[0,1,2,3,4])
    task_a_test = get_split_mnist(train=False, digits_subset=[0,1,2,3,4])
    task_b_train = get_split_mnist(train=True, digits_subset=[5,6,7,8,9])
    task_b_test = get_split_mnist(train=False, digits_subset=[5,6,7,8,9])
    
    retention_a = []
    accuracy_b = []
    
    for beta in betas:
        print(f"Testing Beta={beta}")
        model = ProgressiveDepthNet(hidden_dim=128, initial_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train Task A
        for _ in range(3): train_epoch(model, task_a_train, optimizer, device, task='a')
        
        # Anchor
        model.set_anchor(beta=beta)
        model.add_layers(n=2)
        model.to(device)
        
        # Train Task B
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        for _ in range(5): train_epoch(model, task_b_train, optimizer, device, task='b')
        
        # Eval
        a_acc = evaluate(model, task_a_test, device, task='a')
        b_acc = evaluate(model, task_b_test, device, task='b')
        retention_a.append(a_acc)
        accuracy_b.append(b_acc)
        print(f"  Result: A={a_acc:.2f}%, B={b_acc:.2f}%")
        
    return betas, retention_a, accuracy_b

def plot_beta_sweep(betas, a, b):
    plt.figure(figsize=(8, 5))
    plt.semilogx(betas, a, marker='o', label='Task A Retention')
    plt.semilogx(betas, b, marker='s', label='Task B Accuracy')
    plt.xlabel('Beta (Anchoring Strength)')
    plt.ylabel('Accuracy (%)')
    plt.title('Stability-Plasticity Trade-off (Beta Sweep)')
    plt.legend()
    plt.grid(True)
    plt.savefig('nirodha_beta_sweep.png')
    plt.close()

if __name__ == "__main__":
    b, ret, acc = run_ablation_beta_sweep()
    plot_beta_sweep(b, ret, acc)
