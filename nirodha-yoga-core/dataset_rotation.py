import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# 1. MODELS
# ============================================================================

class NirodhaLayer(nn.Module):
    def __init__(self, layer, beta=0.1):
        super().__init__()
        self.layer = layer
        self.beta = beta
        self.anchor = None
    def set_anchor(self, state_dict):
        self.anchor = {k: v.clone() for k, v in state_dict.items()}
    def nirodha_op(self, x):
        return x / (1 + self.beta * torch.abs(x))
    def forward(self, x, residual=False):
        w = self.layer.weight
        b = self.layer.bias
        if self.anchor:
            aw = self.anchor['layer.weight']
            w = aw + self.nirodha_op(w - aw)
            if self.layer.bias is not None:
                ab = self.anchor['layer.bias']
                b = ab + self.nirodha_op(b - ab)
        if isinstance(self.layer, nn.Conv2d):
            out = F.conv2d(x, w, b, self.layer.stride, self.layer.padding)
        else:
            out = F.linear(x, w, b)
        return x + F.relu(out) if residual else out

class RotationNet(nn.Module):
    def __init__(self, mode='baseline', initial_depth=2):
        super().__init__()
        self.mode = mode
        self.initial_depth = initial_depth
        
        # Feature Extractor
        if mode == 'nirodha':
            self.conv1 = NirodhaLayer(nn.Conv2d(3, 32, 3, padding=1))
            self.conv2 = NirodhaLayer(nn.Conv2d(32, 64, 3, padding=1))
            self.fc_layers = nn.ModuleList([NirodhaLayer(nn.Linear(64*8*8, 256)), NirodhaLayer(nn.Linear(256, 256))])
            self.heads = nn.ModuleList([NirodhaLayer(nn.Linear(256, 10)) for _ in range(5)])
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc_layers = nn.ModuleList([nn.Linear(64*8*8, 256), nn.Linear(256, 256)])
            self.heads = nn.ModuleList([nn.Linear(256, 10) for _ in range(5)])

    def set_anchor(self, beta=10000.0, task_id=0):
        if self.mode != 'nirodha': return
        # Immutable Anchors: only set if None
        if self.conv1.anchor is None: 
            self.conv1.set_anchor(self.conv1.state_dict()); self.conv1.beta = beta
        if self.conv2.anchor is None:
            self.conv2.set_anchor(self.conv2.state_dict()); self.conv2.beta = beta
            
        for layer in self.fc_layers:
            if layer.anchor is None:
                layer.set_anchor(layer.state_dict()); layer.beta = beta
                
        # Heads are task-specific, so we anchor them after training their specific task
        if self.heads[task_id].anchor is None:
            self.heads[task_id].set_anchor(self.heads[task_id].state_dict()); self.heads[task_id].beta = beta

    def add_layers(self, n=1):
        if self.mode == 'nirodha':
            self.fc_layers.append(NirodhaLayer(nn.Linear(256, 256), beta=0.1))
        else:
            self.fc_layers.append(nn.Linear(256, 256))

    def forward(self, x, task_id=0):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        
        # Short-Circuit: Task T only uses layers available at time T
        # Tasks are added sequentially with layers.
        # Task 0 (MNIST) uses fc_layers[0, 1]
        # Task 1 uses fc_layers[0, 1, 2]
        depth_for_task = self.initial_depth + task_id
        for i, layer in enumerate(self.fc_layers):
            if i >= depth_for_task: break
            is_res = (i >= self.initial_depth)
            if self.mode == 'nirodha':
                x = layer(x, residual=is_res)
                if not is_res: x = F.relu(x)
            else:
                x = F.relu(layer(x))
        return self.heads[task_id](x)

# ============================================================================
# 2. UTILS
# ============================================================================

def get_data(name, train=True):
    t_gray = transforms.Compose([transforms.Resize((32,32)), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    t_rgb = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    if name == 'MNIST': return DataLoader(datasets.MNIST('./data', train=train, download=True, transform=t_gray), batch_size=128, shuffle=True)
    if name == 'CIFAR10': return DataLoader(datasets.CIFAR10('./data', train=train, download=True, transform=t_rgb), batch_size=128, shuffle=True)
    if name == 'SVHN': return DataLoader(datasets.SVHN('./data', split=('train' if train else 'test'), download=True, transform=t_rgb), batch_size=128, shuffle=True)
    if name == 'USPS': return DataLoader(datasets.USPS('./data', train=train, download=True, transform=t_gray), batch_size=128, shuffle=True)

def train_one_epoch(model, loader, optimizer, device, task_id):
    model.train()
    for d, t in loader:
        d, t = d.to(device), t.to(device)
        optimizer.zero_grad(); loss = F.cross_entropy(model(d, task_id), t); loss.backward(); optimizer.step()

def eval_acc(model, loader, device, task_id):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            correct += (model(d, task_id).argmax(1) == t).sum().item(); total += t.size(0)
    return 100 * correct / total

# ============================================================================
# 3. EXPERIMENT
# ============================================================================

def run_rotation(mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = ['MNIST', 'CIFAR10', 'SVHN', 'USPS']
    model = RotationNet(mode=mode).to(device)
    
    task_histories = [] # Accuracies on all tasks seen so far after each task
    forgetting_list = [] # Forgetting relative to peak for each task
    
    peak_accs = [0]*len(tasks)
    
    for t, name in enumerate(tasks):
        print(f"Mode {mode} | Task {t+1}: {name}")
        train_loader = get_data(name, True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train Task t
        for _ in range(2): train_one_epoch(model, train_loader, optimizer, device, t)
        
        # Eval all tasks seen so far
        current_accs = []
        for prev_t in range(t + 1):
            acc = eval_acc(model, get_data(tasks[prev_t], False), device, prev_t)
            current_accs.append(acc)
            if acc > peak_accs[prev_t]: peak_accs[prev_t] = acc
        
        avg_acc = sum(current_accs) / len(current_accs)
        task_histories.append(avg_acc)
        
        if mode == 'nirodha' and t < len(tasks)-1:
            model.set_anchor(beta=10000.0, task_id=t)
            model.add_layers(1); model.to(device)

    # Calculate final forgetting for each task
    final_accs = []
    for t_idx in range(len(tasks)):
        acc = eval_acc(model, get_data(tasks[t_idx], False), device, t_idx)
        forgetting_list.append(peak_accs[t_idx] - acc)
        
    return forgetting_list, task_histories

if __name__ == "__main__":
    bf, bt = run_rotation('baseline')
    nf, nt = run_rotation('nirodha')
    
    tasks = ['MNIST', 'CIFAR-10', 'SVHN', 'USPS']
    print("\nRESULTS:")
    print(f"tasks = {tasks}")
    print(f"baseline_forgetting = {bf}")
    print(f"nirodha_forgetting = {nf}")
    print(f"baseline_trajectory = {bt}")
    print(f"nirodha_trajectory = {nt}")
    
    with open('rotation_metrics.txt', 'w') as f:
        f.write(f"tasks = {tasks}\n")
        f.write(f"baseline_forgetting = {bf}\n")
        f.write(f"nirodha_forgetting = {nf}\n")
        f.write(f"baseline_trajectory = {bt}\n")
        f.write(f"nirodha_trajectory = {nt}\n")
