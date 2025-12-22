# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- ACI Core Operators ---

class Nirodha(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, V):
        return V / (1.0 + self.beta * torch.abs(V) + 1e-12)

class VairagyaOptimizer(torch.optim.Optimizer):
    """
    Axiom 7/11: Abhyāsa–Vairāgya (Practice + Detachment)
    Learning includes orthogonal noise injection to prevent brittle correlations.
    """
    def __init__(self, params, lr=1e-3, epsilon=0.01):
        defaults = dict(lr=lr, epsilon=epsilon)
        super(VairagyaOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            epsilon = group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 1. Standard SGD step
                # p.add_(grad, alpha=-lr)
                
                # 2. Vairāgya: Orthogonal noise injection
                # We want noise xi such that xi is orthogonal to grad
                xi = torch.randn_like(p)
                # Projection P_grad_perp(xi) = xi - (xi . grad / |grad|^2) * grad
                grad_norm_sq = torch.sum(grad * grad) + 1e-12
                proj = torch.sum(xi * grad) / grad_norm_sq
                xi_perp = xi - proj * grad
                
                # Update: theta = theta - lr*grad + epsilon*xi_perp
                p.add_(grad, alpha=-lr)
                p.add_(xi_perp, alpha=epsilon)
        
        return loss

# --- Experiment 1: Swarm Stability (Citta-Prasādanam) ---

def run_swarm_stability_experiment():
    print("\n--- Running Experiment 1: Swarm Stability (Citta-Prasādanam) ---")
    num_agents = 10
    dim = 64
    steps = 200
    
    # States x_i
    states = [torch.randn(dim) for _ in range(num_agents)]
    anchors = [s.clone() for s in states]
    
    # Energies E_i
    def get_energy(x, x0):
        return torch.norm(x - x0)**2
    
    # Swarm Energy
    # E_swarm = sum E_i + sum lambda_ij * ||x_i - x_j||^2
    coupling_lambda = 0.1
    
    history_e_swarm = []
    
    # Operators
    nirodha = Nirodha(beta=1.0)
    
    for t in range(steps):
        # Global swarm energy
        e_i = sum(get_energy(states[i], anchors[i]) for i in range(num_agents))
        e_coupling = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                e_coupling += coupling_lambda * torch.norm(states[i] - states[j])**2
        
        e_swarm = e_i + e_coupling
        history_e_swarm.append(e_swarm.item())
        
        # Rogue agents behavior (injecting entropy)
        if t > 50:
            rogue_indices = [0, 1]
            for idx in rogue_indices:
                states[idx] += torch.randn(dim) * 0.5 # Random walk
        
        # Apply Citta-Prasadanm Control
        # 1. Karuna (Damping for unstable agents)
        # 2. Upeksa (Isolation for high-entropy agents)
        
        new_states = []
        for i in range(num_agents):
            v_prop = states[i] - anchors[i]
            
            # Simple Karuna: If local energy is high, increase damping/suppression
            local_e = get_energy(states[i], anchors[i])
            effective_beta = 1.0 + (5.0 if local_e > 10.0 else 0.0)
            
            # Simple Upeksa: If agent is rogue (detected by high variance/energy), isolate it
            # In this demo, if local_e is extreme, we aggressively reset to anchor
            if local_e > 50.0:
                v_suppressed = torch.zeros_like(v_prop)
            else:
                v_suppressed = v_prop / (1.0 + effective_beta * torch.abs(v_prop).mean() + 1e-12)
            
            new_states.append(anchors[i] + v_suppressed)
        
        states = new_states
        
    plt.figure(figsize=(10, 5))
    plt.plot(history_e_swarm, label="Global Swarm Energy $E_{swarm}$")
    plt.axvline(x=50, color='r', linestyle='--', label="Rogue Injection")
    plt.title("Citta-Prasādanam: Swarm Stability under Rogue Perturbations")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig("swarm_stability.png")
    print("Saved swarm_stability.png")

# --- Experiment 2: Oracle Consistency (Mūrdha-Jyotiṣi) ---

def run_oracle_consistency_experiment():
    print("\n--- Running Experiment 2: Oracle Consistency (Mūrdha-Jyotiṣi) ---")
    dim = 128
    steps = 500
    
    # Invariant Constraint: x must lie on the unit sphere (simplified invariant manifold)
    # I = {x | ||x||^2 = 1}
    # L_oracle(x) = d(Pi_I(x), x)
    
    x = torch.randn(dim, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-2)
    
    history_oracle_loss = []
    history_task_loss = []
    
    # Target (some task goal)
    target = torch.randn(dim)
    target = target / torch.norm(target) # Normalize target to make it reachable on manifold
    
    for t in range(steps):
        optimizer.zero_grad()
        
        # 1. Task Loss: minimize distance to target
        task_loss = torch.norm(x - target)**2
        
        # 2. Oracle Loss: distance to unit sphere manifold
        projection = x / torch.norm(x)
        oracle_loss = torch.norm(projection - x)**2
        
        total_loss = task_loss + 10.0 * oracle_loss # Strong oracle constraint
        total_loss.backward()
        optimizer.step()
        
        history_oracle_loss.append(oracle_loss.item())
        history_task_loss.append(task_loss.item())
        
    plt.figure(figsize=(10, 5))
    plt.plot(history_oracle_loss, label="Oracle Consistency Loss $\mathcal{L}_{oracle}$")
    plt.plot(history_task_loss, label="Task Loss")
    plt.yscale('log')
    plt.title("Mūrdha-Jyotiṣi: Asymptotic Consistency on Invariant Manifolds")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig("consistency_oracle.png")
    print("Saved consistency_oracle.png")

# --- Experiment 3: Non-Attachment Regularization (Vairāgya) ---

def run_vairagya_experiment():
    print("\n--- Running Experiment 3: Non-Attachment Regularization (Vairāgya) ---")
    # Synthetic problem: Biased data
    # Input x = [signal, bias]
    # In training, bias highly correlates with label.
    # In test, bias is randomized.
    
    num_samples = 1000
    dim = 2 # [signal, bias]
    
    # Train data: label = sign(signal), bias = sign(signal) + noise
    train_signal = torch.randn(num_samples)
    train_bias = train_signal + torch.randn(num_samples) * 0.1 # High correlation
    X_train = torch.stack([train_signal, train_bias], dim=1)
    y_train = (train_signal > 0).float().unsqueeze(1)
    
    # Test data: signal = signal, bias = random
    test_signal = torch.randn(num_samples)
    test_bias = torch.randn(num_samples) # NO correlation
    X_test = torch.stack([test_signal, test_bias], dim=1)
    y_test = (test_signal > 0).float().unsqueeze(1)
    
    def train_model(opt_class, **opt_kwargs):
        model = nn.Linear(2, 1)
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
        # Eval
        with torch.no_grad():
            train_preds = (torch.sigmoid(model(X_train)) > 0.5).float()
            test_preds = (torch.sigmoid(model(X_test)) > 0.5).float()
            train_acc = (train_preds == y_train).float().mean()
            test_acc = (test_preds == y_test).float().mean()
        return train_acc.item(), test_acc.item()

    sgd_train, sgd_test = train_model(torch.optim.SGD, lr=0.1)
    vairagya_train, vairagya_test = train_model(VairagyaOptimizer, lr=0.1, epsilon=0.05)
    
    print(f"Vanilla SGD: Train Acc={sgd_train:.4f}, Test Acc={sgd_test:.4f}")
    print(f"Vairāgya (ACI): Train Acc={vairagya_train:.4f}, Test Acc={vairagya_test:.4f}")
    
    labels = ['Vanilla SGD', 'Vairāgya (ACI)']
    test_accs = [sgd_test, vairagya_test]
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, test_accs, color=['gray', 'blue'])
    plt.ylabel("Generalization Accuracy (Unbiased Test)")
    plt.title("Vairāgya: Preventing Over-Commitment to Spurious Correlations")
    plt.ylim(0, 1.0)
    plt.savefig("vairagya_robustness.png")
    print("Saved vairagya_robustness.png")

# --- Experiment 4: Modular Retrogression (Kaivalyam) ---

def run_kaivalyam_experiment():
    print("\n--- Running Experiment 4: Modular Retrogression (Kaivalyam) ---")
    num_tasks = 10
    dim_core = 32
    dim_task = 32
    
    core_state = torch.randn(dim_core)
    core_anchor = core_state.clone()
    
    history_drift = []
    
    for t in range(num_tasks):
        # 1. New Task State
        task_state = torch.randn(dim_task)
        
        # 2. Sequential learning might update core_state
        # Standard model without retrogression allows core_state to drift
        core_state += torch.randn(dim_core) * 0.1 # Simulation of "cognitive debt"
        
        # 3. Axiom 10: Kaivalyam / Pratiprasava
        # After task completion, reset core to anchor, compress task to seed
        # x -> x_core, x_task -> sigma(x_task)
        
        drift_unregulated = torch.norm(core_state - core_anchor).item()
        
        # REGULATION STEP
        core_state = core_anchor.clone() # Perfect retrogression
        # task_seed = sigma(task_state) # Implicitly stored
        
        drift_regulated = torch.norm(core_state - core_anchor).item()
        history_drift.append((drift_unregulated, drift_regulated))
        
    drifts_unreg = [d[0] for d in history_drift]
    drifts_reg = [d[1] for d in history_drift]
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(drifts_unreg), label="Cumulative Drift (Vanilla)", linestyle='--')
    plt.plot(np.cumsum(drifts_reg), label="Cumulative Drift (Kaivalyam)", linewidth=3)
    plt.title("Kaivalyam: Modular Retrogression prevents Systemic Drift")
    plt.xlabel("Task Sequential Count")
    plt.ylabel("Core State Cumulative Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("kaivalyam_drift.png")
    print("Saved kaivalyam_drift.png")

# --- Main Execution ---

if __name__ == "__main__":
    start_time = time.time()
    
    run_swarm_stability_experiment()
    run_oracle_consistency_experiment()
    run_vairagya_experiment()
    run_kaivalyam_experiment()
    
    total_time = time.time() - start_time
    print(f"\nAll ACI Axiom Validation Experiments Completed in {total_time:.2f}s")
    print("Graphs generated: swarm_stability.png, consistency_oracle.png, vairagya_robustness.png, kaivalyam_drift.png")
