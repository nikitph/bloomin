import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from nsfo_experiment.optimizer import NavierSchrodinger
from nsfo_experiment.landscape import MultiWell

def run_experiment():
    # Setup
    T_total = 2000 # Boosted for 20D
    T_explore = 1000 # Boosted explore
    T_anchor = 1500 # Boosted settling
    dim = 20
    
    landscape = MultiWell()
    # Start at [3, 3, 3, ...] to be far from origin
    # [3.0, 3.1] pattern repeated to break symmetry
    start_pos = torch.tensor([3.0, 3.1] * (dim // 2))
    global_min_val = -2.0 * dim # -40.0
    
    # Configurations
    configs = [
        {"name": "SGD", "optim": optim.SGD, "kwargs": {"lr": 0.05, "momentum": 0.0}, "phased": False},
        {"name": "Adam", "optim": optim.Adam, "kwargs": {"lr": 0.05}, "phased": False},
        {"name": "Scaled-NSFO", "optim": NavierSchrodinger, 
         "kwargs": {"dt": 0.05, "nu_max": 2.0, "hbar_max": 8.0, "use_convection": False, 
                    "k_anchor": 0.2, "gamma_normal": 5.0, 
                    "use_manifold_constraint": True, "epsilon_support": 1e-2}, 
         "phased": True},
    ]

    print(f"Running Experiment in {dim} Dimensions")
    print(f"Global Minimum Value: {global_min_val}")
    print(f"{'Optimizer':<20} | {'Final Loss':<10} | {'Min Loss':<10} | {'Dist to Global':<10}")
    print("-" * 60)

    for config in configs:
        params = torch.nn.Parameter(start_pos.clone())
        optimizer = config['optim']([params], **config['kwargs'])
        
        trajectory = []
        losses = []
        best_loss_A = float('inf')
        
        # Capture best state
        best_state = None
        
        for step in range(T_total):
            optimizer.zero_grad()
            loss = landscape(params)
            
            if torch.isnan(loss) or loss.item() > 1e4:
                losses.append(np.nan)
                break
                
            loss.backward()
            
            # Phase control
            if config.get('phased', False):
                # Quantum Annealing Schedule
                if step >= T_explore // 2 and step < T_explore:
                    progress = (step - T_explore // 2) / (T_explore - T_explore // 2)
                    hbar_new = 8.0 * (1 - progress) + 0.1 * progress
                    for param_group in optimizer.param_groups:
                        param_group['hbar_max'] = hbar_new

                # Track best A
                if step < T_explore:
                     current_loss = loss.item()
                     if current_loss < best_loss_A:
                        best_loss_A = current_loss
                        config['best_params'] = params.detach().clone()

                # Phase Transitions
                if step == T_explore:
                    print(f"[{config['name']}] End Exploration. Best Loss A: {best_loss_A:.4f}. Switching to Exploitation (Settling).")
                    if 'best_params' in config:
                        params.data.copy_(config['best_params'])
                    # Disable viscosity for settling
                    for param_group in optimizer.param_groups:
                        param_group['nu_max'] = 0.0
                
                if step == T_anchor:
                    print(f"[{config['name']}] Settling complete. Loss: {loss.item():.4f}. Anchoring and Locking.")
                    # Re-enable viscosity for stability
                    for param_group in optimizer.param_groups:
                        param_group['nu_max'] = 2.0
                    optimizer.capture_anchor()

                phase = 'exploration' if step < T_explore else 'exploitation'
                optimizer.step(phase=phase)
            else:
                optimizer.step()
            
            losses.append(loss.item())
        
        if len(losses) > 0 and not np.isnan(losses[-1]):
            final_loss = losses[-1]
            min_loss = min(losses)
            dist_to_global = params.norm().item()
        else:
            final_loss = float('nan')
            min_loss = min(losses) if losses else float('nan')
            dist_to_global = float('nan')
        
        print(f"{config['name']:<20} | {final_loss:.4f}     | {min_loss:.4f}     | {dist_to_global:.4f}")

if __name__ == "__main__":
    run_experiment()
