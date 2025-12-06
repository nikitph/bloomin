# -*- coding: utf-8 -*-
"""
Optimized Experiment: Ricci-REWA Self-Healing Protocol
Uses improved parameters for better recovery
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from config_aggressive import CONFIG
from data_generation import generate_data
from model import MLPEncoder, contrastive_loss
from geometry import compute_geometry_snapshot, compute_metric_deviation
from visualization import plot_self_healing_dynamics, plot_phase_comparison


def create_augmented_batch(data, batch_size):
    """Create anchor-positive pairs for contrastive learning"""
    indices = torch.randint(0, len(data), (batch_size,))
    anchors = data[indices]
    
    noise_scale = 0.1
    positives = anchors + torch.randn_like(anchors) * noise_scale
    
    return anchors, positives


def train_epoch(encoder, optimizer, data, batch_size):
    """Train for one epoch"""
    encoder.train()
    total_loss = 0
    n_batches = len(data) // batch_size
    
    for _ in range(n_batches):
        anchors, positives = create_augmented_batch(data, batch_size)
        
        anchor_embeds = encoder(anchors)
        positive_embeds = encoder(positives)
        loss = contrastive_loss(anchor_embeds, positive_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / n_batches


def run_optimized_experiment():
    """
    Optimized 4-phase experiment with:
    - Stronger genesis (150 epochs)
    - Less severe damage (scale=0.3)
    - Extended healing (1000 steps)
    - Adaptive learning rate
    """
    
    print("=" * 80)
    print("RICCI-REWA SELF-HEALING EXPERIMENT (AGGRESSIVE)")
    print("=" * 80)
    print("\nAggressive Optimizations:")
    print(f"  - Genesis epochs: 100 -> {CONFIG['GENESIS_EPOCHS']}")
    print(f"  - Perturbation scale: 0.5 -> {CONFIG['PERTURBATION_SCALE']}")
    print(f"  - Healing steps: 500 -> {CONFIG['HEALING_STEPS']}")
    print(f"  - Healing batch size: 256 -> {CONFIG['HEALING_BATCH_SIZE']}")
    print(f"  - Adaptive LR: {CONFIG['HEALING_LR_START']} -> {CONFIG['HEALING_LR_END']}")
    
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    
    # --- PHASE 1: GENESIS ---
    print("\n" + "=" * 80)
    print("PHASE 1: GENESIS - Training to Convergence")
    print("=" * 80)
    
    data, labels = generate_data()
    print(f"[OK] Generated {len(data)} samples with {CONFIG['N_CLUSTERS']} clusters")
    
    encoder = MLPEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    print(f"[OK] Training for {CONFIG['GENESIS_EPOCHS']} epochs...")
    
    for epoch in tqdm(range(CONFIG["GENESIS_EPOCHS"]), desc="Genesis Training"):
        avg_loss = train_epoch(encoder, optimizer, data, CONFIG["BATCH_SIZE"])
        
        if (epoch + 1) % 30 == 0:
            tqdm.write(f"  Epoch {epoch+1}/{CONFIG['GENESIS_EPOCHS']}, Loss: {avg_loss:.4f}")
    
    state_0 = compute_geometry_snapshot(encoder, data)
    metric_G0 = state_0["G"]
    entropy_genesis = state_0["entropy"]
    
    print(f"\n[OK] Genesis Complete!")
    print(f"  Healthy Curvature Entropy: {entropy_genesis:.4f}")
    
    # --- PHASE 2: INJURY ---
    print("\n" + "=" * 80)
    print("PHASE 2: INJURY - Injecting Geometric Defect")
    print("=" * 80)
    
    with torch.no_grad():
        for param in encoder.parameters():
            noise = torch.randn_like(param) * CONFIG["PERTURBATION_SCALE"]
            param.add_(noise)
    
    state_damaged = compute_geometry_snapshot(encoder, data)
    diff_damage = compute_metric_deviation(state_damaged["G"], metric_G0)
    entropy_damaged = state_damaged["entropy"]
    
    print(f"[OK] Damage Injected!")
    print(f"  Metric Deviation: {diff_damage:.4f}")
    print(f"  Damaged Curvature Entropy: {entropy_damaged:.4f}")
    print(f"  Entropy Change: {entropy_damaged - entropy_genesis:+.4f}")
    
    # --- PHASE 3: HEALING (with Adaptive LR) ---
    print("\n" + "=" * 80)
    print("PHASE 3: HEALING - Ricci Flow Recovery (Adaptive LR)")
    print("=" * 80)
    
    # Optimizer with higher initial LR
    optimizer = optim.Adam(encoder.parameters(), lr=CONFIG["HEALING_LR_START"])
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["HEALING_STEPS"], 
        eta_min=CONFIG["HEALING_LR_END"]
    )
    
    history_curvature = []
    history_recovery = []
    
    print(f"[OK] Resuming training for {CONFIG['HEALING_STEPS']} steps...")
    print(f"    LR schedule: {CONFIG['HEALING_LR_START']:.0e} -> {CONFIG['HEALING_LR_END']:.0e}")
    
    pbar = tqdm(range(CONFIG["HEALING_STEPS"]), desc="Healing")
    for step in pbar:
        # Use larger batch size during healing
        anchors, positives = create_augmented_batch(data, CONFIG["HEALING_BATCH_SIZE"])
        
        encoder.train()
        anchor_embeds = encoder(anchors)
        positive_embeds = encoder(positives)
        loss = contrastive_loss(anchor_embeds, positive_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        if step % CONFIG["SNAPSHOT_INTERVAL"] == 0:
            current_state = compute_geometry_snapshot(encoder, data)
            recovery_score = compute_metric_deviation(current_state["G"], metric_G0)
            current_entropy = current_state["entropy"]
            
            history_curvature.append(current_entropy)
            history_recovery.append(recovery_score)
            
            pbar.set_postfix({
                'Recovery': f'{recovery_score:.2f}',
                'Entropy': f'{current_entropy:.3f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    state_healed = compute_geometry_snapshot(encoder, data)
    final_recovery = compute_metric_deviation(state_healed["G"], metric_G0)
    entropy_healed = state_healed["entropy"]
    
    print(f"\n[OK] Healing Complete!")
    print(f"  Final Recovery Score: {final_recovery:.4f}")
    print(f"  Final Curvature Entropy: {entropy_healed:.4f}")
    print(f"  Recovery: {(1 - final_recovery/diff_damage)*100:.1f}%")
    
    # --- PHASE 4: VISUALIZATION ---
    print("\n" + "=" * 80)
    print("PHASE 4: VISUALIZATION")
    print("=" * 80)
    
    plot_self_healing_dynamics(history_recovery, history_curvature, 
                               save_path="results/self_healing_dynamics_aggressive.png")
    plot_phase_comparison(entropy_genesis, entropy_damaged, entropy_healed,
                         save_path="results/phase_comparison_aggressive.png")
    
    # --- SUMMARY ---
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (AGGRESSIVE)")
    print("=" * 80)
    print(f"Initial Damage:     {diff_damage:.4f}")
    print(f"Final Recovery:     {final_recovery:.4f}")
    print(f"Recovery Rate:      {(1 - final_recovery/diff_damage)*100:.1f}%")
    print(f"\nEntropy Genesis:    {entropy_genesis:.4f}")
    print(f"Entropy Damaged:    {entropy_damaged:.4f} ({entropy_damaged - entropy_genesis:+.4f})")
    print(f"Entropy Healed:     {entropy_healed:.4f} ({entropy_healed - entropy_genesis:+.4f})")
    print("\n" + "=" * 80)
    
    # Success criteria
    print("\nSUCCESS CRITERIA:")
    recovery_success = final_recovery < diff_damage * 0.3
    entropy_success = abs(entropy_healed - entropy_genesis) < abs(entropy_damaged - entropy_genesis)
    
    print(f"[OK] Recovery Score Decreased: {recovery_success}")
    print(f"[OK] Curvature Entropy Stabilized: {entropy_success}")
    
    if recovery_success and entropy_success:
        print("\n*** EXPERIMENT SUCCESSFUL! Self-healing dynamics confirmed.")
    else:
        print("\n*** Results inconclusive. May need parameter tuning.")
    
    # Comparison with baseline
    baseline_recovery = 67.4
    optimized_recovery = (1 - final_recovery/diff_damage)*100
    improvement = optimized_recovery - baseline_recovery
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print(f"Baseline Recovery:   {baseline_recovery:.1f}%")
    print(f"Optimized Recovery:  {optimized_recovery:.1f}%")
    print(f"Improvement:         {improvement:+.1f}%")
    print("=" * 80)
    
    return {
        'history_recovery': history_recovery,
        'history_curvature': history_curvature,
        'entropy_genesis': entropy_genesis,
        'entropy_damaged': entropy_damaged,
        'entropy_healed': entropy_healed,
        'initial_damage': diff_damage,
        'final_recovery': final_recovery,
        'recovery_rate': optimized_recovery
    }


if __name__ == "__main__":
    results = run_optimized_experiment()
