# -*- coding: utf-8 -*-
"""
Main Experiment: Ricci-REWA Self-Healing Protocol

Demonstrates that neural encoders trained with contrastive loss
exhibit self-healing geometric properties when perturbed.
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from config import CONFIG
from data_generation import generate_data
from model import MLPEncoder, contrastive_loss
from geometry import compute_geometry_snapshot, compute_metric_deviation
from visualization import plot_self_healing_dynamics, plot_phase_comparison


def create_augmented_batch(data, batch_size):
    """
    Create a batch of anchor-positive pairs for contrastive learning
    Simple augmentation: add small Gaussian noise
    
    Args:
        data: full dataset [N, DIM_INPUT]
        batch_size: int
    
    Returns:
        anchors, positives: [batch_size, DIM_INPUT]
    """
    indices = torch.randint(0, len(data), (batch_size,))
    anchors = data[indices]
    
    # Simple augmentation: add noise
    noise_scale = 0.1
    positives = anchors + torch.randn_like(anchors) * noise_scale
    
    return anchors, positives


def train_epoch(encoder, optimizer, data, batch_size):
    """
    Train for one epoch
    
    Returns:
        avg_loss: average loss for the epoch
    """
    encoder.train()
    total_loss = 0
    n_batches = len(data) // batch_size
    
    for _ in range(n_batches):
        anchors, positives = create_augmented_batch(data, batch_size)
        
        # Forward pass
        anchor_embeds = encoder(anchors)
        positive_embeds = encoder(positives)
        
        # Compute loss
        loss = contrastive_loss(anchor_embeds, positive_embeds)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / n_batches


def run_self_healing_experiment():
    """
    Main experiment with 4 phases:
    1. Genesis: Train to convergence
    2. Injury: Inject geometric defect
    3. Healing: Resume training and monitor recovery
    4. Visualization: Plot results
    """
    
    print("=" * 80)
    print("RICCI-REWA SELF-HEALING EXPERIMENT")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    
    # --- PHASE 1: GENESIS (Reach Equilibrium) ---
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
        
        if (epoch + 1) % 20 == 0:
            tqdm.write(f"  Epoch {epoch+1}/{CONFIG['GENESIS_EPOCHS']}, Loss: {avg_loss:.4f}")
    
    # Capture the "Healthy" Geometry
    state_0 = compute_geometry_snapshot(encoder, data)
    metric_G0 = state_0["G"]
    entropy_genesis = state_0["entropy"]
    
    print(f"\n[OK] Genesis Complete!")
    print(f"  Healthy Curvature Entropy: {entropy_genesis:.4f}")
    
    # --- PHASE 2: INJURY (Inject Geometric Defect) ---
    print("\n" + "=" * 80)
    print("PHASE 2: INJURY - Injecting Geometric Defect")
    print("=" * 80)
    
    # Perturb weights to simulate "Manifold Tearing"
    with torch.no_grad():
        for param in encoder.parameters():
            noise = torch.randn_like(param) * CONFIG["PERTURBATION_SCALE"]
            param.add_(noise)
    
    # Capture "Damaged" Geometry
    state_damaged = compute_geometry_snapshot(encoder, data)
    diff_damage = compute_metric_deviation(state_damaged["G"], metric_G0)
    entropy_damaged = state_damaged["entropy"]
    
    print(f"[OK] Damage Injected!")
    print(f"  Metric Deviation: {diff_damage:.4f}")
    print(f"  Damaged Curvature Entropy: {entropy_damaged:.4f}")
    print(f"  Entropy Change: {entropy_damaged - entropy_genesis:+.4f}")
    
    # --- PHASE 3: HEALING (Ricci Flow) ---
    print("\n" + "=" * 80)
    print("PHASE 3: HEALING - Ricci Flow Recovery")
    print("=" * 80)
    
    # Reset optimizer (fresh momentum)
    optimizer = optim.Adam(encoder.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    history_curvature = []
    history_recovery = []
    
    print(f"[OK] Resuming training for {CONFIG['HEALING_STEPS']} steps...")
    
    pbar = tqdm(range(CONFIG["HEALING_STEPS"]), desc="Healing")
    for step in pbar:
        # Standard training step
        anchors, positives = create_augmented_batch(data, CONFIG["BATCH_SIZE"])
        
        encoder.train()
        anchor_embeds = encoder(anchors)
        positive_embeds = encoder(positives)
        loss = contrastive_loss(anchor_embeds, positive_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Monitor the Geometry Flow
        if step % CONFIG["SNAPSHOT_INTERVAL"] == 0:
            current_state = compute_geometry_snapshot(encoder, data)
            
            # 1. How close is current metric G_t to original healthy G_0?
            recovery_score = compute_metric_deviation(current_state["G"], metric_G0)
            
            # 2. Monitor Curvature (Entropy)
            current_entropy = current_state["entropy"]
            
            history_curvature.append(current_entropy)
            history_recovery.append(recovery_score)
            
            pbar.set_postfix({
                'Recovery': f'{recovery_score:.2f}',
                'Entropy': f'{current_entropy:.3f}'
            })
    
    # Final state
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
    
    plot_self_healing_dynamics(history_recovery, history_curvature)
    plot_phase_comparison(entropy_genesis, entropy_damaged, entropy_healed)
    
    # --- SUMMARY ---
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Initial Damage:     {diff_damage:.4f}")
    print(f"Final Recovery:     {final_recovery:.4f}")
    print(f"Recovery Rate:      {(1 - final_recovery/diff_damage)*100:.1f}%")
    print(f"\nEntropy Genesis:    {entropy_genesis:.4f}")
    print(f"Entropy Damaged:    {entropy_damaged:.4f} ({entropy_damaged - entropy_genesis:+.4f})")
    print(f"Entropy Healed:     {entropy_healed:.4f} ({entropy_healed - entropy_genesis:+.4f})")
    print("\n" + "=" * 80)
    
    # Success criteria check
    print("\nSUCCESS CRITERIA:")
    recovery_success = final_recovery < diff_damage * 0.3  # Recovered >70%
    entropy_success = abs(entropy_healed - entropy_genesis) < abs(entropy_damaged - entropy_genesis)
    
    print(f"[OK] Recovery Score Decreased: {recovery_success}")
    print(f"[OK] Curvature Entropy Stabilized: {entropy_success}")
    
    if recovery_success and entropy_success:
        print("\n*** EXPERIMENT SUCCESSFUL! Self-healing dynamics confirmed.")
    else:
        print("\n*** Results inconclusive. May need parameter tuning.")
    
    return {
        'history_recovery': history_recovery,
        'history_curvature': history_curvature,
        'entropy_genesis': entropy_genesis,
        'entropy_damaged': entropy_damaged,
        'entropy_healed': entropy_healed,
        'initial_damage': diff_damage,
        'final_recovery': final_recovery
    }


if __name__ == "__main__":
    results = run_self_healing_experiment()
