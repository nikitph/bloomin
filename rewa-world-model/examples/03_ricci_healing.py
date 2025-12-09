"""
Example 3: Ricci-REWA Self-Healing

Demonstrates:
- Ricci flow evolution on geometric structure
- Perturbation injection
- Self-healing dynamics
- Recovery measurement
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data import HierarchicalGaussianGenerator
from geometry import FisherMetric
from ricci import (
    RicciFlowEvolution,
    EvolutionConfig,
    SelfHealingExperiment
)

def main():
    print("=== Ricci-REWA Self-Healing Experiment ===")
    print()
    
    # 1. Generate dataset
    print("1. Generating hierarchical dataset...")
    generator = HierarchicalGaussianGenerator(
        n_levels=2,
        branching_factor=3,
        dim=8  # Smaller dimension for faster computation
    )
    documents = generator.generate(docs_per_leaf=5)
    print(f"   Generated {len(documents)} documents")
    print()
    
    # 2. Create Fisher metrics
    print("2. Creating Fisher metrics...")
    embeddings = np.array([doc.embedding for doc in documents])
    doc_ids = [doc.id for doc in documents]
    
    # Simple metrics: identity + small perturbations
    metrics = []
    for i, emb in enumerate(embeddings):
        d = len(emb)
        g = np.eye(d)
        
        # Add structure based on cluster
        cluster = documents[i].metadata['cluster_id']
        cluster_hash = hash(cluster) % 100
        g += np.random.RandomState(cluster_hash).randn(d, d) * 0.05
        g = (g + g.T) / 2
        g += np.eye(d) * 0.01
        
        metrics.append(g)
    
    print(f"   Created {len(metrics)} Fisher metrics")
    print()
    
    # 3. Run baseline evolution (no perturbation)
    print("3. Running baseline Ricci flow...")
    config = EvolutionConfig(
        dt=0.01,
        num_steps=50,
        lambda_force=0.1,
        kappa_diffusion=0.05,
        epsilon_noise=0.001,
        checkpoint_interval=10
    )
    
    evolution = RicciFlowEvolution(config)
    baseline_history = evolution.evolve(metrics, doc_ids)
    
    print()
    print(f"Baseline evolution complete")
    print(f"  Initial Ricci norm: {baseline_history[0].ricci_norm:.6f}")
    print(f"  Final Ricci norm: {baseline_history[-1].ricci_norm:.6f}")
    print()
    
    # 4. Run self-healing experiment
    print("4. Running self-healing experiment...")
    print()
    
    experiment = SelfHealingExperiment(config)
    healing_metrics = experiment.run_experiment(
        metrics,
        doc_ids,
        perturbation_scale=0.2,
        healing_threshold=0.1
    )
    
    print()
    
    # 5. Generate visualizations
    print("5. Generating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    evolution.plot_evolution('results/ricci_evolution.png')
    print("   Saved: results/ricci_evolution.png")
    
    print()
    
    # 6. Summary
    print("6. Summary:")
    print(f"   Baseline Ricci decay: {baseline_history[0].ricci_norm:.6f} → {baseline_history[-1].ricci_norm:.6f}")
    print(f"   Self-healing time: {healing_metrics.time_to_heal} steps")
    print(f"   Recovery fidelity: {healing_metrics.recovery_fidelity:.2%}")
    print()
    
    print("✅ Ricci-REWA self-healing experiment complete!")

if __name__ == "__main__":
    main()
