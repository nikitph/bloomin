"""
Self-Healing Experiments Module

Tests the self-healing capabilities of Ricci-REWA evolution:
1. Perturb geometric structure
2. Run Ricci flow evolution
3. Measure recovery fidelity
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from .evolution import RicciFlowEvolution, EvolutionConfig

@dataclass
class HealingMetrics:
    """Metrics for self-healing experiment"""
    time_to_heal: int              # Steps to reach threshold
    recovery_fidelity: float       # Fraction of structure recovered
    final_ricci_norm: float        # Final curvature norm
    initial_perturbation: float    # Initial perturbation magnitude

class SelfHealingExperiment:
    """Self-healing experiment runner"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.evolution = RicciFlowEvolution(config)
    
    def perturb_metrics(
        self,
        metrics: List[np.ndarray],
        perturbation_scale: float = 0.1,
        fraction_perturbed: float = 0.2
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Inject perturbations into metrics.
        
        Args:
            metrics: Original metrics
            perturbation_scale: Scale of perturbation
            fraction_perturbed: Fraction of metrics to perturb
            
        Returns:
            (perturbed_metrics, perturbed_indices)
        """
        n = len(metrics)
        num_perturbed = int(n * fraction_perturbed)
        
        # Select random indices to perturb
        perturbed_indices = np.random.choice(n, num_perturbed, replace=False).tolist()
        
        # Copy metrics
        perturbed = [m.copy() for m in metrics]
        
        # Add perturbations
        for idx in perturbed_indices:
            d = len(metrics[idx])
            noise = np.random.randn(d, d) * perturbation_scale
            noise = (noise + noise.T) / 2  # Symmetrize
            
            perturbed[idx] += noise
            
            # Ensure positive-definite
            perturbed[idx] = (perturbed[idx] + perturbed[idx].T) / 2
            perturbed[idx] += np.eye(d) * 1e-3
        
        return perturbed, perturbed_indices
    
    def measure_recovery(
        self,
        original_metrics: List[np.ndarray],
        recovered_metrics: List[np.ndarray],
        perturbed_indices: List[int]
    ) -> float:
        """
        Measure recovery fidelity.
        
        Fidelity = 1 - ||g_recovered - g_original|| / ||g_perturbed - g_original||
        """
        total_recovery = 0.0
        
        for idx in perturbed_indices:
            g_orig = original_metrics[idx]
            g_recovered = recovered_metrics[idx]
            
            # Distance to original
            dist_recovered = np.linalg.norm(g_recovered - g_orig, 'fro')
            
            # Normalize by original perturbation
            # (would need to store perturbed state, simplified here)
            total_recovery += 1.0 / (1.0 + dist_recovered)
        
        return total_recovery / len(perturbed_indices) if perturbed_indices else 0.0
    
    def run_experiment(
        self,
        metrics: List[np.ndarray],
        doc_ids: List[str],
        perturbation_scale: float = 0.1,
        healing_threshold: float = 0.1
    ) -> HealingMetrics:
        """
        Run self-healing experiment.
        
        Returns:
            HealingMetrics
        """
        print("=== Self-Healing Experiment ===")
        print(f"Perturbation scale: {perturbation_scale}")
        print(f"Healing threshold: {healing_threshold}")
        print()
        
        # Store original metrics
        original_metrics = [m.copy() for m in metrics]
        
        # Perturb
        print("Injecting perturbations...")
        perturbed_metrics, perturbed_indices = self.perturb_metrics(
            metrics,
            perturbation_scale=perturbation_scale
        )
        
        initial_perturbation = np.mean([
            np.linalg.norm(perturbed_metrics[i] - original_metrics[i], 'fro')
            for i in perturbed_indices
        ])
        
        print(f"Perturbed {len(perturbed_indices)} metrics")
        print(f"Initial perturbation: {initial_perturbation:.6f}")
        print()
        
        # Run evolution
        print("Running Ricci flow evolution...")
        history = self.evolution.evolve(
            perturbed_metrics,
            doc_ids,
            target_metrics=original_metrics
        )
        
        # Find time to heal
        time_to_heal = self.config.num_steps
        for state in history:
            if state.ricci_norm < healing_threshold:
                time_to_heal = state.step
                break
        
        # Measure recovery
        final_metrics = history[-1].metrics if history else perturbed_metrics
        recovery_fidelity = self.measure_recovery(
            original_metrics,
            final_metrics,
            perturbed_indices
        )
        
        final_ricci_norm = history[-1].ricci_norm if history else 0.0
        
        metrics = HealingMetrics(
            time_to_heal=time_to_heal,
            recovery_fidelity=recovery_fidelity,
            final_ricci_norm=final_ricci_norm,
            initial_perturbation=initial_perturbation
        )
        
        print()
        print("Results:")
        print(f"  Time to heal: {metrics.time_to_heal} steps")
        print(f"  Recovery fidelity: {metrics.recovery_fidelity:.2%}")
        print(f"  Final Ricci norm: {metrics.final_ricci_norm:.6f}")
        
        return metrics
