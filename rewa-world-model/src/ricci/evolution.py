"""
Ricci Flow Evolution Module

Implements discrete Ricci flow PDE evolution:
dg/dt = -2·Ric + λ·S + κ·Δ_L g + ε·H

Where:
- Ric: Ricci curvature tensor
- S: Forcing tensor from contrastive loss
- Δ_L: Lichnerowicz Laplacian
- H: Stochastic noise

This enables self-healing of geometric structure.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from .ricci_tensor import RicciComputer, RicciTensorField
from .laplacian import LichnerowiczLaplacian

@dataclass
class EvolutionConfig:
    """Configuration for Ricci flow evolution"""
    dt: float = 0.01              # Time step
    num_steps: int = 100          # Number of evolution steps
    lambda_force: float = 0.1     # Forcing term coefficient
    kappa_diffusion: float = 0.05 # Diffusion coefficient
    epsilon_noise: float = 0.001  # Noise amplitude
    checkpoint_interval: int = 10  # Steps between checkpoints

@dataclass
class EvolutionState:
    """State of metric evolution at a time step"""
    step: int
    time: float
    metrics: List[np.ndarray]
    ricci_norm: float
    curvature_mean: float
    curvature_std: float

class RicciFlowEvolution:
    """Ricci flow evolution simulator"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.ricci_computer = RicciComputer()
        self.laplacian = LichnerowiczLaplacian()
        self.history: List[EvolutionState] = []
    
    def compute_forcing_tensor(
        self,
        metric: np.ndarray,
        target_metric: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute forcing tensor S that drives metric toward target.
        
        S = (g_target - g) / ||g_target - g||
        
        If no target, use identity (drives toward Euclidean)
        """
        if target_metric is None:
            target_metric = np.eye(len(metric))
        
        diff = target_metric - metric
        norm = np.linalg.norm(diff, 'fro')
        
        if norm < 1e-10:
            return np.zeros_like(metric)
        
        return diff / norm
    
    def evolve_step(
        self,
        metrics: List[np.ndarray],
        doc_ids: List[str],
        target_metrics: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform one evolution step.
        
        Returns:
            Updated metrics
        """
        n = len(metrics)
        new_metrics = []
        
        # Compute Ricci field
        ricci_field = self.ricci_computer.compute_ricci_field(metrics, doc_ids)
        
        # Evolve each metric
        for i, metric in enumerate(metrics):
            # Get neighbors
            neighbors = []
            for j in range(max(0, i-2), min(n, i+3)):
                if j != i:
                    neighbors.append(metrics[j])
            
            # Ricci term
            ricci = ricci_field.ricci_tensors[i]
            dg_ricci = -2 * ricci
            
            # Forcing term
            target = target_metrics[i] if target_metrics else None
            S = self.compute_forcing_tensor(metric, target)
            dg_force = self.config.lambda_force * S
            
            # Diffusion term (Lichnerowicz Laplacian)
            lap_L = self.laplacian.apply(metric, ricci, neighbors)
            dg_diffusion = self.config.kappa_diffusion * lap_L
            
            # Stochastic noise
            H = np.random.randn(*metric.shape)
            H = (H + H.T) / 2  # Symmetrize
            dg_noise = self.config.epsilon_noise * H
            
            # Total update
            dg = dg_ricci + dg_force + dg_diffusion + dg_noise
            
            # Euler step
            new_metric = metric + self.config.dt * dg
            
            # Ensure positive-definite
            new_metric = (new_metric + new_metric.T) / 2
            new_metric += np.eye(len(new_metric)) * 1e-4
            
            new_metrics.append(new_metric)
        
        return new_metrics
    
    def evolve(
        self,
        initial_metrics: List[np.ndarray],
        doc_ids: List[str],
        target_metrics: Optional[List[np.ndarray]] = None
    ) -> List[EvolutionState]:
        """
        Run full evolution.
        
        Returns:
            Evolution history
        """
        metrics = [m.copy() for m in initial_metrics]
        self.history = []
        
        for step in range(self.config.num_steps):
            # Compute diagnostics
            ricci_field = self.ricci_computer.compute_ricci_field(metrics, doc_ids)
            ricci_norm = np.sqrt(sum(
                np.linalg.norm(r, 'fro')**2 for r in ricci_field.ricci_tensors
            ))
            
            curvatures = ricci_field.scalar_curvatures
            curvature_mean = np.mean(curvatures)
            curvature_std = np.std(curvatures)
            
            # Record state
            if step % self.config.checkpoint_interval == 0:
                state = EvolutionState(
                    step=step,
                    time=step * self.config.dt,
                    metrics=[m.copy() for m in metrics],
                    ricci_norm=ricci_norm,
                    curvature_mean=curvature_mean,
                    curvature_std=curvature_std
                )
                self.history.append(state)
                
                print(f"Step {step}/{self.config.num_steps}: "
                      f"Ricci norm = {ricci_norm:.6f}, "
                      f"Curvature = {curvature_mean:.6f} ± {curvature_std:.6f}")
            
            # Evolve
            metrics = self.evolve_step(metrics, doc_ids, target_metrics)
        
        return self.history
    
    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot evolution curves"""
        if len(self.history) == 0:
            return
        
        steps = [s.step for s in self.history]
        ricci_norms = [s.ricci_norm for s in self.history]
        curvature_means = [s.curvature_mean for s in self.history]
        curvature_stds = [s.curvature_std for s in self.history]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Ricci norm
        axes[0].plot(steps, ricci_norms, 'b-', linewidth=2)
        axes[0].set_xlabel('Evolution Step')
        axes[0].set_ylabel('Ricci Norm')
        axes[0].set_title('Ricci Flow: Curvature Decay')
        axes[0].grid(True, alpha=0.3)
        
        # Scalar curvature
        axes[1].plot(steps, curvature_means, 'r-', linewidth=2, label='Mean')
        axes[1].fill_between(
            steps,
            np.array(curvature_means) - np.array(curvature_stds),
            np.array(curvature_means) + np.array(curvature_stds),
            alpha=0.3,
            color='r',
            label='±1 std'
        )
        axes[1].set_xlabel('Evolution Step')
        axes[1].set_ylabel('Scalar Curvature')
        axes[1].set_title('Scalar Curvature Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
