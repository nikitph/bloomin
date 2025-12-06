"""
Ricci Flow Module

Implements geometric flow updates, curvature computation, and self-healing
via Ricci flow on the semantic manifold.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.stats import entropy
from dataclasses import dataclass


@dataclass
class FlowParams:
    """Parameters for Ricci flow"""
    step_size: float = 1e-3
    viscosity: float = 0.01  # Regularization (kappa)
    max_displacement: float = 0.1  # Clamp for stability
    

class RicciFlow:
    """
    Ricci Flow for geometric self-healing of the semantic manifold.
    
    Implements:
    - Curvature computation from Fisher metric
    - Flow step updates based on error signals
    - Curvature entropy calculation
    - Metric update application
    """
    
    def __init__(self, params: Optional[FlowParams] = None):
        self.params = params if params is not None else FlowParams()
        self.flow_history: List[Dict] = []
        
    def compute_ricci_curvature(self, metric: np.ndarray) -> np.ndarray:
        """
        Compute Ricci curvature tensor from metric.
        
        This is a simplified approximation. In practice, would use
        discrete Ricci curvature (Ollivier-Ricci) or spectral methods.
        
        Args:
            metric: Fisher metric tensor (d x d)
            
        Returns:
            Ricci curvature tensor (d x d)
        """
        # Approximate Ricci via Laplacian of metric
        # Ric ≈ -Δg (simplified)
        
        d = metric.shape[0]
        
        # Compute eigenvalues (curvature related to spectral properties)
        eigenvalues = np.linalg.eigvalsh(metric)
        
        # Ricci approximation: deviation from flat metric
        flat_metric = np.eye(d)
        ricci = metric - flat_metric
        
        # Scale by eigenvalue spectrum (regions with high curvature have extreme eigenvalues)
        mean_eig = np.mean(eigenvalues)
        ricci = ricci * (1.0 / (mean_eig + 1e-6))
        
        return ricci
    
    def compute_curvature_entropy(self, metric: np.ndarray) -> float:
        """
        Compute entropy of curvature distribution.
        
        Args:
            metric: Fisher metric tensor
            
        Returns:
            Spectral entropy S = -sum p(r) log p(r)
        """
        ricci = self.compute_ricci_curvature(metric)
        
        # Get curvature values (eigenvalues of Ricci tensor)
        curvatures = np.linalg.eigvalsh(ricci)
        
        # Normalize to probability distribution
        curvatures_abs = np.abs(curvatures)
        if np.sum(curvatures_abs) < 1e-10:
            return 0.0
        
        p = curvatures_abs / np.sum(curvatures_abs)
        
        # Compute entropy
        S = entropy(p + 1e-10)  # Regularize
        
        return S
    
    def flow_step(
        self,
        current_metric: np.ndarray,
        error_signal: Dict,
        region_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Perform one Ricci flow step to correct geometric inconsistency.
        
        Flow equation: ∂g/∂t = -2 Ric(g) + viscosity·error_signal
        
        Args:
            current_metric: Current Fisher metric
            error_signal: Dictionary with 'kl_divergence', 'target_distribution', etc.
            region_ids: Optional region identifiers for localized flow
            
        Returns:
            Metric update Δg
        """
        d = current_metric.shape[0]
        
        # Compute Ricci curvature
        ricci = self.compute_ricci_curvature(current_metric)
        
        # Flow direction: -2 Ric(g)
        flow_direction = -2.0 * ricci
        
        # Add error correction term
        kl_error = error_signal.get('kl_divergence', 0.0)
        
        # Error signal pushes metric toward consistency
        # Simple heuristic: scale flow by KL error magnitude
        error_scale = np.tanh(kl_error)  # Bounded scaling
        
        # Viscosity term (regularization)
        viscosity_term = self.params.viscosity * (np.eye(d) - current_metric)
        
        # Combined update
        delta_g = self.params.step_size * (
            flow_direction * error_scale + viscosity_term
        )
        
        # Clamp displacement for stability
        delta_g = np.clip(delta_g, -self.params.max_displacement, self.params.max_displacement)
        
        # Log flow step
        self.flow_history.append({
            'kl_error': kl_error,
            'curvature_norm': np.linalg.norm(ricci),
            'update_norm': np.linalg.norm(delta_g),
            'region_ids': region_ids
        })
        
        return delta_g
    
    def apply_metric_update(
        self,
        current_metric: np.ndarray,
        delta_g: np.ndarray
    ) -> np.ndarray:
        """
        Apply metric update and ensure positive definiteness.
        
        Args:
            current_metric: Current metric
            delta_g: Update from flow_step
            
        Returns:
            Updated metric (guaranteed positive definite)
        """
        new_metric = current_metric + delta_g
        
        # Ensure symmetry
        new_metric = (new_metric + new_metric.T) / 2.0
        
        # Ensure positive definiteness via eigenvalue clamping
        eigenvalues, eigenvectors = np.linalg.eigh(new_metric)
        eigenvalues = np.maximum(eigenvalues, 1e-6)  # Clamp to positive
        
        new_metric = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return new_metric
    
    def compute_curvature_anomaly(
        self,
        metric_before: np.ndarray,
        metric_after: np.ndarray
    ) -> float:
        """
        Compute curvature anomaly norm for self-healing evaluation.
        
        Args:
            metric_before: Metric before corruption/healing
            metric_after: Metric after healing
            
        Returns:
            ||Ric(g_after) - Ric(g_before)||_2
        """
        ricci_before = self.compute_ricci_curvature(metric_before)
        ricci_after = self.compute_ricci_curvature(metric_after)
        
        anomaly = np.linalg.norm(ricci_after - ricci_before, ord='fro')
        
        return anomaly
    
    def get_statistics(self) -> Dict:
        """Get flow statistics"""
        if len(self.flow_history) == 0:
            return {
                'num_steps': 0,
                'total_curvature_change': 0.0
            }
        
        return {
            'num_steps': len(self.flow_history),
            'total_curvature_change': sum(h['update_norm'] for h in self.flow_history),
            'mean_kl_error': np.mean([h['kl_error'] for h in self.flow_history]),
            'final_curvature_norm': self.flow_history[-1]['curvature_norm']
        }
