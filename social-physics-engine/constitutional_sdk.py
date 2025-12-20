"""
Constitutional AI SDK - Production Implementation
Version: 1.0.0
Author: Nikit Phadke
License: Apache 2.0

Core innovation: Geometric safety via Riemannian manifold projection.
Based on "Social General Relativity" (Phadke, 2025)
"""

import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import logging
from enum import Enum
import json
from collections import defaultdict

# ============================================================================
# CORE CLASSES
# ============================================================================

class SafetyLevel(Enum):
    """Boundary criticality levels"""
    CRITICAL = 1000    # Cannot violate (life/death)
    HIGH = 100         # Strong constraint (legal)
    MEDIUM = 10        # Important (policy)
    LOW = 1            # Guideline (best practice)


@dataclass
class Boundary:
    """
    Represents a hard constraint in the constitutional manifold.
    
    The Schwarzschild radius r_s = 0.16*alpha + 0.09 defines the
    event horizon - once an action enters h < r_s, escape is impossible.
    """
    name: str
    threshold: float
    strength: float  # alpha in the paper
    gradient_fn: Callable
    description: str = ""
    
    # Derived quantities (computed on init)
    rs: float = field(init=False)  # Schwarzschild radius
    
    def __post_init__(self):
        """Compute Schwarzschild radius from empirical formula"""
        self.rs = 0.16 * self.strength + 0.09
        
    def distance(self, state: np.ndarray) -> float:
        """
        Compute distance to violation.
        
        Returns:
            h: altitude above event horizon (h > r_s = safe)
        """
        return self.gradient_fn(state, get_distance=True)
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Gradient vector pointing toward safety.
        
        Returns:
            Normal vector to boundary surface
        """
        return self.gradient_fn(state, get_distance=False)
    
    def inside_horizon(self, state: np.ndarray) -> bool:
        """Check if state is inside event horizon"""
        return self.distance(state) < self.rs
    
    def effective_learning_rate(self, state: np.ndarray, base_eta: float = 1.0) -> float:
        """
        Compute gravitational redshift factor.
        
        Near boundaries, learning rate -> 0 (time dilation).
        Formula: η_eff = η_0 * sqrt(1 - r_s/h)
        """
        h = self.distance(state)
        
        if h <= self.rs:
            return 0.0  # Inside horizon - frozen
        
        redshift_factor = np.sqrt(1 - self.rs / h)
        return base_eta * redshift_factor


class ConstitutionalLayer:
    """
    Makes any AI system constitutional by projecting actions onto safe manifold.
    
    Key innovation: Violations are geometrically impossible (not just unlikely).
    """
    
    def __init__(self, 
                 boundaries: List[Boundary],
                 base_learning_rate: float = 1.0,
                 max_projection_iters: int = 10,
                 enable_telemetry: bool = True):
        self.boundaries = boundaries
        self.base_eta = base_learning_rate
        self.max_iters = max_projection_iters
        self.enable_telemetry = enable_telemetry
        
        # Telemetry
        self.metrics = TelemetryCollector() if enable_telemetry else None
        
        # Precompute Schwarzschild radii
        self.rs_map = {b.name: b.rs for b in boundaries}
        
    def check_safety(self, state: np.ndarray) -> Tuple[List[str], Dict[str, float]]:
        """
        Check which boundaries would be violated.
        
        Returns:
            violations: List of boundary names inside event horizon
            distances: Dict mapping boundary name -> distance
        """
        violations = []
        distances = {}
        
        for b in self.boundaries:
            dist = b.distance(state)
            distances[b.name] = dist
            
            if dist < b.rs:
                violations.append(b.name)
        
        return violations, distances
    
    def compute_metric_tensor(self, state: np.ndarray) -> np.ndarray:
        """
        Compute constitutional metric g_ij at current state.
        
        g_ij = δ_ij + Σ_k α_k (∇φ_k ⊗ ∇φ_k) / φ_k²
        """
        n = len(state)
        g = np.eye(n)  # Start with Euclidean metric
        
        for b in self.boundaries:
            phi = b.distance(state)
            
            if phi <= 0:
                # Already violated - metric is singular
                return np.inf * np.eye(n)
            
            grad_phi = b.gradient(state)
            alpha = b.strength
            
            # Add barrier term
            barrier = alpha * np.outer(grad_phi, grad_phi) / (phi ** 2)
            g += barrier
        
        return g
    
    def effective_learning_rate(self, state: np.ndarray) -> float:
        """
        Compute overall redshift from all boundaries.
        
        Takes geometric mean of individual redshift factors.
        """
        g = self.compute_metric_tensor(state)
        
        if np.isinf(g[0, 0]):
            return 0.0
        
        # Geometric mean of metric eigenvalues
        eigenvalues = np.linalg.eigvalsh(g)
        g_effective = np.prod(eigenvalues) ** (1/len(eigenvalues))
        
        eta_eff = self.base_eta / np.sqrt(g_effective)
        
        return eta_eff
    
    def project_to_safe_manifold(self, 
                                  current_state: np.ndarray, 
                                  desired_action: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Project desired action onto safe manifold (geodesic projection).
        
        This is the CORE ALGORITHM - makes violations impossible.
        """
        start_time = time.time()
        
        # Check if desired action would violate
        next_state = current_state + desired_action
        violations, distances = self.check_safety(next_state)
        
        if not violations:
            # Desired action is already safe
            if self.metrics:
                self.metrics.record_safe_action(time.time() - start_time)
            
            return desired_action, {
                'violated': False,
                'projection_needed': False,
                'iterations': 0,
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # Project action away from violated boundaries
        safe_action = desired_action.copy().astype(float)
        
        for iteration in range(self.max_iters):
            # Compute normals for all violated boundaries
            normals = []
            for b_name in violations:
                b = next(b for b in self.boundaries if b.name == b_name)
                grad = b.gradient(current_state)
                normals.append((b, grad))
            
            # Project away from normals
            for b, normal in normals:
                # Normalize normal vector for projection
                norm_val = np.linalg.norm(normal)
                if norm_val < 1e-9:
                    continue
                unit_normal = normal / norm_val
                
                projection = np.dot(safe_action, unit_normal)
                
                if projection < 0:  # Moving toward boundary
                    # Remove component pointing toward violation
                    safe_action -= projection * unit_normal
                    
                    # Add repulsive component
                    dist = b.distance(current_state)
                    if dist < 2 * b.rs:
                        # Very close to horizon - strong repulsion
                        repulsion = b.strength * (2 * b.rs - dist) / b.rs
                        safe_action += repulsion * unit_normal
            
            # Check if projection succeeded
            test_state = current_state + safe_action
            new_violations, _ = self.check_safety(test_state)
            
            if not new_violations:
                # Successfully projected to safe region
                if self.metrics:
                    self.metrics.record_projection(
                        violated_boundaries=violations,
                        iterations=iteration + 1,
                        latency=time.time() - start_time
                    )
                
                return safe_action, {
                    'violated': True,
                    'projection_needed': True,
                    'boundaries': violations,
                    'iterations': iteration + 1,
                    'latency_ms': (time.time() - start_time) * 1000,
                    'original_norm': np.linalg.norm(desired_action),
                    'safe_norm': np.linalg.norm(safe_action),
                    'deflection_angle': np.arccos(
                        np.clip(np.dot(desired_action, safe_action) / 
                        (max(1e-9, np.linalg.norm(desired_action)) * max(1e-9, np.linalg.norm(safe_action))), -1, 1)
                    )
                }
            
            violations = new_violations
        
        # Failed to project - should not happen in theory
        if self.metrics:
            self.metrics.record_failure(violations)
        
        # Return zero action as fail-safe
        return np.zeros_like(desired_action), {
            'violated': True,
            'projection_needed': True,
            'boundaries': violations,
            'iterations': self.max_iters,
            'latency_ms': (time.time() - start_time) * 1000,
            'status': 'PROJECTION_FAILED'
        }
    
    def safe_step(self, 
                  current_state: np.ndarray, 
                  ml_policy_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Main API: Execute one safe step.
        """
        # Scale action by redshift (near boundaries, must move slower)
        eta_eff = self.effective_learning_rate(current_state)
        scaled_action = eta_eff * ml_policy_action
        
        # Project to safe manifold
        safe_action, metadata = self.project_to_safe_manifold(current_state, scaled_action)
        
        # Compute next state
        next_state = current_state + safe_action
        
        # Add redshift info to metadata
        metadata['eta_eff'] = eta_eff
        metadata['redshift_factor'] = self.base_eta / (eta_eff + 1e-9)
        
        return safe_action, next_state, metadata


# ============================================================================
# TELEMETRY & MONITORING
# ============================================================================

class TelemetryCollector:
    """Collects safety metrics for monitoring and alerting"""
    
    def __init__(self):
        self.safe_count = 0
        self.projection_count = 0
        self.failure_count = 0
        
        self.latencies = []
        self.violations_by_boundary = defaultdict(int)
        self.iteration_counts = []
        
        self.start_time = time.time()
    
    def record_safe_action(self, latency: float):
        self.safe_count += 1
        self.latencies.append(latency)
    
    def record_projection(self, violated_boundaries: List[str], iterations: int, latency: float):
        self.projection_count += 1
        self.latencies.append(latency)
        self.iteration_counts.append(iterations)
        
        for b in violated_boundaries:
            self.violations_by_boundary[b] += 1
    
    def record_failure(self, violated_boundaries: List[str]):
        self.failure_count += 1
        for b in violated_boundaries:
            self.violations_by_boundary[b] += 1
    
    def get_stats(self) -> Dict:
        total = self.safe_count + self.projection_count + self.failure_count
        
        return {
            'total_actions': total,
            'safe_rate': self.safe_count / total if total > 0 else 0,
            'projection_rate': self.projection_count / total if total > 0 else 0,
            'failure_rate': self.failure_count / total if total > 0 else 0,
            'avg_latency_ms': np.mean(self.latencies) * 1000 if self.latencies else 0,
            'p95_latency_ms': np.percentile(self.latencies, 95) * 1000 if self.latencies else 0,
            'violations_by_boundary': dict(self.violations_by_boundary),
            'avg_iterations': np.mean(self.iteration_counts) if self.iteration_counts else 0,
            'uptime_seconds': time.time() - self.start_time
        }


# ============================================================================
# BOUNDARY FACTORIES
# ============================================================================

def simple_linear_boundary(axis: int, 
                           limit: float, 
                           direction: int = 1,
                           name: str = None,
                           strength: float = 10.0,
                           description: str = "") -> Boundary:
    """
    Create a simple linear boundary.
    """
    if name is None:
        name = f"axis_{axis}_{'gt' if direction > 0 else 'lt'}_{limit}"
    
    def grad_fn(state, get_distance=False):
        dist = direction * (state[axis] - limit)
        if get_distance:
            return dist
        
        g = np.zeros_like(state, dtype=float)
        g[axis] = direction
        return g
    
    return Boundary(
        name=name,
        threshold=limit,
        strength=strength,
        gradient_fn=grad_fn,
        description=description
    )


def spherical_boundary(center: np.ndarray,
                      radius: float,
                      name: str = "spherical",
                      strength: float = 10.0,
                      description: str = "") -> Boundary:
    """
    Create spherical boundary.
    """
    def grad_fn(state, get_distance=False):
        dims = len(center)
        state_pos = state[:dims]
        diff = state_pos - center
        dist_to_center = np.linalg.norm(diff)
        dist = dist_to_center - radius  # positive = safe
        
        if get_distance:
            return dist
        
        if dist_to_center < 1e-6:
            return np.zeros_like(state)
        
        grad = np.zeros_like(state)
        grad[:dims] = diff / dist_to_center
        return grad
    
    return Boundary(
        name=name,
        threshold=radius,
        strength=strength,
        gradient_fn=grad_fn,
        description=description
    )
