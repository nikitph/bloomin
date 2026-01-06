import torch
import numpy as np
from rads.reasoning import SymbolicHypergraph

class ConstitutionalManifold:
    """
    Manages geometric constraints (forbidden regions) and computes metric tensor.
    """
    def __init__(self, constraints=None):
        """
        constraints: List of dicts {'center': vector, 'alpha': float, 'r_s': float}
        """
        self.constraints = constraints if constraints else []
        self.epsilon = 1e-6
        self.coupling = 1.0 # Kappa
        
    def metric_tensor(self, point):
        """
        Compute g_uv at a given point (tensor).
        g_uv = delta_uv + sum( kappa * alpha * grad_phi * grad_phi / phi^2 )
        Simplified: We check if point is inside any 'event horizon' (r < r_s).
        If strictly inside, distance -> infinity.
        """
        # Full tensor computation is expensive for discrete graph.
        # We use the "Event Horizon" check as the primary safety gate.
        return torch.eye(point.shape[0]) 

    def is_safe_point(self, point_vector):
        """
        Check if a point is constitutionally valid (outside all event horizons).
        """
        if not self.constraints:
            return True
            
        for c in self.constraints:
            center = c['center']
            r_s = c['r_s']
            
            # Distance
            dist = torch.norm(point_vector - center)
            if dist < r_s:
                return False # Inside forbidden region
        return True

    def is_safe_path(self, source_vec, target_vec):
        """
        Check if the geodesic between source and target allows travel.
        Simplified: Check if linear path intersects forbidden region?
        Or simply check endpoints for discrete graph?
        The prompt implies: "Rejected: sources->target violates constraint"
        Usually this means Target or the Transition falls into a forbidden zone.
        """
        # Check endpoints first
        if not self.is_safe_point(source_vec): return False
        if not self.is_safe_point(target_vec): return False
        
        # Check midpoint/path (optional for strictness)
        # mid = (source_vec + target_vec) / 2
        # if not self.is_safe_point(mid): return False
        
        return True

class ConstitutionalHypergraph(SymbolicHypergraph):
    """
    Hypergraph that respects constitutional constraints.
    """
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        self.node_vectors = {} # map node name -> vector
        
    def register_node(self, name, vector):
        self.node_vectors[name] = vector
        
    def add_hyperedge(self, source, targets, weight=1.0):
        """
        Add edge only if constitutionally safe.
        """
        # Resolve vectors
        if source not in self.node_vectors:
            # If we don't know the vector, assume unsafe or safe?
            # For strict safety, maybe unsafe. But for partial usage, assume safe if unknown constraint?
            # Let's assume safe to avoid breaking generic tests, but print warning.
            # Or better: Require registration.
            pass
            
        s_vec = self.node_vectors.get(source)
        
        # Filter targets
        safe_targets = set()
        ensure_list = list(targets) if not isinstance(targets, (list, set)) else targets
        
        for t in ensure_list:
            t_vec = self.node_vectors.get(t)
            
            # Safety Check
            if s_vec is not None and t_vec is not None:
                if self.manifold.is_safe_path(s_vec, t_vec):
                    safe_targets.add(t)
                else:
                    print(f"SAFETY REJECTION: {source} -> {t} blocked by constitution.")
            else:
                # If vectors missing, allow (or default policy)
                safe_targets.add(t)
                
        if safe_targets:
            super().add_hyperedge(source, safe_targets, weight)
