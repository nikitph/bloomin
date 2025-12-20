import numpy as np
from constitutional_sdk import ConstitutionalLayer, Boundary, spherical_boundary, simple_linear_boundary
from typing import List, Dict, Tuple

class ConstitutionalAVController:
    """
    Controller that integrates the Constitutional SDK with AV sensor data.
    """
    def __init__(self, speed_limit=20.0):
        # Static boundaries (e.g., global speed limit)
        self.static_boundaries = [
            simple_linear_boundary(
                axis=2,  # Velocity dimension in [x, y, v, heading]
                limit=speed_limit,
                direction=-1,
                name="speed_limit",
                strength=100.0,
                description=f"Global speed limit {speed_limit} m/s"
            )
        ]
        self.speed_limit = speed_limit
        
    def get_safe_action(self, vehicle_state: np.ndarray, ml_action: np.ndarray, lidar_data: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Calculates safe control inputs by combining sensors and constitution.
        
        Args:
            vehicle_state: [x, y, v, psi]
            ml_action: [a, delta]
            lidar_data: list of obstacles
            
        Returns:
            safe_action: [a, delta]
            metadata: curvature and projection stats
        """
        # 1. Convert LIDAR detections to dynamic boundaries
        dynamic_boundaries = []
        v = vehicle_state[2]
        # Calculate dynamic safety buffer based on stopping distance (v^2 / 2a)
        stopping_buffer = (v**2) / (2 * 8.0) # 8.0 m/s^2 as typical braking limit
        
        for obs in lidar_data:
            b = spherical_boundary(
                center=obs['pos'],
                radius=obs['radius'] + 0.5 + stopping_buffer,
                name=obs['name'],
                strength=100.0,
                description=f"Obstacle at {obs['pos']} (incl. {stopping_buffer:.1f}m stopping buffer)"
            )
            dynamic_boundaries.append(b)
            
        # 2. Re-initialize layer with static + active dynamic boundaries
        # Note: In a true production system, we'd use a sparse or cached manifold
        layer = ConstitutionalLayer(self.static_boundaries + dynamic_boundaries)
        
        # 3. Project the 4D state action (state increment)
        # ml_action is [a, delta]. We need to map this to the state space [x, y, v, psi]
        # or just project the [a, delta] if we define boundaries in control space.
        # For simplicity in this demo, we project the state increment.
        
        # Approximate the next state delta based on the action
        v = vehicle_state[2]
        psi = vehicle_state[3]
        dt = 0.1 # matching sim
        
        # Current state components
        # state = [x, y, v, psi]
        
        # The ML policy provides [acceleration, steering]
        # We can treat the "intended action" as the resulting 4D vector delta
        # But we want to modify [accel, steering] primarily.
        
        # Let's project in the [a, delta] space directly by defining 
        # how [a, delta] affects the distance to boundaries.
        
        safe_action, _, metadata = layer.safe_step(vehicle_state, self._map_control_to_state_delta(vehicle_state, ml_action))
        
        # Extract the resulting a, delta from safe_action (which is a state delta)
        # This is a bit complex since many a,delta can yield same dx,dy.
        # Instead, let's use the SDK's projection on the ml_action vector directly 
        # by providing gradients in control space.
        
        # REFINED APPROACH: Direct Control Projection
        # We use safe_step on a control-mapped manifold to get redshift + projection
        safe_control, _, metadata = self._project_control_safe_step(vehicle_state, ml_action, dynamic_boundaries + self.static_boundaries)
        
        return safe_control, metadata

    def _project_control_safe_step(self, state, ml_control, boundaries):
        """
        Projects [accel, steering] using the safe_step API on a control-space manifold.
        """
        v = state[2]
        L = 2.5
        dt = 0.1
        
        control_boundaries = []
        for b in boundaries:
            def control_grad_fn(u, get_distance=False, b_orig=b):
                a, delta = u
                # Calculate state after one step at this control
                # We need a predictable position for the distance check
                next_v = state[2] + a * dt
                next_psi = state[3] + (state[2]/L) * np.tan(delta) * dt
                # Position is mostly determined by current velocity in first order
                next_x = state[0] + state[2] * np.cos(state[3]) * dt
                next_y = state[1] + state[2] * np.sin(state[3]) * dt
                hypo_state = np.array([next_x, next_y, next_v, next_psi])
                
                if get_distance:
                    return b_orig.distance(hypo_state)
                
                # Control Jacobian J = [ dState/da , dState/ddelta ]
                grad_s = b_orig.gradient(state)
                J = np.zeros((4, 2))
                # dv/da = dt
                J[2, 0] = dt
                # dx/da = (dx/dv_next) * (dv_next/da) = dt * dt
                J[0, 0] = np.cos(state[3]) * dt * dt
                J[1, 0] = np.sin(state[3]) * dt * dt
                # dpsi/ddelta
                J[3, 1] = (state[2]/L) * (1/np.cos(delta)**2) * dt
                return grad_s @ J
            
            # Use stronger strength for AV safety
            cb = Boundary(b.name, b.threshold, b.strength, control_grad_fn, b.description)
            control_boundaries.append(cb)
            
        control_layer = ConstitutionalLayer(control_boundaries)
        # Current 'state' in control space is [0, 0] (no change yet)
        return control_layer.safe_step(np.array([0.0, 0.0]), ml_control)

    def _map_control_to_state_delta(self, state, control):
        dt = 0.1
        L = 2.5
        a, delta = control
        v = state[2]
        psi = state[3]
        
        dx = v * np.cos(psi) * dt
        dy = v * np.sin(psi) * dt
        dv = a * dt
        dpsi = (v / L) * np.tan(delta) * dt
        return np.array([dx, dy, dv, dpsi])
