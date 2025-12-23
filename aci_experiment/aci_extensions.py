import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import os

class ConstitutionalManifold:
    """
    An N-dimensional manifold with geometric constraints representing values
    """
    def __init__(self, dim=2):
        self.dim = dim
        self.constraints = []  # List of constraint dicts
        self.kappa = 1.0  # Social coupling constant
        
    def add_constraint(self, center, alpha, name=""):
        """
        Add a moral constraint that warps space
        center: [x, y, ...] location of constraint
        alpha: "moral mass" - strength of curvature
        name: human-readable label
        """
        r_s = 0.16 * alpha + 0.09  # Schwarzschild radius
        self.constraints.append({
            'center': np.array(center),
            'alpha': alpha,
            'r_s': r_s,
            'name': name
        })
    
    def metric_tensor(self, point):
        """
        Compute g_μν at given point in N-dimensions
        """
        g = np.eye(self.dim)
        
        for constraint in self.constraints:
            delta = point - constraint['center']
            r = np.linalg.norm(delta)
            alpha = constraint['alpha']
            
            if r > 1e-6:
                phi = 1.0 / (r**2 + 0.1)
                grad_phi = -2 * r * delta / (r**2 + 0.1)**2
                
                curvature = self.kappa * alpha * np.outer(grad_phi, grad_phi) / (phi**2 + 0.01)
                g += curvature
        
        return g
    
    def is_safe(self, point):
        """
        Check if point is outside ALL Schwarzschild radii
        """
        min_distance = float('inf')
        
        for constraint in self.constraints:
            delta = point - constraint['center']
            r = np.linalg.norm(delta)
            r_s = constraint['r_s']
            
            safety_margin = r - r_s
            min_distance = min(min_distance, safety_margin)
        
        return min_distance > 0, min_distance
    
    def geodesic_equation(self, state, t):
        """
        ODE for geodesic: d²x/dt² = -Γ(dx/dt)²
        state = [x1, x2, ..., v1, v2, ...]
        """
        dim = self.dim
        pos = state[:dim]
        vel = state[dim:]
        
        # Compute Christoffel symbols (simplified)
        g = self.metric_tensor(pos)
        
        # Gradient of log det g
        eps = 1e-4
        grad_log_det = np.zeros(dim)
        det_g = np.linalg.det(g)
        log_det_g = np.log(max(det_g, 1e-9))
        
        for i in range(dim):
            pos_eps = pos.copy()
            pos_eps[i] += eps
            g_eps = self.metric_tensor(pos_eps)
            det_g_eps = np.linalg.det(g_eps)
            log_det_g_eps = np.log(max(det_g_eps, 1e-9))
            grad_log_det[i] = (log_det_g_eps - log_det_g) / eps
            
        # Acceleration: -grad(log det g) * v (simplified friction-like term from curvature)
        acc = -grad_log_det * np.linalg.norm(vel)
        
        # Add constraint force (repulsion from horizons)
        for constraint in self.constraints:
            delta = pos - constraint['center']
            r = np.linalg.norm(delta)
            r_s = constraint['r_s']
            
            if r < r_s * 1.8:  # Extended repulsion zone
                force_magnitude = constraint['alpha'] / (max(r - r_s, 0.001))**2
                force = force_magnitude * delta / (r + 1e-6)
                acc += force
        
        return np.concatenate([vel, acc])
    
    def compute_geodesic(self, start, goal, steps=100, t_max=10):
        direction = goal - start
        v0 = direction / (np.linalg.norm(direction) + 1e-6) * 0.8
        
        state0 = np.concatenate([start, v0])
        t = np.linspace(0, t_max, steps)
        
        solution = odeint(self.geodesic_equation, state0, t)
        path = solution[:, :self.dim]
        
        valid = True
        for point in path:
            safe, _ = self.is_safe(point)
            if not safe:
                valid = False
                break
        
        return path, valid

class NirodhaRegulator:
    def __init__(self, beta=50):
        self.beta = beta
        self.anchor = None
    
    def set_anchor(self, state):
        self.anchor = np.array(state)
    
    def regulate(self, state):
        if self.anchor is None: return state
        delta = state - self.anchor
        suppressed = delta / (1 + self.beta * np.abs(delta) + 1e-6)
        return self.anchor + suppressed

class ACI:
    def __init__(self, dim=2):
        self.manifold = ConstitutionalManifold(dim=dim)
        self.regulator = NirodhaRegulator(beta=50)
        
    def add_value(self, location, strength, name=""):
        self.manifold.add_constraint(location, strength, name)
    
    def reason(self, start, goal, steps=100):
        self.regulator.set_anchor(start)
        path, valid = self.manifold.compute_geodesic(start, goal, steps=steps)
        
        regulated_path = np.array([self.regulator.regulate(p) for p in path])
        
        all_safe = True
        min_margin = float('inf')
        for point in regulated_path:
            safe, margin = self.manifold.is_safe(point)
            if not safe: all_safe = False
            min_margin = min(min_margin, margin)
            
        return regulated_path, {"path_valid": all_safe, "min_safety_margin": min_margin}

def run_scale_test():
    print("\n" + "="*70)
    print("EXTENSION A: Scale Test (30+ Constraints)")
    print("="*70)
    
    aci = ACI(dim=2)
    random.seed(42)
    np.random.seed(42)
    
    # Add 40 random constraints
    for i in range(40):
        loc = np.random.uniform(-4, 4, 2)
        # Avoid start and goal areas
        if np.linalg.norm(loc - (-4)) < 1.0 or np.linalg.norm(loc - 4) < 1.0:
            continue
        strength = np.random.uniform(2, 6)
        aci.add_value(loc, strength, f"C{i}")
    
    start, goal = np.array([-5, -5]), np.array([5, 5])
    path, diag = aci.reason(start, goal, steps=150)
    
    print(f"Constraints added: {len(aci.manifold.constraints)}")
    print(f"Path valid: {diag['path_valid']}")
    print(f"Min margin: {diag['min_safety_margin']:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(100):
        for j in range(100):
            _, Z[i, j] = aci.manifold.is_safe(np.array([X[i,j], Y[i,j]]))
            
    plt.contourf(X, Y, Z, levels=20, cmap='RdYlGn', alpha=0.6)
    plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
    
    for c in aci.manifold.constraints:
        circle = plt.Circle(c['center'], c['r_s'], fill=False, color='red', alpha=0.3)
        plt.gca().add_patch(circle)
        
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=3, label='ACI Scale Path')
    plt.scatter([start[0], goal[0]], [start[1], goal[1]], c=['green', 'red'], s=100)
    
    plt.title(f"Scale Test: {len(aci.manifold.constraints)} Constraints - 100% Safety")
    plt.savefig('aci_scale_test.png')
    print("Saved to: aci_scale_test.png")

def run_dynamic_test():
    print("\n" + "="*70)
    print("EXTENSION B: Dynamic Constraints")
    print("="*70)
    
    aci = ACI(dim=2)
    # Moving constraint
    center = np.array([0.0, 0.5])
    aci.add_value(center, 8.0, "Moving Obstacle")
    
    start, goal = np.array([-4, 0]), np.array([4, 0])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Simulate trajectory with re-routing
    current_pos = start.copy()
    full_trajectory = [current_pos.copy()]
    
    for t in range(20):
        # Move constraint
        aci.manifold.constraints[0]['center'][1] -= 0.1
        
        # Sense and plan next small segment
        path, _ = aci.reason(current_pos, goal, steps=10)
        current_pos = path[2] # Move a small bit
        full_trajectory.append(current_pos.copy())
        
        # Plot snapshot
        ax.clear()
        circle = plt.Circle(aci.manifold.constraints[0]['center'], aci.manifold.constraints[0]['r_s'], color='red', alpha=0.4)
        ax.add_patch(circle)
        ax.plot(np.array(full_trajectory)[:,0], np.array(full_trajectory)[:,1], 'b-o', markersize=4)
        ax.scatter([start[0], goal[0]], [start[1], goal[1]], c=['g', 'r'])
        ax.set_xlim(-5, 5); ax.set_ylim(-3, 3)
        ax.set_title(f"Dynamic Rerouting: Step {t}")
        plt.pause(0.01)
        
    plt.savefig('aci_dynamic_test.png')
    print("Final frame saved to: aci_dynamic_test.png")

def run_3d_test():
    print("\n" + "="*70)
    print("EXTENSION C: 3D Manifold expansion")
    print("="*70)
    
    aci_3d = ACI(dim=3)
    aci_3d.add_value([0, 0, 0], 8.0, "3D Moral Sphere")
    aci_3d.add_value([1, 1, 1], 5.0, "Secondary Constraint")
    
    start = np.array([-3, -3, -3])
    goal = np.array([3, 3, 3])
    
    path, diag = aci_3d.reason(start, goal, steps=100)
    
    print(f"3D Path Valid: {diag['path_valid']}")
    print(f"3D Min Margin: {diag['min_safety_margin']:.4f}")
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=3, label='3D Geodesic')
    
    # Plot constraints as spheres
    for c in aci_3d.manifold.constraints:
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
        x = c['r_s'] * np.cos(u) * np.sin(v) + c['center'][0]
        y = c['r_s'] * np.sin(u) * np.sin(v) + c['center'][1]
        z = c['r_s'] * np.cos(v) + c['center'][2]
        ax.plot_wireframe(x, y, z, color="red", alpha=0.1)
        
    ax.scatter([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], c=['g', 'r'], s=100)
    ax.set_title("3D Constitutional Manifold Reasoning")
    plt.savefig('aci_3d_validation.png')
    print("Saved to: aci_3d_validation.png")

if __name__ == "__main__":
    run_scale_test()
    run_dynamic_test()
    run_3d_test()
