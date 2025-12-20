import numpy as np
import matplotlib.pyplot as plt
from engine import SocialPhysicsEngine, Action
from gr_core import (
    constitutional_metric, 
    numerical_derivative_metric, 
    christoffel_symbols, 
    numerical_derivative_christoffel, 
    riemann_tensor, 
    ricci_tensor, 
    ricci_scalar,
    role_tension_tensor
)
from matplotlib.colors import LogNorm

def run_einstein_social_experiment():
    print("Initializing Einstein-Social Field mapping...")
    engine = SocialPhysicsEngine(num_agents=1)
    
    # Ensure role boundaries are in engine
    # (they are initialized in engine.py constructor)
    boundaries = [engine.role_boundaries['parent'], engine.role_boundaries['professional']]
    
    # Define Roles as functional objects for tensor computation
    def parent_grad(theta): return np.array([-1.0, 0.0]) # Pulls toward lower risk
    def prof_grad(theta): return np.array([0.0, 1.0]) # Pulls toward more time
    
    roles = [
        {'name': 'parent', 'weight': 1.0, 'gradient': parent_grad},
        {'name': 'professional', 'weight': 1.0, 'gradient': prof_grad}
    ]
    
    res = 30
    # Avoid exact 0.0 or 1.0 for stability where phi -> 0
    risks = np.linspace(0.1, 0.9, res)
    deadlines = np.linspace(1.1, 5.0, res) 
    
    ricci_map = np.zeros((res, res))
    det_map = np.zeros((res, res))
    tension_map = np.zeros((res, res))
    redshift_map = np.zeros((res, res))
    
    print(f"Propagating fields over {res}x{res} grid...")
    for i, r in enumerate(risks):
        if i % 5 == 0: print(f"  Row {i}/{res}...")
        for j, d in enumerate(deadlines):
            theta = np.array([r, d])
            
            # 1. Metric
            g = constitutional_metric(theta, boundaries)
            det_map[i, j] = np.linalg.det(g)
            
            # 2. Curvature
            dg = numerical_derivative_metric(theta, boundaries)
            Gamma = christoffel_symbols(g, dg)
            dGamma = numerical_derivative_christoffel(theta, boundaries)
            R_t = riemann_tensor(Gamma, dGamma)
            Ric = ricci_tensor(R_t)
            R_s = ricci_scalar(Ric, g)
            ricci_map[i, j] = abs(R_s)
            
            # 3. Tension
            R_ij = role_tension_tensor(theta, roles)
            tension_map[i, j] = np.linalg.norm(R_ij)
            
            # 4. Redshift (Learning Rate)
            # eta_eff = 1 / sqrt(g_ii)
            redshift_map[i, j] = 1.0 / np.sqrt(g[0,0] + g[1,1])

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # A. Ricci Curvature Scalar (with safety offset for LogNorm)
    ricci_safe = np.maximum(ricci_map, 1e-10)
    im1 = axes[0,0].imshow(ricci_safe, cmap='magma', extent=[1.1, 5.0, 0.1, 0.9], origin='lower', 
                           norm=LogNorm(vmin=1e-10, vmax=ricci_safe.max()))
    axes[0,0].set_title("Ricci Scalar Curvature $|R|$ (Constitutional Tension)")
    axes[0,0].set_ylabel("Child Safety Risk")
    axes[0,0].set_xlabel("Time Until Deadline")
    plt.colorbar(im1, ax=axes[0,0])
    
    # B. Metric Determinant
    im2 = axes[0,1].imshow(det_map, cmap='viridis', extent=[1.1, 5.0, 0.1, 0.9], origin='lower', norm=LogNorm())
    axes[0,1].set_title("Metric Determinant $\det(g)$ (Volume Distortion)")
    axes[0,1].set_xlabel("Time Until Deadline")
    plt.colorbar(im2, ax=axes[0,1])
    
    # C. Role Tension Norm
    im3 = axes[1,0].imshow(tension_map, cmap='Reds', extent=[1.1, 5.0, 0.1, 0.9], origin='lower')
    axes[1,0].set_title("Role Tension Tensor Norm $||R_{ij}||$")
    axes[1,0].set_xlabel("Time Until Deadline")
    axes[1,0].set_ylabel("Child Safety Risk")
    plt.colorbar(im3, ax=axes[1,0])
    
    # D. Gravitational Redshift (Learning Rate)
    im4 = axes[1,1].imshow(redshift_map, cmap='coolwarm', extent=[1.1, 5.0, 0.1, 0.9], origin='lower')
    axes[1,1].set_title("Social Gravitational Redshift (Learning Rate $\eta_{eff}$)")
    axes[1,1].set_xlabel("Time Until Deadline")
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig("einstein_social_mapping.png")
    print("Mapping complete. Save as einstein_social_mapping.png")

if __name__ == "__main__":
    run_einstein_social_experiment()
