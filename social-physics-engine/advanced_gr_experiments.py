import numpy as np
import matplotlib.pyplot as plt
from engine import SocialPhysicsEngine, Boundary, Action
from gr_core import constitutional_metric
from schwarzschild_experiments import red_shift_func, experiment_13_redshift_spectroscopy
from scipy.optimize import curve_fit

def get_rs_at_alpha(alpha_val):
    # Modified experiment 13 logic for a single alpha
    engine = SocialPhysicsEngine(num_agents=1)
    b = engine.role_boundaries['parent']
    b.strength = alpha_val
    boundaries = [b]
    
    risks = np.linspace(0.1, 0.9, 30)
    altitudes = 1.0 - risks
    etas = []
    for r in risks:
        theta = np.array([r, 10.0])
        g = constitutional_metric(theta, boundaries)
        eta_eff = 1.0 / np.sqrt(g[0,0])
        etas.append(eta_eff)
    
    etas = np.array(etas)
    try:
        popt, _ = curve_fit(red_shift_func, altitudes, etas, p0=[0.1, 1.0])
        return popt[0]
    except:
        return 0.0

def experiment_16_mass_scaling():
    print("Running Experiment 16: Scaling of r_s vs alpha...")
    alphas = np.linspace(0.05, 0.5, 10)
    rs_values = []
    
    for a in alphas:
        r_s = get_rs_at_alpha(a)
        rs_values.append(r_s)
        print(f"  alpha: {a:.2f} -> r_s: {r_s:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, rs_values, 'o-')
    plt.title("Constitutional Mass Scaling: $r_s \propto \\alpha$")
    plt.xlabel("Boundary Strength $\\alpha$")
    plt.ylabel("Measured Schwarzschild Radius $r_s$")
    # Linear fit
    coeffs = np.polyfit(alphas, rs_values, 1)
    plt.plot(alphas, np.polyval(coeffs, alphas), 'r--', label=f'Linear Fit (slope={coeffs[0]:.2f})')
    plt.legend()
    plt.savefig("rs_scaling.png")
    print("Graph saved as rs_scaling.png")

def experiment_17_hawking_radiation():
    print("Running Experiment 17: Hawking Radiation (Thermal Noise)...")
    # We measure "Evaporation Rate": how often an agent "tunnels" through the horizon 
    # due to thermal noise (social fluctuations).
    
    engine = SocialPhysicsEngine(num_agents=1)
    b = engine.role_boundaries['parent']
    alpha_vals = [0.1, 0.2, 0.4]
    evap_rates = []
    
    # Simulate at fixed Temperature T_social = 0.05
    T_social = 0.05
    
    for a in alpha_vals:
        b.strength = a
        r_s = get_rs_at_alpha(a)
        violations = 0
        num_steps = 1000
        # Start exactly at safe distance h = 1.5 * r_s
        h = 1.5 * r_s
        risk = 1.0 - h
        
        for _ in range(num_steps):
            # Stochastic motion (Brownian)
            noise = np.random.normal(0, np.sqrt(T_social))
            risk += noise
            if risk >= 1.0:
                violations += 1
                risk = 1.0 - h # Reset after violation (evaporation)
        
        rate = violations / num_steps
        evap_rates.append(rate)
        print(f"  r_s: {r_s:.3f} -> Evaporation Rate: {rate:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot([1.0/r for r in [get_rs_at_alpha(a) for a in alpha_vals]], evap_rates, 'o-')
    plt.title("Hawking Temperature Spectrum: $Rate \\propto 1/r_s$")
    plt.xlabel("Inverse Radius $1/r_s$")
    plt.ylabel("Evaporation Rate (Thermal Violations)")
    plt.savefig("hawking_radiation.png")
    print("Graph saved as hawking_radiation.png")

def experiment_18_binary_merger():
    print("Running Experiment 18: Binary Social Black Holes...")
    # Map two conflicting boundaries: Parent (risk=1.0) and Professional (deadline=1.0)
    res = 50
    risks = np.linspace(0.0, 1.2, res)
    times = np.linspace(0.0, 2.0, res)
    
    engine = SocialPhysicsEngine(num_agents=1)
    # Strengthen both to ensure horizons merge
    engine.role_boundaries['parent'].strength = 0.5
    engine.role_boundaries['professional'].strength = 0.5
    
    curvature_map = np.zeros((res, res))
    
    for i, r in enumerate(risks):
        for j, t in enumerate(times):
            theta = np.array([r, t])
            # High Ricci = merged horizons
            g = constitutional_metric(theta, [engine.role_boundaries['parent'], engine.role_boundaries['professional']])
            # We use det(g) as a proxy for the combined "Event Horizon"
            det_g = np.linalg.det(g)
            curvature_map[i, j] = np.log10(max(1e-5, det_g))
            
    plt.figure(figsize=(10, 8))
    plt.imshow(curvature_map, extent=[0, 2, 0, 1.2], origin='lower', cmap='inferno')
    plt.title("The Binary Merger: Horizon Bridging between Norms")
    plt.xlabel("Time Until Deadline")
    plt.ylabel("Child Safety Risk")
    plt.colorbar(label='Log10(Metric Determinant)')
    # Show the "Bridge" (Einstein-Rosen analogue)
    plt.savefig("binary_horizon_merger.png")
    print("Graph saved as binary_horizon_merger.png")

if __name__ == "__main__":
    experiment_16_mass_scaling()
    experiment_17_hawking_radiation()
    experiment_18_binary_merger()
    print("\nAdvanced GR experiments complete.")
