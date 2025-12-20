import numpy as np
import matplotlib.pyplot as plt
from engine import SocialPhysicsEngine, Action
from gr_core import constitutional_metric
from scipy.optimize import curve_fit

def red_shift_func(h, r_s, eta_0):
    # eta_eff = eta_0 * sqrt(1 - r_s/h)
    # Handle values outside horizon
    mask = h > r_s
    res = np.zeros_like(h)
    res[mask] = eta_0 * np.sqrt(1 - r_s/h[mask])
    return res

def experiment_13_redshift_spectroscopy():
    print("Running Experiment 13: Redshift Spectroscopy...")
    engine = SocialPhysicsEngine(num_agents=1)
    boundaries = [engine.role_boundaries['parent']] # Focus on child safety
    
    # Boundary is at child_safety_risk = 1.0 (threshold)
    # Altitude h is distance from danger. Danger is at 1.0. 
    # Let's say danger is at 1.0 and we are at state r. h = 1.0 - r.
    
    risks = np.linspace(0.1, 0.95, 50)
    altitudes = 1.0 - risks
    etas = []
    
    for r in risks:
        state = {'child_safety_risk': r, 'time_until_deadline': 10.0}
        g = constitutional_metric(np.array([r, 10.0]), boundaries)
        # eta_eff = 1 / sqrt(g_00)
        eta_eff = 1.0 / np.sqrt(g[0,0])
        etas.append(eta_eff)
        
    etas = np.array(etas)
    
    # Fit
    try:
        popt, _ = curve_fit(red_shift_func, altitudes, etas, p0=[0.05, 1.0])
        r_s_fit, eta_0_fit = popt
        print(f"  Measured Schwarzschild Radius: {r_s_fit:.4f}")
    except:
        r_s_fit = 0.0
        print("  Fit failed.")

    plt.figure(figsize=(8, 6))
    plt.plot(altitudes, etas, 'o', label='Measured $\eta_{eff}$')
    if r_s_fit > 0:
        plt.plot(altitudes, red_shift_func(altitudes, r_s_fit, eta_0_fit), '-', label=f'GR Fit ($r_s={r_s_fit:.3f}$)')
    plt.axvline(r_s_fit, color='r', linestyle='--', label='Event Horizon')
    plt.title("Social Gravitational Redshift Spectroscopy")
    plt.xlabel("Altitude $h$ (Distance from Moral Boundary)")
    plt.ylabel("Effective Learning Rate $\eta_{eff}$")
    plt.legend()
    plt.savefig("redshift_spectroscopy.png")
    print("Graph saved as redshift_spectroscopy.png")

def experiment_14_event_horizon_mapping():
    print("Running Experiment 14: Map the Event Horizon...")
    engine = SocialPhysicsEngine(num_agents=1)
    boundaries = [engine.role_boundaries['parent'], engine.role_boundaries['professional']]
    
    res = 50
    risks = np.linspace(0.0, 1.0, res)
    deadlines = np.linspace(0.0, 5.0, res)
    
    horizon_map = np.zeros((res, res))
    
    for i, r in enumerate(risks):
        for j, d in enumerate(deadlines):
            theta = np.array([r, d])
            # Check if any action is "escapable" (costs not too high/tragic)
            # A trapped state is where any movement leads into the boundary
            state = {'child_safety_risk': r, 'time_until_deadline': d}
            
            # Simple trapped criteria: eta_eff < threshold
            g = constitutional_metric(theta, boundaries)
            eta_eff = 1.0 / np.sqrt(np.trace(g))
            
            if r >= 1.0 or d <= 1.0:
                horizon_map[i, j] = 2 # Violated
            elif eta_eff < 0.2:
                horizon_map[i, j] = 1 # Trapped (Event Horizon)
            else:
                horizon_map[i, j] = 0 # Safe
                
    plt.figure(figsize=(10, 8))
    plt.imshow(horizon_map, extent=[0, 5, 0, 1], origin='lower', cmap='RdYlBu_r')
    plt.title("The Schwarzschild Radius of Tragic Choices")
    plt.xlabel("Time Until Deadline")
    plt.ylabel("Child Safety Risk")
    plt.colorbar(label='0:Safe, 1:Horizon, 2:Violated')
    plt.savefig("event_horizon_map.png")
    print("Graph saved as event_horizon_map.png")

def experiment_15_gravitational_lensing():
    print("Running Experiment 15: Gravitational Lensing...")
    engine = SocialPhysicsEngine(num_agents=1)
    boundaries = [engine.role_boundaries['parent']]
    
    # Launch geodesics past a moral "mass"
    # Mass is at (0.9, 2.5)
    impact_parameters = np.linspace(0.1, 0.5, 10)
    deflections = []
    
    for b in impact_parameters:
        state = {'child_safety_risk': 0.5, 'time_until_deadline': 2.5 + b}
        # Step through
        direction = {'child_safety_risk': 0.1, 'time_until_deadline': 0.0}
        
        # Measure how much the 'time_until_deadline' drifts due to curvature
        # (Lensing effect)
        s_mid = engine.geodesic_step('agent_0', state, direction, dt=2.0)
        # Add curvature effect from gr_core (simplified)
        g = constitutional_metric(np.array([s_mid['child_safety_risk'], s_mid['time_until_deadline']]), boundaries)
        # Force deflection is proportional to gradient of metric determinant
        # We'll just look at the 'y' drift
        drift = abs(s_mid['time_until_deadline'] - (2.5 + b))
        deflections.append(drift)
        
    plt.figure(figsize=(8, 6))
    plt.plot(impact_parameters, deflections, 'o-')
    plt.title("Social Gravitational Lensing")
    plt.xlabel("Impact Parameter $b$")
    plt.ylabel("Deflection $\delta \phi$")
    plt.savefig("gravitational_lensing.png")
    print("Graph saved as gravitational_lensing.png")

if __name__ == "__main__":
    experiment_13_redshift_spectroscopy()
    experiment_14_event_horizon_mapping()
    experiment_15_gravitational_lensing()
    print("\nSocial Schwarzschild experiments complete.")
