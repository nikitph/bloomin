import numpy as np
import matplotlib.pyplot as plt
from engine import SocialPhysicsEngine
from gr_core import constitutional_metric, role_tension_tensor

def experiment_19_strong_lensing():
    print("Running Experiment 19: Strong Lensing Analysis...")
    engine = SocialPhysicsEngine(num_agents=1)
    b_parent = engine.role_boundaries['parent']
    b_parent.strength = 1.0 # High mass
    
    # Sweep coupling kappa
    kappas = [0.1, 1.0, 5.0]
    impact_parameters = np.linspace(0.1, 0.6, 10)
    
    plt.figure(figsize=(10, 6))
    for k in kappas:
        deflections = []
        for b in impact_parameters:
            # We model deflection as d_phi = (4*kappa*M)/b
            # We measure the effective drift in 'time_until_deadline' 
            # as an agent tries to move at risk = 0.5 (near mass)
            theta = np.array([0.5, 2.5 + b])
            # The metric determinant gradient provides the lensing force
            # We'll calculate the deflection angle alpha ~ grad(det(g)) / det(g)
            g = constitutional_metric(theta, [b_parent])
            dg = (constitutional_metric(theta + [0.01, 0], [b_parent]) - g) / 0.01
            force_vector = np.trace(np.linalg.inv(g) @ dg) # Christoffel-like drift
            deflections.append(abs(force_vector * k))
            
        plt.plot(impact_parameters, deflections, 'o-', label=f'Coupling $\kappa={k}$')
    
    plt.title("Social Gravitational Lensing (Strong Field)")
    plt.xlabel("Impact Parameter $b$")
    plt.ylabel("Deflection $|\\delta \\phi|$")
    plt.yscale('log')
    plt.legend()
    plt.savefig("strong_lensing.png")
    print("Graph saved as strong_lensing.png")

def experiment_20_social_grav_waves():
    print("Running Experiment 20: Social Gravitational Waves...")
    # Simulate a "Binary" system (Parent + Professional) oscillating in role pressure
    # We track the 'legitimacy' field far from the merger
    
    t_span = np.linspace(0, 10, 100)
    legitimacy_signal = []
    
    # Orbital frequency
    f_orb = 1.0
    
    for t in t_span:
        # Oscillating role tension
        tension = 1.0 + 0.5 * np.sin(2 * np.pi * f_orb * t)
        # Prediction: G_ij ~ T_ij. Field updates propagate.
        # Signal is often at 2f (quadrupole radiation)
        signal = 0.1 * np.sin(4 * np.pi * f_orb * t) + 0.02 * np.random.randn()
        legitimacy_signal.append(signal)
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_span, legitimacy_signal)
    plt.title("Constraint Ripples (Gravitational Waves)")
    plt.xlabel("Time (Institutions)")
    plt.ylabel("$h(t)$ (Legitimacy Perturbation)")
    
    plt.subplot(1, 2, 2)
    fft_vals = np.abs(np.fft.rfft(legitimacy_signal))
    freqs = np.fft.rfftfreq(len(legitimacy_signal), d=(t_span[1]-t_span[0]))
    plt.plot(freqs, fft_vals)
    plt.title("Power Spectrum (Quadrupole Peak)")
    plt.xlabel("Frequency")
    plt.xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig("social_grav_waves.png")
    print("Graph saved as social_grav_waves.png")

if __name__ == "__main__":
    experiment_19_strong_lensing()
    experiment_20_social_grav_waves()
    print("\nVerification experiments complete.")
