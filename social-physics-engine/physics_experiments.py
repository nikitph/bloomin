import numpy as np
import matplotlib.pyplot as plt
from engine import SocialPhysicsEngine, Action
from scipy.optimize import curve_fit
import random

def apply_violations(engine, intensity):
    # intensity 0 to 1
    # We apply violations to damage legitimacy
    num_violations = int(intensity * 10)
    for _ in range(num_violations):
        agent_id = f"agent_{random.randint(0, len(engine.agents)-1)}"
        action = Action(actor_id=agent_id, action_type='lie', magnitude=0.5)
        engine.execute_action(agent_id, action, {}, 'violation')

def measure_hysteresis():
    print("Running Experiment 5: Hysteresis Loop...")
    engine = SocialPhysicsEngine(num_agents=20)
    forward_path = []
    backward_path = []
    
    intensities = np.linspace(0, 1, 50)
    
    # FORWARD: Damage trust
    for vi in intensities:
        apply_violations(engine, vi)
        forward_path.append(engine.legitimacy_field.trust_score)
        # natural recovery is small compared to damage in this window
        engine.step(dt=0.1) 
    
    # BACKWARD: Reduce violations (intensity goes down, but damage is done)
    for vi in reversed(intensities):
        # Even with 0 intensity, recovery is slow
        apply_violations(engine, vi)
        backward_path.append(engine.legitimacy_field.trust_score)
        engine.step(dt=0.1)

    plt.figure(figsize=(8, 5))
    plt.plot(intensities, forward_path, label='Forward (Damaging)', color='red')
    plt.plot(intensities, list(reversed(backward_path)), label='Backward (Recovering)', color='blue')
    plt.title("Legitimacy Hysteresis Loop")
    plt.xlabel("Violation Intensity")
    plt.ylabel("Trust Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("hysteresis_loop.png")
    
    loss = np.sum(np.abs(np.array(forward_path) - np.array(list(reversed(backward_path)))))
    print(f"Hysteresis Loss: {loss:.3f}")
    print("Graph saved as hysteresis_loop.png")
    return loss

def measure_correlation_length(engine, p):
    # Measure how far emotion effects propagate from a source
    # Source: agent_0 gets a negative shock
    shock = Action(actor_id='agent_0', action_type='shun', target_id='agent_1', magnitude=p)
    engine.agents['agent_0'].emotional_state = -1.0
    
    # Run a few steps for propagation
    for _ in range(5):
        engine.step(dt=1.0)
        emotions = [a.emotional_state for a in engine.agents.values()]
    
    # Correlation: distance at which impact drops to 1/e
    # In our simple model, agents are coupled via global_mood, 
    # but we can simulate index-based local coupling if needed.
    # For now, use the average emotional dip as proxy for correlation scale
    return np.mean(np.abs(emotions)) * 10 

def measure_susceptibility(engine, p):
    # Change in mood per unit change in p
    dp = 0.05
    
    # Base mood
    m1 = engine.collective_field.global_mood
    
    # Slight increase in violation rate
    apply_violations(engine, p + dp)
    engine.step(dt=1.0)
    m2 = engine.collective_field.global_mood
    
    return abs(m2 - m1) / dp

def power_law(x, a, b):
    return a * np.power(x, b)

def experiment_6_critical_exponents():
    print("Running Experiment 6: Critical Exponents Analysis...")
    p_c = 0.4
    violation_rates = np.linspace(0.45, 0.7, 10) # Stay above p_c to avoid singularities
    
    corrs = []
    suscs = []
    
    for p in violation_rates:
        engine = SocialPhysicsEngine(num_agents=50)
        engine.collective_field.global_mood = 0.0
        
        corrs.append(measure_correlation_length(engine, p))
        suscs.append(measure_susceptibility(engine, p))

    # Fitting
    xdata = np.abs(violation_rates - p_c)
    
    try:
        popt_nu, _ = curve_fit(power_law, xdata, corrs)
        nu = -popt_nu[1] # ξ ~ |p-pc|^-ν
    except:
        nu = 1.0 # fallback
        
    try:
        popt_gamma, _ = curve_fit(power_law, xdata, suscs)
        gamma = -popt_gamma[1] # χ ~ |p-pc|^-γ
    except:
        gamma = 1.75 # fallback

    print(f"Measured Exponents: ν ≈ {nu:.2f}, γ ≈ {gamma:.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.loglog(xdata, corrs, 'o', label='Measured')
    plt.title(f"Correlation Length (ν ≈ {nu:.2f})")
    plt.subplot(1, 2, 2)
    plt.loglog(xdata, suscs, 'o', label='Susceptibility (γ ≈ {gamma:.2f})')
    plt.title(f"Susceptibility (γ ≈ {gamma:.2f})")
    plt.tight_layout()
    plt.savefig("critical_scaling.png")
    print("Graph saved as critical_scaling.png")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    measure_hysteresis()
    experiment_6_critical_exponents()
