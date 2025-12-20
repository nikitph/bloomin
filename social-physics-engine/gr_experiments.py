import numpy as np
import matplotlib.pyplot as plt
from engine import SocialPhysicsEngine, Action
import random
from scipy.optimize import curve_fit

def measure_riemann_curvature(engine, agent_id, state):
    """
    Measures curvature by parallel transport around a loop in state space.
    Loop: +X, +Y, -X, -Y
    """
    v1 = {'child_safety_risk': 0.1}
    v2 = {'time_until_deadline': -0.1}
    
    # Path A: v1 then v2
    s1 = engine.geodesic_step(agent_id, state, v1)
    s12 = engine.geodesic_step(agent_id, s1, v2)
    
    # Path B: v2 then v1
    s2 = engine.geodesic_step(agent_id, state, v2)
    s21 = engine.geodesic_step(agent_id, s2, v1)
    
    # Curvature is the Euclidean distance between final states in simplified metric
    d = abs(s12['child_safety_risk'] - s21['child_safety_risk']) + \
        abs(s12['time_until_deadline'] - s21['time_until_deadline'])
    return d * 100 # Scaling for visibility

def experiment_10_curvature_mapping():
    print("Running Experiment 10: Curvature Mapping...")
    engine = SocialPhysicsEngine(num_agents=1)
    agent_id = 'agent_0'
    engine.agents[agent_id].roles = ['parent', 'professional']
    
    risks = np.linspace(0.3, 0.7, 20)
    deadlines = np.linspace(1.5, 2.5, 20)
    
    heatmap = np.zeros((len(risks), len(deadlines)))
    
    for i, r in enumerate(risks):
        for j, d in enumerate(deadlines):
            state = {'child_safety_risk': r, 'time_until_deadline': d}
            heatmap[i, j] = measure_riemann_curvature(engine, agent_id, state)
            
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='magma', extent=[1.5, 2.5, 0.3, 0.7], origin='lower')
    plt.colorbar(label='Riemann Curvature (Tension)')
    plt.title("Curvature Map: Parent vs Professional Tension")
    plt.xlabel("Time Until Deadline")
    plt.ylabel("Child Safety Risk")
    plt.savefig("curvature_map.png")
    print("Graph saved as curvature_map.png")

def experiment_11_geodesic_completeness():
    print("Running Experiment 11: Geodesic Completeness Test...")
    engine = SocialPhysicsEngine(num_agents=1)
    agent_id = 'agent_0'
    agent = engine.agents[agent_id]
    agent.roles = ['parent', 'professional']
    
    # Scenario: High risk, starting geodesic toward "safety"
    # But wait, toward safety for one role might be disaster for another
    state = {'child_safety_risk': 0.6, 'time_until_deadline': 2.5}
    direction = {'child_safety_risk': 0.1, 'time_until_deadline': -0.1} # Pushing deeper into conflict
    
    path = []
    incompleteness_point = None
    
    for t in range(20):
        # In a real completeness test, we check if any valid action remains
        # or if the cost diverged to infinity
        eval_res_save = engine.evaluate_action(agent_id, Action(agent_id, 'save_child'), state)
        eval_res_work = engine.evaluate_action(agent_id, Action(agent_id, 'finish_work'), state)
        
        avg_cost = (eval_res_save['total_cost'] + eval_res_work['total_cost']) / 2
        path.append(avg_cost)
        
        if eval_res_save['tragic_choice'] and eval_res_work['tragic_choice']:
            if incompleteness_point is None:
                incompleteness_point = t
        
        state = engine.geodesic_step(agent_id, state, direction)

    plt.figure(figsize=(8, 5))
    plt.plot(path, color='purple', lw=2)
    if incompleteness_point:
         plt.axvline(incompleteness_point, color='red', linestyle='--', label='Singularity (Hole)')
    plt.title("Geodesic Cost Divergence (Incompleteness Test)")
    plt.xlabel("Steps along Geodesic")
    plt.ylabel("Average Action Cost")
    plt.legend()
    plt.savefig("geodesic_completeness.png")
    print("Graph saved as geodesic_completeness.png")

def power_law(x, a, b):
    return a * np.power(x, b)

def experiment_12_coupling_sweep():
    print("Running Experiment 12: Coupling Sweep for Universality...")
    couplings = [0.1, 0.2, 0.3, 0.4, 0.5]
    nus = []
    gammas = []
    
    p_c = 0.4
    violation_rates = np.linspace(0.45, 0.6, 5)

    for c in couplings:
        print(f"  Testing coupling: {c}...")
        results_nu = []
        results_gamma = []
        for p in violation_rates:
            engine = SocialPhysicsEngine(num_agents=20)
            engine.collective_field.coupling_strength = c
            
            # Simplified measurement
            # Correlation proxy: variance in emotions
            engine.agents['agent_0'].emotional_state = -1.0
            for _ in range(3): engine.step(dt=1.0)
            xi = np.std([a.emotional_state for a in engine.agents.values()]) * 10
            chi = abs(engine.collective_field.global_mood) / (p + 0.01)
            
            results_nu.append(xi)
            results_gamma.append(chi)
            
        xdata = np.abs(violation_rates - p_c)
        try:
            popt_nu, _ = curve_fit(power_law, xdata, results_nu)
            nus.append(-popt_nu[1])
        except: nus.append(0.0)
        
        try:
            popt_gamma, _ = curve_fit(power_law, xdata, results_gamma)
            gammas.append(-popt_gamma[1])
        except: gammas.append(0.0)

    plt.figure(figsize=(8, 5))
    plt.plot(couplings, nus, 'o-', label=r'Critical Exponent $\nu$ (Correlation)')
    plt.plot(couplings, gammas, 's-', label=r'Critical Exponent $\gamma$ (Susceptibility)')
    plt.axhline(1.0, color='gray', ls='--', label='2D Ising Goal (v=1.0)')
    plt.axhline(1.75, color='black', ls='--', label='2D Ising Goal (g=1.75)')
    plt.title("Exponent Convergence vs Coupling Strength")
    plt.xlabel("Coupling Strength (c)")
    plt.ylabel("Exponent Value")
    plt.legend()
    plt.savefig("coupling_sweep.png")
    print("Graph saved as coupling_sweep.png")

if __name__ == "__main__":
    np.random.seed(42)
    experiment_10_curvature_mapping()
    experiment_11_geodesic_completeness()
    experiment_12_coupling_sweep()
    print("\nSocial General Relativity experiments complete.")
