import numpy as np
import matplotlib.pyplot as plt
from world import State, Agent, StateField
from models import ConstitutionalOBDS, GNNBaseline, TransformerBaseline
from scenario import get_scenario_components
import copy

def run_simulation(model_class, name, steps=50):
    field = StateField()
    norms, institutions = get_scenario_components()
    model = model_class(field, norms, institutions)
    
    # 2 agents: 1 Citizen, 1 Official
    agents = [
        Agent(id=1, identity="Citizen", state=State(position=np.array([0.0, 0.0]))),
        Agent(id=2, identity="Official", state=State(position=np.array([2.0, 2.0])))
    ]
    
    history = {a.id: [] for a in agents}
    violations_over_time = {a.id: [] for a in agents}

    for t in range(steps):
        model.step(agents)
        for a in agents:
            history[a.id].append(a.state.position.copy())
            violations_over_time[a.id].append(len(a.violations))

    return history, violations_over_time

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'ConstitutionalOBDS': 'blue', 'GNNBaseline': 'red', 'TransformerBaseline': 'green'}
    
    for name, (history, violations) in results.items():
        # Trajectories
        citizen_hist = np.array(history[1])
        official_hist = np.array(history[2])
        
        ax1.plot(citizen_hist[:, 0], citizen_hist[:, 1], label=f"{name} (Citizen)", color=colors[name], linestyle='-')
        ax1.plot(official_hist[:, 0], official_hist[:, 1], label=f"{name} (Official)", color=colors[name], linestyle='--')
        
        # Violations
        ax2.plot(violations[1], label=f"{name} (Citizen)", color=colors[name])

    # Draw Authority Zone
    circle = plt.Circle((10, 10), 5, color='gray', fill=False, linestyle=':', label='Authority Boundary')
    ax1.add_patch(circle)
    ax1.set_xlim(-1, 15)
    ax1.set_ylim(-1, 15)
    ax1.set_title("Trajectories (Citizen: solid, Official: dashed)")
    ax1.legend()

    ax2.set_title("Cumulative Violations (Citizen)")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Count")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("simulation_results.png")
    print("Results saved to simulation_results.png")

if __name__ == "__main__":
    regimes = {
        "ConstitutionalOBDS": ConstitutionalOBDS,
        "GNNBaseline": GNNBaseline,
        "TransformerBaseline": TransformerBaseline
    }
    
    all_results = {}
    for name, model_class in regimes.items():
        print(f"Running {name}...")
        all_results[name] = run_simulation(model_class, name)
        
    plot_results(all_results)
