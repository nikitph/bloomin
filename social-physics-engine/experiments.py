import numpy as np
import matplotlib.pyplot as plt
import copy
from engine import SocialPhysicsEngine, Action
import random

def generate_random_action(agent_id, num_agents):
    action_types = ['report_fraud', 'stay_silent', 'anonymous_tip', 'lie', 'work_overtime', 'leave_child_alone']
    symbols = ['casual', 'uniform', 'robe', 'white_coat']
    contexts = ['office', 'legal', 'secret', 'street', 'courtroom', 'hospital']
    
    target_id = f"agent_{np.random.randint(num_agents)}" if np.random.rand() > 0.5 else None
    
    return Action(
        actor_id=agent_id,
        action_type=np.random.choice(action_types),
        target_id=target_id,
        magnitude=np.random.uniform(0, 1),
        context=np.random.choice(contexts),
        symbolic_marker=np.random.choice(symbols)
    )

def experiment_1_critical_slowing():
    print("Running Experiment 1: Critical Slowing Down...")
    violation_rates = np.linspace(0.1, 0.9, 10)
    response_times = []
    
    for rate in violation_rates:
        engine = SocialPhysicsEngine(num_agents=20)
        # Goal: Measure steps to reach strength > 0.8 for a specific rule
        t = 0
        stabilized = False
        while t < 200:
            # We use 'shun' as the target rule to stabilize
            # Context for shun in InformalInstitution is (actor_A.status > actor_B.status, 'shun')
            # Let's force this context
            if np.random.rand() < rate:
                # Agent 1 (high status?) vs Agent 0
                # Initialize status to ensure context
                engine.agents['agent_1'].status = 0.9
                engine.agents['agent_0'].status = 0.1
                
                action = Action(
                    actor_id='agent_1',
                    action_type='shun',
                    target_id='agent_0',
                    magnitude=1.0,
                    context='office',
                    symbolic_marker='casual'
                )
                # Success rate of violation affects strength growth
                # If outcome is 'success' (violation worked/aligned), strength increases
                engine.execute_action('agent_1', action, {}, 'success')
            
            # Check if any rule in informal_rules became strong
            max_strength = 0
            for rule in engine.informal_rules.implicit_rules.values():
                max_strength = max(max_strength, rule['strength'])
            
            if max_strength > 0.8:
                stabilized = True
                break
            t += 1
        
        response_times.append(t)
    
    plt.figure(figsize=(8, 5))
    plt.plot(violation_rates, response_times, 'o-', color='blue')
    plt.title("Critical Slowing Down: Stabilization Time vs Violation Rate")
    plt.xlabel("Violation Rate (Signal Frequency)")
    plt.ylabel("Steps to Stabilize Informal Rule")
    plt.grid(True)
    plt.savefig("critical_slowing.png")
    print("Graph saved as critical_slowing.png")

def experiment_2_legitimacy_collapse():
    print("Running Experiment 2: Legitimacy Collapse Cascade...")
    engine = SocialPhysicsEngine(num_agents=20)
    damage_history = []
    trust_history = []
    
    for step in range(50):
        before = engine.legitimacy_field.trust_score
        
        # Consistent small violations
        action = Action(
            actor_id='agent_0',
            action_type='lie',
            magnitude=0.1,
            context='office'
        )
        engine.execute_action('agent_0', action, {}, 'violation')
        
        after = engine.legitimacy_field.trust_score
        damage = before - after
        damage_history.append(damage)
        trust_history.append(after)
        
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(trust_history, color='red')
    plt.title("Legitimacy Decay")
    plt.ylabel("Trust Score")
    
    plt.subplot(2, 1, 2)
    plt.plot(damage_history, color='orange')
    plt.title("Marginal Trust Damage (Non-Linearity Check)")
    plt.xlabel("Step")
    plt.ylabel("Delta Trust")
    
    plt.tight_layout()
    plt.savefig("legitimacy_collapse.png")
    print("Graph saved as legitimacy_collapse.png")

def experiment_3_symbolic_resonance():
    print("Running Experiment 3: Symbolic Power Resonance...")
    engine = SocialPhysicsEngine(num_agents=2)
    symbols = ['casual', 'uniform', 'robe', 'white_coat']
    contexts = ['office', 'street', 'courtroom', 'hospital']
    
    amplification_matrix = np.zeros((len(symbols), len(contexts)))
    
    for i, symbol in enumerate(symbols):
        for j, context in enumerate(contexts):
            action = Action(
                actor_id='agent_0',
                action_type='command',
                magnitude=1.0,
                context=context,
                symbolic_marker=symbol
            )
            # Evaluate evaluation cost or authority
            eval_res = engine.evaluate_action('agent_0', action, {})
            amplification_matrix[i, j] = eval_res['system_states']['symbolic_authority']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(amplification_matrix, cmap='YlOrRd')
    plt.colorbar(label='Authority Multiplier')
    plt.xticks(range(len(contexts)), contexts)
    plt.yticks(range(len(symbols)), symbols)
    plt.title("Symbolic Resonance Heatmap")
    plt.xlabel("Context")
    plt.ylabel("Symbol")
    plt.savefig("symbolic_resonance.png")
    print("Graph saved as symbolic_resonance.png")

def experiment_4_tragic_choice_scaling():
    print("Running Experiment 4: Tragic Choice Frequency...")
    num_agents_list = [5, 10, 20, 50, 100]
    tragedy_rates = []
    
    # State that might trigger role conflict
    # We need a state that challenges multiple boundaries
    state = {
        'child_safety_risk': 0.8,
        'time_until_deadline': 1.0,
    }
    
    for n in num_agents_list:
        engine = SocialPhysicsEngine(n)
        tragedies = 0
        num_trials = 200
        for _ in range(num_trials):
            agent_id = f'agent_{np.random.randint(n)}'
            # Give random agent multiple roles to increase tragedy chance
            engine.agents[agent_id].roles = ['parent', 'professional', 'citizen']
            
            # Action that might trigger conflict: work_overtime vs leave_child_alone
            # Let's try something neutral and see if random roles trigger it
            action_type = np.random.choice(['work_overtime', 'leave_child_alone', 'steal'])
            action = Action(actor_id=agent_id, action_type=action_type)
            
            eval_res = engine.evaluate_action(agent_id, action, state)
            if eval_res['tragic_choice']:
                tragedies += 1
        
        tragedy_rates.append(tragedies / num_trials)
    
    plt.figure(figsize=(8, 5))
    plt.plot(num_agents_list, tragedy_rates, 's-', color='green')
    plt.title("Frequency of Tragic Choices vs System Size")
    plt.xlabel("Number of Agents")
    plt.ylabel("Tragic Choice Rate")
    plt.grid(True)
    plt.savefig("tragic_choice_scaling.png")
    print("Graph saved as tragic_choice_scaling.png")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    experiment_1_critical_slowing()
    experiment_2_legitimacy_collapse()
    experiment_3_symbolic_resonance()
    experiment_4_tragic_choice_scaling()
    
    print("\nAll experiments completed successfully.")
