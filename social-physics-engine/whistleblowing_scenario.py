import numpy as np
from engine import SocialPhysicsEngine, Action
import random

# Step 1: Initialize System
engine = SocialPhysicsEngine(num_agents=20)
whistleblower = engine.agents['agent_0']
whistleblower.roles = ['professional', 'citizen', 'parent']
whistleblower.status = 0.3
engine.collective_field.global_mood = -0.4

# Step 2: Define Dilemma State
state = {
    'fraud_severity': 0.9,
    'child_safety_risk': 0.2,
    'time_until_deadline': 5.0,
    'legal_protection': 0.6,
    'office_culture_strength': 0.8
}

# Step 3: Define Action Options
actions = [
    Action(actor_id='agent_0', action_type='report_fraud', target_id='agent_19', magnitude=0.9, context='legal', symbolic_marker='casual'),
    Action(actor_id='agent_0', action_type='stay_silent', target_id=None, magnitude=0.0, context='office', symbolic_marker='casual'),
    Action(actor_id='agent_0', action_type='anonymous_tip', target_id='external', magnitude=0.6, context='secret', symbolic_marker='casual')
]

# Step 4: Evaluate Each Action
for action in actions:
    print(f"\n{'='*60}")
    print(f"EVALUATING: {action.action_type}")
    print(f"{'='*60}")
    evaluation = engine.evaluate_action('agent_0', action, state)
    print(f"\nTotal Cost: {evaluation['total_cost']:.3f}")
    print(f"Tragic Choice: {evaluation['tragic_choice']}")
    if evaluation['violations']:
        print("\nRole Violations:")
        for role, violation in evaluation['violations']:
            print(f"  - {role}: {violation:.3f}")
    print("\nSystem States:")
    for system, value in evaluation['system_states'].items():
        print(f"  {system}: {value:.3f}")

# Step 5: Simulate Action Execution
chosen_action, min_cost = min(
    [(a, engine.evaluate_action('agent_0', a, state)['total_cost']) for a in actions],
    key=lambda x: x[1]
)
print(f"\nCHOSEN ACTION: {chosen_action.action_type} (Cost: {min_cost:.3f})")
actual_outcome = np.random.choice(['success', 'disaster', 'mixed'], p=[0.3, 0.4, 0.3])
result = engine.execute_action('agent_0', chosen_action, state, actual_outcome)
print(f"OUTCOME: {actual_outcome}")

# Step 6: Observe System Evolution (10 days)
for day in range(10):
    engine.step(dt=1.0)
    for agent_id in ['agent_1', 'agent_2', 'agent_3']:
        reaction_type = 'gossip' if actual_outcome == 'success' else 'shun'
        reaction = Action(actor_id=agent_id, action_type=reaction_type, target_id='agent_0', magnitude=0.5, context='office', symbolic_marker='casual')
        engine.execute_action(agent_id, reaction, state, 'neutral')
    
    state_snapshot = engine.get_system_state()
    print(f"\nDay {day + 1}:")
    print(f"  Legitimacy: {state_snapshot['legitimacy']:.3f}")
    print(f"  Collective Mood: {state_snapshot['collective_mood']:.3f}")
    print(f"  Avg Guilt: {state_snapshot['avg_guilt']:.3f}")
    print(f"  Informal Rules Learned: {state_snapshot['informal_rules_count']}")

# Step 7: Analyze Emergent Patterns
print("\n" + "="*60)
print("EMERGENT INFORMAL INSTITUTIONS")
print("="*60)
for context, rule in engine.informal_rules.implicit_rules.items():
    if rule['strength'] > 0.5:
        print(f"\nContext: {context}")
        print(f"  Strength: {rule['strength']:.3f}")
        print(f"  Success Rate: {rule['success_rate']:.3f}")
        print(f"  Observations: {rule['count']}")

print("\n" + "="*60)
print("NORM ADAPTATIONS")
print("="*60)
for norm, boundary in engine.adaptive_boundaries.items():
    print(f"{norm}: threshold moved to {boundary.threshold:.3f}")

print("\n" + "="*60)
print("RETROSPECTIVE MORAL JUDGMENTS")
print("="*60)
for record in engine.judgment_system.action_history[-5:]:
    print(f"\nAction: {record['action'].action_type}")
    print(f"  Expected: {record['expected']}")
    print(f"  Actual: {record['actual']}")
    print(f"  Judgment: {record['judgment']:.3f}")
