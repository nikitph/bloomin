from engine import SocialPhysicsEngine, Action
import numpy as np

def run_critical_tragic_test():
    print("Running Critical Tragic Choice Test...")
    engine = SocialPhysicsEngine(num_agents=1)
    agent = engine.agents['agent_0']
    agent.roles = ['parent', 'professional', 'citizen']
    
    # Scenario: Provably no solution
    state = {
        'child_in_danger': True,
        'deadline_near': True,
        'law_requires_reporting': True,
        'choice_is_exclusive': True # Human-enforced in code below
    }
    
    # Define actions that satisfy exactly one role each
    actions = [
        Action(actor_id='agent_0', action_type='save_child', context='home'),
        Action(actor_id='agent_0', action_type='finish_work', context='office'),
        Action(actor_id='agent_0', action_type='report_crime', context='legal')
    ]
    
    all_tragic = True
    for action in actions:
        print(f"\nEvaluating: {action.action_type}")
        eval_res = engine.evaluate_action('agent_0', action, state)
        
        print(f"  Total Cost: {eval_res['total_cost']:.3f}")
        print(f"  Tragic Choice: {eval_res['tragic_choice']}")
        print(f"  Violations: {[v[0] for v in eval_res['violations']]}")
        
        if not eval_res['tragic_choice']:
            all_tragic = False
            print(f"  !! FAILED: {action.action_type} was NOT flagged as tragic.")
        else:
            print(f"  SUCCESS: Correctly flagged as tragic.")

    if all_tragic:
        print("\n" + "="*40)
        print("CRITICAL TEST PASSED: SYSTEM DETECTS STRUCTURAL IMPOSSIBILITY")
        print("="*40)
    else:
        print("\n" + "="*40)
        print("CRITICAL TEST FAILED: DETECTION LOGIC IS BROKEN")
        print("="*40)

if __name__ == "__main__":
    run_critical_tragic_test()
