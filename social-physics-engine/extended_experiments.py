from engine import SocialPhysicsEngine, Action
import numpy as np

def experiment_7_moral_weight_sensitivity():
    print("\nExperiment 7: Moral Weight Sensitivity")
    # Sweep parent boundary threshold
    thresholds = [0.5, 0.8, 1.0, 1.5]
    
    state = {
        'child_in_danger': True,
        'deadline_near': True,
        'law_requires_reporting': True
    }
    
    for weight in thresholds:
        engine = SocialPhysicsEngine(num_agents=1)
        # Update weight
        engine.role_boundaries['parent'].threshold = weight
        engine.agents['agent_0'].roles = ['parent', 'professional', 'citizen']
        
        actions = [
            Action(actor_id='agent_0', action_type='save_child'),
            Action(actor_id='agent_0', action_type='finish_work'),
            Action(actor_id='agent_0', action_type='report_crime')
        ]
        
        results = []
        for action in actions:
            eval_res = engine.evaluate_action('agent_0', action, state)
            results.append((action.action_type, eval_res['total_cost']))
        
        # Sort by cost to find "least bad"
        results.sort(key=lambda x: x[1])
        print(f"Parent Weight {weight:.1f}: Least bad is '{results[0][0]}' (Cost: {results[0][1]:.3f})")
        for act, cost in results:
            print(f"  - {act}: {cost:.3f}")

def experiment_8_role_compatibility():
    print("\nExperiment 8: Role Compatibility Analysis")
    engine = SocialPhysicsEngine(num_agents=1)
    agent = engine.agents['agent_0']
    # Adding a 4th role
    agent.roles = ['parent', 'professional', 'citizen', 'friend']
    
    state = {
        'child_in_danger': True,
        'deadline_near': True,
        'law_requires_reporting': True,
        'friend_needs_help': True
    }
    
    # Define actions including a "compatible" one
    actions = [
        Action(actor_id='agent_0', action_type='save_child'),      # parent
        Action(actor_id='agent_0', action_type='finish_work'),     # professional
        Action(actor_id='agent_0', action_type='report_crime'),    # citizen
        Action(actor_id='agent_0', action_type='call_for_help')    # parent + friend (in our engine logic)
    ]
    
    # We need to ensure call_for_help satisfies both parent and friend
    # In engine.py, ProtectChildBoundary checks for ['save_child', 'stay_at_home']
    # and ProtectFriendBoundary checks for ['call_for_help', 'support_friend']
    # Wait, call_for_help should satisfy parent too to be "compatible"
    
    print("Evaluating actions for 4-role agent:")
    for action in actions:
        eval_res = engine.evaluate_action('agent_0', action, state)
        violations = [v[0] for v in eval_res['violations']]
        print(f"Action '{action.action_type}': Costs {eval_res['total_cost']:.3f}, Violations: {violations}, Tragic: {eval_res['tragic_choice']}")

def experiment_9_temporal_tragedy():
    print("\nExperiment 9: Temporal Tragedy Accumulation")
    engine = SocialPhysicsEngine(num_agents=1)
    agent = engine.agents['agent_0']
    agent.roles = ['parent', 'professional']
    
    state = {
        'child_in_danger': True,
        'deadline_near': True
    }
    
    cumulative_guilt = 0
    print("Simulating 10 steps of repeated tragic choices:")
    for t in range(5): # Doing 5 steps for brevity
        # Choose "least bad" action at each step
        actions = [
            Action(actor_id='agent_0', action_type='save_child'),
            Action(actor_id='agent_0', action_type='finish_work')
        ]
        
        evals = []
        for action in actions:
            eval_res = engine.evaluate_action('agent_0', action, state)
            evals.append((action, eval_res))
            
        evals.sort(key=lambda x: x[1]['total_cost'])
        best_action, best_eval = evals[0]
        
        # Execute it
        engine.execute_action('agent_0', best_action, state, 'success')
        
        # Observe guilt accumulation
        guilt = agent.internal_guilt
        print(f"Step {t+1}: Action '{best_action.action_type}', Guilt: {guilt:.3f}, Violations in eval: {[v[0] for v in best_eval['violations']]}")
        
        # Advance time - maybe risk persists
        engine.step(dt=1.0)

if __name__ == "__main__":
    experiment_7_moral_weight_sensitivity()
    experiment_8_role_compatibility()
    experiment_9_temporal_tragedy()
