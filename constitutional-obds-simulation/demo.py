from core import State, Identity, Agent, AuthorityZoneNorm, Court
from loops import step_constitutional, step_gnn, step_transformer, authority_zone
import random

def run_experiment(name, sim_step, steps=100):
    print(f"\n--- Running Experiment: {name} ---")
    agent = Agent(
        state=State(0, 0),
        identity=Identity.CITIZEN,
        violations=[]
    )

    for t in range(steps):
        sim_step(agent)
        if authority_zone.inside(agent.state):
            print(f"FAILED: ENTERED FORBIDDEN ZONE at t={t}, state=({agent.state.x:.2f}, {agent.state.y:.2f})")
            return False

    print("SUCCESS: NEVER ENTERED FORBIDDEN ZONE")
    return True

if __name__ == "__main__":
    # Setup for Constitutional OBDS
    norms = [authority_zone]
    institutions = [Court()]

    # Run side-by-side
    random.seed(42) # For reproducibility in Transformer
    
    results = []
    
    # 1. OBDS
    results.append(run_experiment(
        "Constitutional OBDS", 
        lambda a: step_constitutional(a, norms, institutions)
    ))

    # 2. GNN
    # We'll run it with a penalty that eventually gets overwhelmed by the gradient
    results.append(run_experiment(
        "GNN (Soft Penalty)", 
        lambda a: step_gnn(a, norm_penalty=0.5) 
    ))

    # 3. Transformer
    results.append(run_experiment(
        "Transformer (Probabilistic)", 
        lambda a: step_transformer(a, compliance_prob=0.9)
    ))

    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    names = ["Constitutional OBDS", "GNN (Soft Penalty)", "Transformer (Probabilistic)"]
    for name, success in zip(names, results):
        status = "PASSED (Zero Violations)" if success else "FAILED (Violations Occurred)"
        why = "Hard projection in control flow" if "Constitutional" in name else "Governance is empirical/learned"
        print(f"{name:30} | {status:25} | {why}")
    print("="*40)
