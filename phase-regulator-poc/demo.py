from orchestrator import PhaseAwareInference
import time

def run_demo():
    engine = PhaseAwareInference()
    
    queries = [
        # Factual repeated queries to trigger freezing
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        "The capital of France is",
        
        # Novel query
        "Write a poem about a robot drinking tea",
        
        # Shifted factual (to trigger melting if we are lucky with GPT-2's variance)
        "The capital of Germany is",
        "The capital of Germany is",
    ]
    
    print("\n--- PHASE REGULATOR POC DEMO ---\n")
    
    for i, q in enumerate(queries):
        print(f"[{i+1}] Query: '{q}'")
        answer, phase, latency, dO, dW = engine.answer_query(q)
        cluster = engine.get_cluster(q)
        state = engine.cluster_states[cluster]
        print(f"    Answer: {answer}")
        print(f"    Target Phase: {state.phase} (Stable Count: {state.stable_count})")
        print(f"    Routing: {phase}")
        print(f"    Latency: {latency:.4f}s")
        if dO > 0:
            print(f"    Metrics: dO={dO:.2f}, dW={dW:.2f}, H={state.H_hat:.3f}")
        print("-" * 30)

    # Thermal Maintenance test
    print("\nRunning Thermal Maintenance (Decay)...")
    count = engine.bloom.decay(time.time() + 100000) # Jump forward in time
    print(f"Active witnesses after decay: {count}")

if __name__ == "__main__":
    run_demo()
