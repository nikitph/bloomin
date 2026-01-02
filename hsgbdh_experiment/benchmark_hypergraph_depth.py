import torch
import time
import matplotlib.pyplot as plt
from hsgbdh.hypergraph import HyperedgeReasoner

def run_depth_benchmark():
    """
    Compare discrete hypergraph scaling vs neural dual-mode.
    Task: Sequential chain with 'Context Gate' (AND-logic) at every step.
    Chain: {A_i, Context_i} => A_{i+1}
    """
    print("--- Hypergraph Scaling Benchmark (L=20) ---")
    print("Scenario: A sequential reasoning chain where each step requires an external context/key.")
    
    reasoner = HyperedgeReasoner()
    
    # 1. Setup the chain up to L=20
    # Each step i -> i+1 requires Key_i
    max_len = 20
    for i in range(max_len):
        reasoner.compose_join({f'Node_{i}', f'Key_{i}'}, {f'Node_{i+1}'})
        # Add a way to get the key (simulating retrieval or observation)
        reasoner.add_hyperedge(f'Observe_{i}', {f'Key_{i}'})
        
    # 2. Query at different depths
    test_depths = [3, 5, 10, 15, 20]
    results = []
    
    print("\n[Executing Queries]")
    for depth in test_depths:
        # To reach Node_depth, we need Node_0 AND all keys from 0 to depth-1
        # We simulate this by starting with Node_0 and all 'Observe' triggers
        start_nodes = ['Node_0'] + [f'Observe_{j}' for j in range(depth)]
        
        # We need to adapt query to handle multiple starting nodes
        # Our current query(source, target) only takes one source.
        # Let's fix that in a wrapper or the class if needed.
        
        # For this benchmark, we'll create a 'SuperStart' node
        super_start = f'Start_{depth}'
        reasoner.add_hyperedge(super_start, set(start_nodes))
        
        start_time = time.time()
        result = reasoner.query(super_start, f'Node_{depth}')
        end_time = time.time()
        
        confidence = 1.0 if result['reachable'] else 0.0
        print(f"Depth {depth}: Confidence={confidence} (Time: {(end_time - start_time)*1000:.2f}ms)")
        results.append((depth, confidence))

    # 3. Visualization Comparison Data (Neural baseline from previous run)
    neural_depths = [3, 5, 10, 15, 20]
    neural_scores = [0.86, 0.56, 0.42, 0.50, 0.40] # Approximate from previous output
    
    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in results], [r[1] for r in results], 's--', label='Hypergraph (Symbolic)', color='blue')
    plt.plot(neural_depths, neural_scores, 'o-', label='Dual-Mode (Neural Baseline)', color='red')
    
    plt.title("Scaling Truth: Hypergraph vs Neural Dual-Mode")
    plt.xlabel("Chain Length (Depth)")
    plt.ylabel("Confidence/Accuracy")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('hypergraph_scaling_l20.png')
    print("\nPlot saved to hypergraph_scaling_l20.png")
    
    # 4. Stress Test (Structural Complexity)
    print("\n[Conflict Stress Test]")
    print("Introducing a conflict at Depth 10: 'Node_10 => ¬Node_11'")
    reasoner.add_hyperedge('Node_10', {'¬Node_11'})
    
    result_conflict = reasoner.query('Start_20', 'Node_20')
    print(f"Depth 20 query after conflict: reachable={result_conflict['reachable']}, exception={result_conflict['exception']}")
    if result_conflict['exception']:
        print(f"Conflicts found: {[c['statement'] for c in result_conflict['conflicts']]}")

if __name__ == "__main__":
    run_depth_benchmark()
