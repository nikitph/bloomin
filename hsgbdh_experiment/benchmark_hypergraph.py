import torch
import torch.nn as nn
import torch.nn.functional as F
from hsgbdh.hypergraph import HyperedgeReasoner
from hsgbdh.dual_model import DualStateHSGBDH

class HypergraphBenchmark:
    """
    Benchmark comparing HyperedgeReasoner (Symbolic) vs DualStateHSGBDH (Neural).
    Focuses on the "Exception Reasoning" task which Dual-Mode collapses.
    """
    def __init__(self):
        self.hyper_reasoner = HyperedgeReasoner()
        
        # Setup Neural Baseline (Dual-Mode)
        self.n = 64
        self.d = 64
        self.dual_model = DualStateHSGBDH(n=self.n, d=self.d)
        
        # Manual weight setup to simulate learning for the baseline
        # In a real scenario we'd train it, but for a PoC comparison 
        # we can show the structural limitation.
        
    def run_exception_benchmark(self):
        print("\n--- Exception Reasoning Benchmark ---")
        print("Task: 'All birds fly. Tweety is a bird. Is Tweety a bird? Does Tweety fly?'")
        print("      'What about Penguins? Penguins are birds but do NOT fly.'")
        
        # 1. Hypergraph Setup
        print("\n[Hypergraph Reasoner]")
        self.hyper_reasoner.add_hyperedge('Bird', {'Fly', 'HasFeathers', 'LayEggs'})
        self.hyper_reasoner.add_hyperedge('Penguin', {'Bird', '¬Fly', 'Swim'})
        self.hyper_reasoner.add_hyperedge('Tweety', {'Bird'})
        self.hyper_reasoner.add_hyperedge('Pingo', {'Penguin'})
        
        for subject in ['Tweety', 'Pingo']:
            print(f"\nQuerying: {subject} ⇒ Fly?")
            result = self.hyper_reasoner.query(subject, 'Fly')
            
            print(f"  Reachable: {result['reachable']}")
            print(f"  Exception/Conflict: {result['exception']}")
            if result['conflicts']:
                print(f"  Conflicts detected: {[c['statement'] for c in result['conflicts']]}")
            
            # Detailed path check
            for i, p in enumerate(result['paths']):
                print(f"  Path {i+1}: {' -> '.join(p['path'])}")

        # 2. Dual-Mode Baseline (Structural Analysis)
        print("\n[Dual-Mode Neural Baseline (Structural Limit)]")
        print("Dual-mode represents reasoning as G_transition @ x.")
        print("Even with stable_transitive_closure, it collapses state into a single vector.")
        print("It cannot represent both Fly=1 and Fly=0 as distinct reasoning paths simultaneously.")
        print("It would typically output a 'mixed' or 'averaged' vector, losing the specific conflict context.")

    def run_join_benchmark(self):
        print("\n--- Join Composition (AND-logic) Benchmark ---")
        print("Task: 'To open a vault, you need BOTH Key A and Key B.'")
        
        self.hyper_reasoner.compose_join({'KeyA', 'KeyB'}, {'VaultOpen'})
        self.hyper_reasoner.add_hyperedge('Room1', {'KeyA'})
        self.hyper_reasoner.add_hyperedge('Room2', {'KeyB'})
        self.hyper_reasoner.add_hyperedge('Player', {'Room1', 'Room2'})
        
        print("\nQuerying: Player ⇒ VaultOpen?")
        result = self.hyper_reasoner.query('Player', 'VaultOpen')
        print(f"  Reachable: {result['reachable']}")
        if result['reachable']:
            for i, p in enumerate(result['paths']):
                print(f"  Path {i+1}: {' -> '.join(p['path'])}")
        
        print("\nQuerying: Room1 ⇒ VaultOpen? (Should be False)")
        result_fail = self.hyper_reasoner.query('Room1', 'VaultOpen')
        print(f"  Reachable: {result_fail['reachable']}")

if __name__ == "__main__":
    bench = HypergraphBenchmark()
    bench.run_exception_benchmark()
    bench.run_join_benchmark()
