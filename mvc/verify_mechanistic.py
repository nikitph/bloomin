import numpy as np
import json
from mvc.mechanistic import (
    CompositionInterpreter,
    CompositionCircuitDiscovery,
    CompositionVisualizer,
    FixedPointFinder,
    RepresentationAnalyzer
)
from mvc.interpretable_transformer import InterpretableTransformer

def run_validation():
    print("="*80)
    print("MECHANISTIC INTERPRETABILITY VALIDATION")
    print("="*80)

    # 1. Test Interpretable Transformer (Interpretability by Design)
    print("\n[1] Testing InterpretableTransformer (By Design)...")
    model = InterpretableTransformer(d_model=32, n_layers=2)
    x = np.random.randn(1, 32)
    _ = model.forward(x)
    
    interp = model.interpret()
    print("✓ Model Interpretation (Summary):")
    print(f"  Input:  {interp['input']}")
    print(f"  Output: {interp['output']}")
    print(f"  Body:   Found {len(interp['body'])} interpretable layers.")

    # 2. Test CompositionInterpreter (Post-hoc Lifting)
    print("\n[2] Testing CompositionInterpreter (Post-hoc Lifting)...")
    # Mock some layers
    class MockLayer:
        def __init__(self, name, shape):
            self.weight = np.random.randn(*shape)
            self.name = name
        def __repr__(self): return self.name

    layers = [
        MockLayer("Input_CNN", (8, 8, 3, 16)),
        MockLayer("Attention_Head", (32, 32)),
        MockLayer("Final_FC", (32, 10))
    ]
    # Patch the attention mock to have 'attention' in its type name
    class AttentionMock(MockLayer): pass
    layers[1] = AttentionMock("Attention_Head", (32, 32))

    interpreter = CompositionInterpreter(layers)
    post_hoc_interp = interpreter.interpret_network()
    print("✓ Post-hoc Lifting Results:")
    for idx, semantics in post_hoc_interp.items():
        print(f"  Layer {idx} ({layers[idx]}): {semantics}")

    # 3. Test Circuit Discovery and Visualization
    print("\n[3] Testing Circuit Discovery and Visualization...")
    discovery = CompositionCircuitDiscovery(interpreter)
    circuits = discovery.discover_circuits()
    print(f"✓ Discovered {len(circuits)} irreducible circuits.")
    
    visualizer = CompositionVisualizer()
    visualizer.visualize(circuits, output_path="/Users/truckx/PycharmProjects/bloomin/mvc/circuit_graph.png")

    # 4. Test Toolkit (Fixed Points & Symmetries)
    print("\n[4] Testing Toolkit (Dynamics & Symmetries)...")
    analyzer = RepresentationAnalyzer()
    symmetries = analyzer.analyze_symmetries(np.eye(32))
    print(f"✓ Symmetry Analysis (Identity matrix): {symmetries}")

    finder = FixedPointFinder()
    def mock_forward(x): return 0.5 * x + 0.1 # Simple stable system
    attractors = finder.find_attractors(mock_forward, input_dim=32)
    print(f"✓ Found {len(attractors)} attractors. Stability: {attractors[0].stability}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_validation()
