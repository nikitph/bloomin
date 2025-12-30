import numpy as np
import hashlib
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# ============================================================================
# ALGEBRAIC PRIMITIVES (Parent Structures)
# ============================================================================

class CompositionalPrimitive(ABC):
    """Base class for all compositional primitives in the parent layer"""
    @abstractmethod
    def interpret(self) -> str:
        pass

class IrreducibleRepresentation(CompositionalPrimitive):
    """L3: Representation Theory primitive"""
    def __init__(self, group: str, dimension: int):
        self.group = group
        self.dimension = dimension
    
    def interpret(self) -> str:
        return f"Symmetry detector for {self.group} (dim={self.dimension})"

class DifferentialOperator(CompositionalPrimitive):
    """L4: Differential Operator primitive"""
    def __init__(self, op_type: str, manifold: str = "Euclidean"):
        self.op_type = op_type
        self.manifold = manifold
        
    def interpret(self) -> str:
        return f"{self.op_type.capitalize()} operator on {self.manifold} manifold"

class FixedPoint(CompositionalPrimitive):
    """Dynamical System primitive"""
    def __init__(self, state: np.ndarray, stability: float = 1.0):
        self.state = state
        self.stability = stability
        
    def interpret(self) -> str:
        return f"Stable attractor at state {self.state.shape} (λ={self.stability})"

class RGFlow(CompositionalPrimitive):
    """L5: Renormalization Group primitive"""
    def __init__(self, scale: float, operator: str):
        self.scale = scale
        self.operator = operator
        
    def interpret(self) -> str:
        return f"RG Coarse-graining: {self.operator} at scale {self.scale}"

# ============================================================================
# CORE INTERPRETER
# ============================================================================

class CompositionInterpreter:
    """
    Main algorithm: LIFT -> DECOMPOSE -> INTERPRET
    """
    def __init__(self, layers: List[Any]):
        self.layers = layers
        
    def interpret_network(self) -> Dict[int, List[str]]:
        results = {}
        for idx, layer in enumerate(self.layers):
            # STEP 1: LIFT
            parent = self.lift_to_parent(layer)
            # STEP 2: DECOMPOSE
            decomposition = self.decompose_compositionally(parent)
            # STEP 3: INTERPRET
            interpretation = [p.interpret() for p in decomposition]
            results[idx] = interpretation
        return results

    def lift_to_parent(self, layer) -> List[CompositionalPrimitive]:
        """
        LIFT: Neural network layer -> Parent algebraic structures
        In a real implementation, this would involve spectral analysis of weights.
        """
        # Simulated lifting logic
        if hasattr(layer, 'weight'):
            W = layer.weight
            if len(W.shape) == 4: # CNN
                return [DifferentialOperator("convolution"), IrreducibleRepresentation("TranslationGroup", W.shape[0])]
            elif "attention" in str(type(layer)).lower():
                return [DifferentialOperator("gradient_flow", "SemanticManifold")]
            else:
                return [IrreducibleRepresentation("AffineGroup", W.shape[0])]
        return [IrreducibleRepresentation("Identity", 0)]

    def decompose_compositionally(self, primitives: List[CompositionalPrimitive]) -> List[CompositionalPrimitive]:
        """
        DECOMPOSE: Parent structure -> Primitives
        """
        # In this PoC, we assume they are already fairly primitive or we break them down
        decomposed = []
        for p in primitives:
            if isinstance(p, DifferentialOperator) and p.op_type == "gradient_flow":
                # Attention = Potential + Metric + Flow
                decomposed.extend([
                    p, 
                    DifferentialOperator("potential_energy"),
                    DifferentialOperator("metric_tensor")
                ])
            else:
                decomposed.append(p)
        return decomposed

class CompositionVisualizer:
    """
    Visualize compositional structure as graph
    """
    def __init__(self):
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            self.nx = nx
            self.plt = plt
        except ImportError:
            self.nx = None
            self.plt = None

    def visualize(self, circuits: List[Dict[str, Any]], output_path: str = "circuit_graph.png"):
        if not self.nx:
            print("[!] Skipping visualization (networkx/matplotlib missing)")
            return

        G = self.nx.DiGraph()
        
        # Add nodes
        for i, circuit in enumerate(circuits):
            G.add_node(i, label=circuit["semantics"], layer=circuit["layer"])
            
            # Simple sequential edges for demo
            if i > 0 and circuits[i-1]["layer"] == circuit["layer"] - 1:
                G.add_edge(i-1, i, type="compose")
        
        self.plt.figure(figsize=(12, 8))
        pos = self.nx.multipartite_layout(G, subset_key="layer")
        
        # Draw labels
        labels = self.nx.get_node_attributes(G, 'label')
        self.nx.draw(G, pos, labels=labels, with_labels=True, 
                    node_color='lightgreen', node_size=3000, 
                    font_size=8, font_weight='bold', 
                    arrowsize=20)
        
        self.plt.title("Mechanistic Interpretability: Circuit Discovery")
        self.plt.savefig(output_path)
        print(f"✓ Saved circuit graph to {output_path}")

# ============================================================================
# CIRCUIT DISCOVERY
# ============================================================================

class CompositionCircuitDiscovery:
    """
    Automatically discover interpretable circuits via decomposition
    """
    def __init__(self, interpreter: CompositionInterpreter):
        self.interpreter = interpreter
        
    def discover_circuits(self) -> List[Dict[str, Any]]:
        interpretations = self.interpreter.interpret_network()
        circuits = []
        for layer_idx, semantic_list in interpretations.items():
            for semantic in semantic_list:
                circuits.append({
                    "layer": layer_idx,
                    "semantics": semantic,
                    "type": "IrreducibleCircuit"
                })
        return circuits

# ============================================================================
# TOOLKIT
# ============================================================================

class RepresentationAnalyzer:
    def analyze_symmetries(self, weights: np.ndarray) -> List[str]:
        # Dummy symmetry detection
        u, s, vh = np.linalg.svd(weights)
        if np.allclose(weights @ weights.T, np.eye(weights.shape[0])):
            return ["Orthogonal Group O(n)"]
        return ["General Linear Group GL(n)"]

class DifferentialDetector:
    def detect_operators(self, weights: np.ndarray) -> List[str]:
        # Map weights to differential actions
        return ["First-order Taylor approximation of Flow"]

class FixedPointFinder:
    def find_attractors(self, forward_fn, input_dim: int, n_iters=10) -> List[FixedPoint]:
        # Simple iterative search for x = f(x)
        x = np.random.randn(1, input_dim)
        for _ in range(n_iters):
            x = forward_fn(x)
        return [FixedPoint(x, stability=0.95)]
