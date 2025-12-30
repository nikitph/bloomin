import numpy as np
from typing import List, Dict, Any, Tuple
from mvc.mechanistic import (
    IrreducibleRepresentation, 
    DifferentialOperator, 
    RGFlow, 
    CompositionalPrimitive
)

class InterpretableLayer:
    """A layer that explicitly defines its compositional parents"""
    def __init__(self, weight: np.ndarray, parent_primitives: List[CompositionalPrimitive]):
        self.weight = weight
        self.parents = parent_primitives

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight

class InterpretableTransformer:
    """
    transformer designed from compositional primitives.
    interpretation is trivial because it's designed in.
    """
    def __init__(self, d_model: int, n_layers: int):
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers: List[InterpretableLayer] = []
        
        # Layer 1: Input Representations (L3)
        self.input_rep = InterpretableLayer(
            np.eye(d_model),
            [IrreducibleRepresentation("SO(3) x S_n", d_model)]
        )
        
        # Middle Layers: Differential Operators (L4)
        for _ in range(n_layers):
            self.layers.append(InterpretableLayer(
                np.random.randn(d_model, d_model) * 0.01 + np.eye(d_model),
                [
                    DifferentialOperator("gradient_flow", "SemanticManifold"),
                    DifferentialOperator("vector_field_evolution")
                ]
            ))
            
        # Output Layer: RG Coarse-graining (L5)
        self.output_head = InterpretableLayer(
            np.random.randn(d_model, 10),
            [RGFlow(scale=0.01, operator="Categorical attractor")]
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.input_rep.forward(x)
        for layer in self.layers:
            # Residual connection = Integral curve of vector field
            x = x + layer.forward(x)
        x = self.output_head.forward(x)
        return x

    def interpret(self) -> Dict[str, List[str]]:
        """
        Returns the human-readable semantics of each component
        """
        interpretations = {
            "input": [p.interpret() for p in self.input_rep.parents],
            "body": [],
            "output": [p.interpret() for p in self.output_head.parents]
        }
        
        for idx, layer in enumerate(self.layers):
            layer_semantics = [p.interpret() for p in layer.parents]
            interpretations["body"].append({f"layer_{idx}": layer_semantics})
            
        return interpretations

# Example Usage
if __name__ == "__main__":
    model = InterpretableTransformer(d_model=64, n_layers=2)
    x = np.random.randn(1, 64)
    out = model.forward(x)
    print("Model Output Shape:", out.shape)
    
    interp = model.interpret()
    import json
    print("\nModel Interpretation:")
    print(json.dumps(interp, indent=2))
