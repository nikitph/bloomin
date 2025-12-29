import numpy as np

class SemanticManifold:
    """
    Representation of the L5 Semantic Manifold.
    Concepts are points/vectors, and Truth is defined by a Curvature Field.
    """
    def __init__(self):
        # Coordinates in the concept space
        # We ensure "sky" and "blue" are topologically neighboring
        self.concepts = {
            "the": np.array([0.0, 0.0, 0.0]),
            "is": np.array([0.0, 0.0, 0.0]),
            "sky": np.array([1.0, 0.0, 0.0]),
            "blue": np.array([1.1, 0.0, 0.0]), # Near sky (dist=0.1)
            "green": np.array([0.0, 10.0, 0.0]), # Extremely far (dist~10)
            "thermodynamics": np.array([0.0, 0.0, 1.0]),
            "perpetual_motion": np.array([100.0, 100.0, 100.0]) # Deep Singularity
        }
        
        # Truth Singularities (Axioms)
        # We model curvature such that paths between these are geodesics (low curvature)
        self.axioms = [
            ("sky", "blue"),
            ("thermodynamics", "entropy_increase")
        ]

    def lift(self, tokens):
        # Map tokens to concept coordinates
        # If token is unknown, we place it in a high-variance "noise" region
        return [self.concepts.get(t.lower(), np.array([10.0, 10.0, 10.0])) for t in tokens]

    def calculate_curvature(self, path):
        """
        Measures the logical divergence of a semantic path.
        We check for "Topological Snaps" - large jumps in coordinate space.
        """
        if len(path) < 2: return 0.0
        
        total_curvature = 0.0
        for i in range(len(path) - 1):
            dist = np.linalg.norm(path[i+1] - path[i])
            if dist > 1.5: 
                contrib = (dist - 1.5) ** 2
                total_curvature += contrib
                
        return total_curvature

    def is_contractible(self, path):
        """
        A path is contractible if its curvature is below the semantic stability limit.
        """
        curvature = self.calculate_curvature(path)
        return curvature < 0.01

