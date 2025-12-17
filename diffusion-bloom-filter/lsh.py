
import numpy as np

class LSHEncoder:
    def __init__(self, input_dim: int, output_range: int, seed: int = 42):
        """
        Initializes the Locality Sensitive Hash encoder.
        
        Args:
            input_dim: Dimension of input vectors.
            output_range: Size of the 1D cyclic array (M).
            seed: Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.output_range = output_range
        self.rng = np.random.default_rng(seed)
        
        # We project to 2D to get an angle for cyclic mapping
        self.projection_matrix = self.rng.standard_normal((input_dim, 2))
        
    def hash(self, vector: np.ndarray) -> int:
        """
        Hashes a vector to an index in [0, output_range-1].
        
        Args:
            vector: Input vector of shape (input_dim,).
            
        Returns:
            Integer index.
        """
        # Project to 2D
        # shape: (2,)
        projected = np.dot(vector, self.projection_matrix)
        
        # Calculate angle in [-pi, pi]
        angle = np.arctan2(projected[1], projected[0])
        
        # Normalize to [0, 1]
        # -pi -> 0, pi -> 1
        normalized = (angle + np.pi) / (2 * np.pi)
        
        # Map to [0, output_range]
        index = int(normalized * self.output_range)
        
        # Clip just in case of float precision issues at the boundary
        return np.clip(index, 0, self.output_range - 1)
