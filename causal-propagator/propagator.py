import numpy as np

class CausalPropagator:
    """
    Parent: Koopman Operator Theory (L4).
    Linearizes chaos by lifting data into a higher-dimensional spectral space.
    """
    def __init__(self, n_modes=10):
        self.n_modes = n_modes
        self.U = None # The Koopman Operator
        self.modes = None
        self.eigenvalues = None
        self.last_state = None

    def lift_to_koopman_operator(self, data):
        """
        Approximate the Koopman Operator using Dynamic Mode Decomposition (DMD).
        X_next = U * X
        """
        # Create delay-embedding matrix (Lifting to higher dimensions)
        # We use a Hankel matrix for more stable spectral decomposition
        d = self.n_modes
        X = []
        # Increase visibility window
        for i in range(len(data) - d):
            X.append(data[i:i+d])
        X = np.array(X).T # Each column is a state vector
        
        # SVD-based DMD for better stability
        U_svd, s, Vh = np.linalg.svd(X[:, :-1], full_matrices=False)
        S_inv = np.diag(1.0 / s)
        # Approximate Koopman operator on the reduced subspace
        self.U = X[:, 1:] @ Vh.T @ S_inv @ U_svd.T
        
        self.last_state = X[:, -1]
        
        # Spectral decomposition for analysis
        vals, vecs = np.linalg.eig(self.U)
        self.eigenvalues = vals
        self.modes = vecs

    def propagate(self, horizon):
        """
        Linear evolution in the Parent Space.
        No error growth because we evolve the field, not the point.
        """
        forecast_lifted = []
        current = self.last_state
        for _ in range(horizon):
            current = self.U @ current
            forecast_lifted.append(current[-1]) # Project back to real dimension
            
        return np.array(forecast_lifted)

    def detect_caustics(self, future_field):
        """
        Identifies points of constructive interference in the causal field.
        """
        # In this prototype, we look for high-magnitude spectral modes
        return np.where(np.abs(self.eigenvalues) > 0.99)[0]
