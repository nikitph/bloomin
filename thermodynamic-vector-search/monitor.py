import numpy as np
from scipy.ndimage import convolve

class SpectralMonitor:
    """
    Real-time monitoring of spectral properties
    Uses power iteration for efficiency
    """
    
    def __init__(self, max_iterations=20, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # History for trend analysis
        self.gap_history = []
        self.time_history = []
    
    def compute_spectral_gap(self, field_state):
        """
        Estimate spectral gap Δ = λ₁ (first non-zero eigenvalue)
        """
        # Inverse power iteration to find smallest non-zero eigenvalue
        lambda_1 = self._inverse_power_iteration(field_state)
        
        # Gap is directly lambda_1 for diffusion
        gap = lambda_1
        
        # Store in history
        self.gap_history.append(gap)
        self.time_history.append(field_state.time)
        
        return gap
    
    def _inverse_power_iteration(self, field_state):
        """
        Inverse iteration to find SMALLEST nonzero eigenvalue
        
        Solves: (L - shift·I)⁻¹·v = λ·v
        """
        from scipy import sparse
        from scipy.sparse import linalg as splinalg
        import numpy as np
        
        phi = field_state.phi
        N = phi.size
        
        # Build Laplacian matrix
        L = self._build_laplacian_matrix(phi.shape, field_state)
        
        # Shift (small positive to avoid zero eigenvalue)
        shift = 1e-6
        I = sparse.eye(N)
        A = L - shift * I
        
        # Random initial vector
        v = np.random.randn(N)
        v = v / (np.linalg.norm(v) + 1e-12)
        
        # Inverse iteration
        for _ in range(self.max_iterations):
            # Solve A·v_next = v
            # Use try-except for singular matrix if shift is bad
             try:
                v_next = splinalg.spsolve(A, v)
             except RuntimeError:
                # If fails, return last guess or break
                break

             norm_v_next = np.linalg.norm(v_next)
             if norm_v_next < 1e-12:
                 break

             v_next = v_next / norm_v_next
             
             if np.linalg.norm(v_next - v) < self.tolerance:
                 break
             
             v = v_next
        
        # Rayleigh quotient gives λ (approx)
        # But we need to use L, not A
        Lv = L @ v
        lambda_est = np.dot(v, Lv)
        
        return abs(lambda_est) # Should be positive
    
    def _build_laplacian_matrix(self, shape, field_state):
        """
        Build sparse Laplacian matrix using finite differences
        """
        from scipy.sparse import diags, kron, eye
        import numpy as np
        
        ndim = len(shape)
        N_per_dim = shape[0]
        
        h = field_state.grid.grids[field_state.level]['cell_size'][0]
        
        # 1D Laplacian: [1, -2, 1] / h^2
        # Use negative definite Laplacian L (so eigenvalues are negative)
        # Power iteration finds largest magnitude.
        # Inverse find smallest magnitude.
        # Diffusion operator is usually -L (positive semi-definite).
        # We want smallest non-zero eigenvalue of -L.
        # Let's build L as [1, -2, 1]. This has negative eigenvalues.
        # -L has positive eigenvalues.
        # Let's build Positive Definite Laplacian (-L).
        # -(-2) = 2 on diagonal. -1 on off-diagonal.
        # diagonals: [-1, 2, -1]. 
        # Wait, the previous implementation in `operators.py` used [1, -2, 1] for L.
        # If we use L directly, eigenvalues are 0, -pi^2, ...
        # small eigenvalue is close to 0. large is -4d.
        # Inverse iteration on L finds 0 (if we don't shift).
        # We want second eigenvalue (closest to 0 but not 0).
        # Applying shift 1e-6 helps avoid 0.
        # But constructing Positive Definite matrix is better for solvers?
        # Let's construct L as in Operators (negative definite) and use `shift`.
        # Diagonals for [1, -2, 1]:
        
        diagonals = [np.ones(N_per_dim-1), -2*np.ones(N_per_dim), np.ones(N_per_dim-1)]
        L_1d = diags(diagonals, [-1, 0, 1], shape=(N_per_dim, N_per_dim))
        L_1d = L_1d / (h**2)
        
        # Negate to make it Positive Semi-Definite (for easier interpretation of lambda > 0)
        # The user said "Gap = lambda_1 - lambda_0".
        # If we return positive lambda, it's easier.
        L_1d = -L_1d
        
        if ndim == 1:
            return L_1d
        
        L_curr = L_1d
        for d in range(1, ndim):
            dim_so_far = N_per_dim ** d
            I_prev = eye(dim_so_far)
            I_new = eye(N_per_dim)
            L_curr = kron(L_curr, I_new) + kron(I_prev, L_1d)
        
        return L_curr
    
    def _apply_laplacian(self, field, field_state):
        """
        Apply discrete Laplacian operator
        ΔΦ = Σ_neighbors (Φ_neighbor - Φ_center)
        """
        laplacian = np.zeros_like(field)
        
        # Use convolution with Laplacian kernel
        # For d-dimensional grid: kernel has -2d at center, 
        # +1 at each of 2d face neighbors
        
        kernel = self._get_laplacian_kernel(field.ndim)
        
        # Convolve (with boundary handling)
        laplacian = convolve(field, kernel, mode='constant', cval=0.0)
        
        return laplacian
    
    def _get_laplacian_kernel(self, ndim):
        """
        Create discrete Laplacian kernel for ndim dimensions
        """
        # Create kernel: shape [3, 3, ..., 3] (ndim times)
        kernel_shape = [3] * ndim
        kernel = np.zeros(kernel_shape)
        
        # Center has -2d
        center = tuple([1] * ndim)
        kernel[center] = -2 * ndim
        
        # Each face neighbor has +1
        for d in range(ndim):
            # Positive direction
            idx_pos = list(center)
            idx_pos[d] = 2
            kernel[tuple(idx_pos)] = 1.0
            
            # Negative direction
            idx_neg = list(center)
            idx_neg[d] = 0
            kernel[tuple(idx_neg)] = 1.0
        
        return kernel
    
    def predict_collapse_time(self, current_gap, critical_gap):
        """
        Predict when spectral gap will close based on trend
        """
        if len(self.gap_history) < 10:
            return float('inf')  # Not enough data
        
        # Fit linear trend to recent history
        recent_gaps = self.gap_history[-20:]
        recent_times = self.time_history[-20:]
        
        # Linear regression: gap = a*t + b
        slope = self._linear_fit_slope(recent_times, recent_gaps)
        
        if slope >= 0:
            return float('inf')  # Gap not closing
        
        # Time when gap reaches critical threshold
        time_to_collapse = (critical_gap - current_gap) / slope
        
        return time_to_collapse
    
    def _linear_fit_slope(self, x, y):
        """Simple linear regression slope"""
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean)**2 for i in range(n))
        
        if denominator < 1e-12:
            return 0.0
        
        return numerator / denominator
