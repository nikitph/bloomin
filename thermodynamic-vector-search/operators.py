import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import maximum_position as argmax_pos
import math
from utils import euclidean_distance

class WaveOperator:
    """
    Phase 1: Fast coarse scan using damped wave equation
    
    Equation: ∂²φ/∂t² + γ∂φ/∂t = c²∇²φ + f_query(x)
    """
    
    def __init__(self, wave_speed=1.0, damping=0.5, dt=0.01):
        self.c = wave_speed      # Wave propagation speed
        self.gamma = damping     # Damping coefficient
        self.dt = dt             # Time step
        
        self.name = "WaveOperator"
    
    def initialize_query(self, field_state, query_vector):
        """
        Initialize wave source at query location
        """
        # Add source at query position
        field_state.add_source(query_vector, weight=1.0)
        
        # Initialize velocity to zero
        field_state.phi_dot.fill(0.0)
    
    def step(self, field_state):
        """
        Single time step of wave evolution
        
        Uses leap-frog integration for stability
        """
        phi = field_state.phi
        phi_dot = field_state.phi_dot
        
        # Compute Laplacian: ∇²φ
        laplacian = self._compute_laplacian(phi, field_state)
        
        # Wave equation: ∂²φ/∂t² = c²∇²φ - γ∂φ/∂t
        phi_dot_dot = self.c**2 * laplacian - self.gamma * phi_dot
        
        # Update velocity: φ_dot(t+dt) = φ_dot(t) + φ_dot_dot * dt
        phi_dot_new = phi_dot + phi_dot_dot * self.dt
        
        # Update field: φ(t+dt) = φ(t) + φ_dot * dt
        phi_new = phi + phi_dot_new * self.dt
        
        # Store updates
        field_state.phi = phi_new
        field_state.phi_dot = phi_dot_new
        field_state.time += self.dt
        
        return field_state
    
    def _compute_laplacian(self, phi, field_state):
        """Compute discrete Laplacian"""
        # Use 2d+1 point stencil (center + 2d neighbors)
        laplacian = np.zeros_like(phi)
        
        # Create Laplacian kernel
        ndim = len(phi.shape)
        kernel = self._laplacian_kernel(ndim)
        
        # Apply via convolution
        laplacian = convolve(phi, kernel, mode='constant', cval=0.0)
        
        # Normalize by cell size
        h = field_state.grid.grids[field_state.level]['cell_size'][0]
        laplacian = laplacian / (h**2)
        
        return laplacian
    
    def _laplacian_kernel(self, ndim):
        """Create Laplacian stencil kernel"""
        kernel_shape = [3] * ndim
        kernel = np.zeros(kernel_shape)
        
        center = tuple([1] * ndim)
        kernel[center] = -2 * ndim
        
        for d in range(ndim):
            idx_pos = list(center)
            idx_pos[d] = 2
            kernel[tuple(idx_pos)] = 1.0
            
            idx_neg = list(center)
            idx_neg[d] = 0
            kernel[tuple(idx_neg)] = 1.0
        
        return kernel
    
    def detect_resonance(self, field_state, threshold=0.1):
        """
        Detect strong resonance regions
        Returns: (has_resonance, resonance_location, resonance_strength)
        """
        phi = field_state.phi
        
        # Find maximum amplitude
        # argmax returns flat index, we need unravel_index or similar
        max_idx = np.unravel_index(np.argmax(phi), phi.shape)
        max_value = phi[max_idx]
        
        # Check if above threshold
        if max_value > threshold:
            # Convert grid indices to physical location
            location = field_state.grid.grid_index_to_point(
                max_idx, field_state.level
            )
            return True, location, max_value
        
        return False, None, 0.0
    
    def get_resonance_region(self, field_state, center, radius):
        """
        Extract bounding box around resonance region
        Returns: (min_bounds, max_bounds) for zooming
        """
        min_bounds = [c - radius for c in center]
        max_bounds = [c + radius for c in center]
        
        # Clamp to grid bounds
        for d in range(len(center)):
            grid_min, grid_max = field_state.grid.bounds[d]
            min_bounds[d] = max(min_bounds[d], grid_min)
            max_bounds[d] = min(max_bounds[d], grid_max)
        
        return min_bounds, max_bounds


class HeatOperator:
    """
    Phase 2: Local refinement using anisotropic heat diffusion
    
    Equation: ∂φ/∂t = ∇·(D(x)∇φ) + f_query(x)
    """
    
    def __init__(self, diffusion_coeff=0.1, dt=0.01, 
                 anisotropic=True):
        self.D0 = diffusion_coeff  # Base diffusion coefficient
        self.dt = dt
        self.anisotropic = anisotropic
        self.name = "HeatOperator"
        self.diffusion_tensors = None
        
        # Calculate stable timestep on init if possible, or assume user provides reasonable dt
        # But user requested adaptive timestep as 'Quick Fix' or Crank-Nicolson.
        # I will implement Crank-Nicolson as 'Best Choice'.
    
    def initialize_query(self, field_state, query_vector, 
                        database=None):
        """Initialize heat source at query location"""
        field_state.add_source(query_vector, weight=1.0)
        # We don't necessarily need tensors for isotropic Crank-Nicolson unless we implement full anisotropic FE/FD.
        # User's example for Crank-Nicolson uses 'self.D0 * L', which implies ISOTROPIC diffusion for the stability fix.
        # Anisotropic implicit is much harder (requires building B matrix with tensors).
        # User said: "(I - dt/2·D·Δ)φ = ..." which uses scalar D.
        # If anisotropic=True, we should perhaps stick to explicit with adaptive dt, OR use isotropic implicit.
        # The user's solution B shows: "A = sparse.eye... - ... self.D0 * L". This ignores diffusion_tensors.
        # So I will assume we switch to Isotropic Implicit/CN for stability, 
        # OR I implement adaptive explicit for anisotropic.
        # User said "Best choice for production: More accurate, still stable." for CN.
        # But if we lose Anisotropy, we lose the "Shape" matching capability?
        # The user's prompt didn't explicitly say "drop anisotropy", but their code snippet for CN used self.D0.
        # I will implement Crank-Nicolson using D0 (Isotropic) for now as per snippet.
        # If anisotropic is needed, explicit with small dt is better or very complex sparse matrix construction.
        # Given "Problem 1: Exponential Growth", stability is key.
        pass

    def step(self, field_state):
        """
        Crank-Nicolson Step: (I - dt/2·D·Δ)φ^(n+1) = (I + dt/2·D·Δ)φ^n
        """
        # We use purely isotropic diffusion for the stable step as per fix suggestion
        phi = field_state.phi
        phi_new = self._crank_nicolson_step(phi, field_state)
        
        field_state.phi = phi_new
        field_state.time += self.dt
        return field_state

    def _crank_nicolson_step(self, phi, field_state):
        from scipy import sparse
        from scipy.sparse import linalg as splinalg
        
        L = self._build_laplacian_matrix(phi.shape, field_state)
        N = phi.size
        
        # Build matrices
        # A * x = B * b
        # (I - dt/2 * D * L) * phi_new = (I + dt/2 * D * L) * phi_old
        
        factor = (self.dt / 2) * self.D0
        I = sparse.eye(N)
        
        A = I - factor * L
        B = I + factor * L
        
        rhs = B @ phi.flatten()
        phi_new_flat = splinalg.spsolve(A, rhs)
        
        return phi_new_flat.reshape(phi.shape)

    def _build_laplacian_matrix(self, shape, field_state):
        """Build sparse Laplacian matrix using finite differences"""
        from scipy.sparse import diags, kron, eye
        import numpy as np
        
        ndim = len(shape)
        N_per_dim = shape[0]
        
        h = field_state.grid.grids[field_state.level]['cell_size'][0]
        
        # 1D Laplacian: [-1, 2, -1] / h^2? No, usually [1, -2, 1].
        # User snippet: [-2 at center].
        # So [-1, 2, -1] is wrong sign for Laplacian?
        # Laplacian of Gaussian is negative at peak.
        # User snippet: "Kernel... center has -2d". So it is negative definite?
        # User snippet for matrix: "diagonals = [ones, -2*ones, ones]".
        # This is standard [1, -2, 1].
        # So L matches the kernel used earlier.
        
        diagonals = [np.ones(N_per_dim-1), -2*np.ones(N_per_dim), np.ones(N_per_dim-1)]
        L_1d = diags(diagonals, [-1, 0, 1], shape=(N_per_dim, N_per_dim))
        L_1d = L_1d / (h**2)
        
        if ndim == 1:
            return L_1d
            
        # Kronecker sum: L_2d = L_1d x I + I x L_1d
        # scipy.sparse.kron(A, B)
        
        # Recursive or iterative construction
        # L_nd = L_1d x I x ... + I x L_1d x ... + ...
        
        # Start with 1D
        L_curr = L_1d
        
        for d in range(1, ndim):
            dim_so_far = N_per_dim ** d
            I_prev = eye(dim_so_far)
            I_new = eye(N_per_dim)
            
            # L_new = L_prev x I_new + I_prev x L_1d
            # Be careful with order.
            # Grid is usually (x, y, z...). 
            # Flattening is C-style (last index varies fastest).
            # So L_nd = L_{n-1} x I + I_{n-1} x L_1d
            
            L_curr = kron(L_curr, I_new) + kron(I_prev, L_1d)
            
        return L_curr
    
    def _compute_laplacian(self, phi, field_state):
        """Isotropic Laplacian for convenience"""
        ndim = len(phi.shape)
        kernel = np.zeros([3] * ndim)
        center = tuple([1] * ndim)
        kernel[center] = -2 * ndim
        
        for d in range(ndim):
            idx_pos = list(center)
            idx_pos[d] = 2
            kernel[tuple(idx_pos)] = 1.0
            
            idx_neg = list(center)
            idx_neg[d] = 0
            kernel[tuple(idx_neg)] = 1.0
        
        laplacian = convolve(phi, kernel, mode='constant', cval=0.0)
        h = field_state.grid.grids[field_state.level]['cell_size'][0]
        return laplacian / (h**2)
    
    def has_converged(self, field_state, tolerance=1e-4):
        """
        Check if field has reached steady state
        """
        # Need to handle first step where phi_prev is zeros
        if np.all(field_state.phi_prev == 0) and field_state.time == 0:
             field_state.phi_prev = field_state.phi.copy()
             return False

        # Compute change
        change = np.linalg.norm(field_state.phi - field_state.phi_prev)
        total = np.linalg.norm(field_state.phi)
        
        relative_change = change / (total + 1e-12)
        
        # Update previous
        field_state.phi_prev = field_state.phi.copy()
        
        return relative_change < tolerance
    
    def extract_local_maximum(self, field_state):
        """
        Find location of maximum field value
        """
        phi = field_state.phi
        max_idx = np.unravel_index(np.argmax(phi), phi.shape)
        
        # Convert to physical coordinates
        location = field_state.grid.grid_index_to_point(
            max_idx, field_state.level
        )
        
        value = phi[max_idx]
        
        return location, value
