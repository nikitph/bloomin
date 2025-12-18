import numpy as np
from itertools import product
from scipy.ndimage import convolve

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

class MultiResolutionGrid:
    """
    Hierarchical grid structure for multi-scale field evolution
    """
    
    def __init__(self, dimension, bounds, resolutions):
        """
        Args:
            dimension: Vector dimensionality (e.g., 128, 768, 1536)
            bounds: [(min_0, max_0), (min_1, max_1), ...] for each dimension
            resolutions: [coarse_res, medium_res, fine_res] 
                        e.g., [64, 256, 1024]
        """
        self.dimension = dimension
        self.bounds = bounds
        
        # Create grid hierarchy
        self.grids = {}
        for level, res in enumerate(resolutions):
            self.grids[level] = self._create_grid(res)
    
    def _create_grid(self, resolution):
        """Create a single grid level"""
        # Grid shape: [res, res, ..., res] (dimension times)
        shape = [resolution] * self.dimension
        
        return {
            'phi': np.zeros(shape),        # Field values
            'phi_dot': np.zeros(shape),    # Field velocity (for wave equation)
            'phi_prev': np.zeros(shape),   # Previous timestep (for momentum)
            'resolution': resolution,
            'cell_size': self._compute_cell_size(resolution)
        }
    
    def _compute_cell_size(self, resolution):
        """Compute physical size of each grid cell"""
        cell_sizes = []
        for (min_val, max_val) in self.bounds:
            cell_sizes.append((max_val - min_val) / resolution)
        return cell_sizes
    
    def point_to_grid_index(self, point, level):
        """Convert physical point to grid indices"""
        grid = self.grids[level]
        indices = []
        
        for d, (min_val, max_val) in enumerate(self.bounds):
            # Normalize to [0, 1]
            normalized = (point[d] - min_val) / (max_val - min_val)
            # Scale to grid
            idx = int(normalized * grid['resolution'])
            # Clamp to valid range
            idx = clamp(idx, 0, grid['resolution'] - 1)
            indices.append(idx)
        
        return tuple(indices)
    
    def grid_index_to_point(self, indices, level):
        """Convert grid indices back to physical point"""
        grid = self.grids[level]
        point = []
        
        for d, idx in enumerate(indices):
            min_val, max_val = self.bounds[d]
            # Denormalize
            normalized = idx / grid['resolution']
            coord = min_val + normalized * (max_val - min_val)
            point.append(coord)
        
        return np.array(point)
    
    def interpolate_field(self, point, level, field_name='phi'):
        """
        Trilinear interpolation to get field value at arbitrary point
        """
        grid = self.grids[level]
        field = grid[field_name]
        
        # Get surrounding grid points
        indices_float = []
        for d, (min_val, max_val) in enumerate(self.bounds):
            normalized = (point[d] - min_val) / (max_val - min_val)
            idx_float = normalized * grid['resolution']
            indices_float.append(idx_float)
        
        # Perform multi-linear interpolation
        return self._multilinear_interpolate(field, indices_float)
    
    def _multilinear_interpolate(self, field, indices_float):
        """
        Generalized multi-linear interpolation in d dimensions
        """
        # Get integer and fractional parts
        indices_low = [int(idx) for idx in indices_float]
        indices_high = [min(int(idx) + 1, s - 1) 
                       for idx, s in zip(indices_float, field.shape)]
        weights = [idx - int(idx) for idx in indices_float]
        
        # Iterate over all 2^d corners of hypercube
        result = 0.0
        for corner in product([0, 1], repeat=len(indices_float)):
            # Get indices for this corner
            corner_indices = tuple(
                indices_low[d] if corner[d] == 0 else indices_high[d]
                for d in range(len(corner))
            )
            
            # Get field value at corner
            value = field[corner_indices]
            
            # Compute weight for this corner
            weight = 1.0
            for d, c in enumerate(corner):
                weight *= weights[d] if c == 1 else (1 - weights[d])
            
            result += weight * value
        
        return result


class FieldState:
    """
    Complete state of the information field at a given time
    """
    
    def __init__(self, grid, level):
        self.grid = grid
        self.level = level
        self.time = 0.0
        
        # Field components (references to grid storage)
        self.phi = grid.grids[level]['phi']           # Main field
        self.phi_dot = grid.grids[level]['phi_dot']   # Velocity
        self.phi_prev = grid.grids[level]['phi_prev'] # Previous
        
        # Metadata
        self.sources = []  # List of (position, weight) for data points
        self.energy = 0.0
        self.spectral_gap = 1.0
    
    def add_source(self, position, weight=1.0):
        """Add a Dirac delta source at position"""
        self.sources.append((position, weight))
        
        # Update field with delta function (approximated by Gaussian)
        indices = self.grid.point_to_grid_index(position, self.level)
        sigma = 2.0  # Width of Gaussian approximation
        
        # Add Gaussian bump
        self._add_gaussian(indices, weight, sigma)
    
    def _add_gaussian(self, center_indices, weight, sigma):
        """
        Add Gaussian bump to field
        (Approximation of Dirac delta)
        """
        # Iterate over neighborhood
        radius = int(3 * sigma)  # 3-sigma rule
        
        for offset in self._get_neighborhood(radius):
            indices = tuple(c + o for c, o in zip(center_indices, offset))
            
            # Check bounds
            if not self._in_bounds(indices):
                continue
            
            # Compute Gaussian weight
            dist_sq = sum(o**2 for o in offset)
            gauss_weight = np.exp(-dist_sq / (2 * sigma**2))
            
            # Add to field
            self.phi[indices] += weight * gauss_weight
    
    def _get_neighborhood(self, radius):
        """Generate all offsets within radius"""
        # Note: self.phi.shape determines dimension
        ranges = [range(-radius, radius + 1)] * len(self.phi.shape)
        return product(*ranges)
    
    def _in_bounds(self, indices):
        """Check if indices are within grid bounds"""
        return all(0 <= idx < size 
                  for idx, size in zip(indices, self.phi.shape))
    
    def compute_energy(self):
        """Compute total energy (Dirichlet energy)"""
        gradient_val = self._compute_gradient(self.phi)
        # Assuming gradient returns list of arrays, one for each dimension
        # Norm squared sum
        grad_sq_sum = sum(g**2 for g in gradient_val) if isinstance(gradient_val, list) else gradient_val**2
        self.energy = 0.5 * np.sum(grad_sq_sum)
        return self.energy
    
    def _compute_gradient(self, field):
        """Compute gradient using finite differences"""
        # Use numpy/scipy gradient
        # For each dimension, compute ∂φ/∂xi
        gradients = []
        for d in range(len(field.shape)):
            grad_d = np.gradient(field, axis=d)
            gradients.append(grad_d)
        
        return gradients
