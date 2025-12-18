import numpy as np
from core import MultiResolutionGrid, FieldState
from monitor import SpectralMonitor
from operators import WaveOperator, HeatOperator
from regulator import ThermodynamicRegulator
from utils import VectorDatabase, normalize, euclidean_distance

class ThermodynamicVectorSearch:
    """
    Main search system combining all components
    """
    
    def __init__(self, dimension, database, 
                 resolutions=[64, 256, 1024]):
        
        self.dimension = dimension
        self.database = database
        
        # Compute bounds from data
        self.bounds = self._compute_bounds(database)
        
        # Initialize multi-resolution grid
        self.grid = MultiResolutionGrid(
            dimension, self.bounds, resolutions
        )
        
        # Initialize components
        self.monitor = SpectralMonitor()
        self.regulator = ThermodynamicRegulator()
        
        # Build index (populate grid with data)
        self._build_index()
    
    def _compute_bounds(self, database):
        """
        Compute bounding box from database vectors
        """
        if len(database.vectors) == 0:
            # Default unit hypercube
            return [(0.0, 1.0)] * self.dimension
        
        vectors = np.array(database.vectors)
        min_vals = vectors.min(axis=0)
        max_vals = vectors.max(axis=0)
        
        # Add 10% padding
        padding = (max_vals - min_vals) * 0.1
        # Handle zero range case (e.g. single point or all same)
        padding[padding == 0] = 0.1 
        
        min_vals -= padding
        max_vals += padding
        
        return list(zip(min_vals, max_vals))
    
    def _build_index(self):
        """
        Populate grid with data points
        This is the "indexing" phase (preprocessing)
        """
        print(f"Building index for {len(self.database.vectors)} vectors...")
        
        # Use coarse grid for initial indexing
        coarse_level = 0
        field_state = FieldState(self.grid, coarse_level)
        
        # Add all vectors as sources
        for vector in self.database.vectors:
            field_state.add_source(vector, weight=1.0)
        
        # Light diffusion to smooth
        heat_op = HeatOperator(diffusion_coeff=0.05, dt=0.01)
        for _ in range(10):
            heat_op.step(field_state)
        
        # Store preprocessed field
        self.preprocessed_field = field_state
        
        print("Index built.")
    
    def search(self, query_vector, k=10):
        """
        Main search function
        
        Args:
            query_vector: numpy array of shape (dimension,)
            k: number of nearest neighbors
        
        Returns:
            List of (index, distance, metadata) tuples
        """
        # Normalize query
        query_vector = normalize(query_vector)
        
        # Phase 1: Wave scan (coarse)
        print("\n=== Phase 1: Wave Scan ===")
        resonance_region = self._phase1_wave_scan(query_vector)
        
        if resonance_region is None:
            print("No resonance found. Falling back to brute force.")
            return self._phase3_exact_ranking(None, k, query_vector)
        
        # Phase 2: Heat refinement (fine)
        print("\n=== Phase 2: Heat Refinement ===")
        refined_location = self._phase2_heat_refine(
            query_vector, resonance_region
        )
        
        # Phase 3: Exact Ranking (Hybrid Approach)
        print("\n=== Phase 3: Exact Ranking (Hybrid) ===")
        # Fetch more candidates for re-ranking (e.g. 10x k or at least 100)
        num_candidates = max(100, k * 10)
        
        results = self._phase3_exact_ranking(
            refined_location, num_candidates, query_vector
        )
        
        # Sort by exact distance to query and take top k
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def _phase1_wave_scan(self, query_vector):
        """
        Phase 1: Fast wave scan to find resonance region
        """
        # Start at coarse level
        level = 0
        field_state = FieldState(self.grid, level)
        
        # Initialize with query
        wave_op = WaveOperator(wave_speed=1.0, damping=0.5)
        wave_op.initialize_query(field_state, query_vector)
        
        # Also add background field from index
        # Need to make sure self.preprocessed_field is at same level
        # Assuming preprocessed is level 0
        field_state.phi += self.preprocessed_field.phi
        
        # Reset regulator
        self.regulator.energy_spent = 0
        self.regulator.start_phase("wave", field_state)
        
        # Evolve wave
        max_steps = 100
        for step in range(max_steps):
            # Step operator
            field_state = wave_op.step(field_state)
            
            # Check for transition
            should_switch, reason, next_type = \
                self.regulator.should_transition(
                    field_state, wave_op, self.monitor
                )
            
            if should_switch:
                print(f"  Wave transition: {reason}")
                
                if reason == "resonance_detected":
                    # Extract resonance region
                    has_res, location, strength = \
                        wave_op.detect_resonance(field_state)
                    
                    print(f"  Resonance at {location}, strength={strength:.3f}")
                    
                    # Get bounding box for zoom
                    region = wave_op.get_resonance_region(
                        field_state, location, radius=0.5
                    )
                    return region
                
                elif reason == "wave_timeout":
                    print("  Wave timeout - no clear resonance")
                    return None
                
                elif reason == "gap_collapse":
                    print("  Wave collapsed - likely no match")
                    return None
            
            # Record energy
            self.regulator.record_energy(1)
            
            if step % 10 == 0:
                gap = self.monitor.compute_spectral_gap(field_state)
                # phi max
                phi_max = np.max(field_state.phi)
                print(f"  Step {step}: gap={gap:.4f}, max_phi={phi_max:.3f}")
        
        return None
    
    def _phase2_heat_refine(self, query_vector, region):
        """
        Phase 2: Local heat diffusion refinement
        """
        # Zoom to finer level
        # Check if we have fine resolutions available
        level = 1 # Medium res first or fine? Pseudocode said 2
        if level not in self.grid.grids:
            level = 0 # fallback if no multi-res
        
        field_state = FieldState(self.grid, level)
        
        # Crop grid to region (Simulated by just running on full grid but focused, 
        # or actually changing bounds? Pseudocode implies logic for zoom)
        # Pseudocode: field_state.grid.bounds = ...
        # But FieldState shares grid. Modifying grid modifies it for everyone?
        # Creating a NEW grid or subgrid would be better.
        # For this implementation, we will keep global bounds but maybe just focus 
        # on the area if we implemented sparse grids.
        # But `FieldState` uses `self.grid`. 
        # To strictly follow pseudocode: "field_state.grid.bounds = ..."
        # This implies modifying the grid instance derived from main grid.
        # Let's NOT modify the main grid object. We can simulate zoom by just passing region, 
        # but Wave/Heat operators work on whole 'phi'.
        # If we just run it on the same grid resolution it won't be "finer".
        # The pseudocode implies dynamic regridding or just switching level.
        # "field_state.grid.bounds = list(zip(min_bounds, max_bounds))"
        # If we change bounds, point_to_grid_index changes mapping.
        # But 'phi' array size is fixed by resolution.
        # So we effectively map a smaller physical region to the same number of grid cells -> higher resolution.
        
        # Create a temporary local grid for this phase
        # We need to copy grid structure or create new one
        # Because we want higher resolution in physical space
        
        min_bounds, max_bounds = region
        new_bounds = list(zip(min_bounds, max_bounds))
        
        # Create a temporary one-level grid for refinement
        # We want 'level 2' resolution count (e.g. 1024) but covered only 'region'
        # Or maybe just use the 'level' resolution count.
        resolution = self.grid.grids[level]['resolution']
        
        # We need a new Grid object basically
        # Hack: Create a new MultiResolutionGrid with single level for this scope?
        # Or just modify the field_state's reference to a modified grid copy.
        
        # Let's create a partial grid dict
        local_grid_data = self.grid._create_grid(resolution)
        
        # We need a dummy object that looks like MultiResolutionGrid or just pass needed info
        # FieldState needs .grid to have .bounds and .point_to_grid_index
        
        class LocalGrid:
            def __init__(self, dimension, bounds, grids):
                self.dimension = dimension
                self.bounds = bounds
                self.grids = grids
            
            def point_to_grid_index(self, point, level):
                # Copy from MultiResolutionGrid
                grid = self.grids[level]
                indices = []
                for d, (min_val, max_val) in enumerate(self.bounds):
                    normalized = (point[d] - min_val) / (max_val - min_val)
                    idx = int(normalized * grid['resolution'])
                    idx = max(0, min(idx, grid['resolution'] - 1)) # clamp
                    indices.append(idx)
                return tuple(indices)
            
            def grid_index_to_point(self, indices, level):
                grid = self.grids[level]
                point = []
                for d, idx in enumerate(indices):
                    min_val, max_val = self.bounds[d]
                    normalized = idx / grid['resolution']
                    coord = min_val + normalized * (max_val - min_val)
                    point.append(coord)
                return np.array(point)

        local_grids = {level: local_grid_data}
        local_grid_obj = LocalGrid(self.dimension, new_bounds, local_grids)
        
        field_state = FieldState(local_grid_obj, level)
        
        # Initialize heat operator
        heat_op = HeatOperator(
            diffusion_coeff=0.1,
            anisotropic=True
        )
        heat_op.initialize_query(
            field_state, query_vector, self.database
        )
        
        self.regulator.start_phase("heat", field_state)
        
        # Evolve heat equation
        max_steps = 100
        for step in range(max_steps):
            # Step
            field_state = heat_op.step(field_state)
            
            # Check transition
            should_switch, reason, next_type = \
                self.regulator.should_transition(
                    field_state, heat_op, self.monitor
                )
            
            if should_switch:
                print(f"  Heat transition: {reason}")
                break
            
            self.regulator.record_energy(1)
            
            if step % 10 == 0:
                gap = self.monitor.compute_spectral_gap(field_state)
                phi_max = np.max(field_state.phi)
                print(f"  Step {step}: gap={gap:.4f}, max_phi={phi_max:.3f}")
        
        # Extract final location
        location, value = heat_op.extract_local_maximum(field_state)
        print(f"  Refined location: {location}, value={value:.3f}")
        
        return location
    
    def _phase3_exact_ranking(self, location, k, query_vector=None):
        """
        Phase 3: Hybrid Refinement
        Finds k candidates near 'location' and ranks them by distance to 'query_vector'
        """
        if location is None:
            if query_vector is None:
                return []
            print("  Fallback: Global ranking near query point")
            candidates = self.database.get_nearest_to_point(query_vector, k)
            results = []
            for idx, dist in candidates:
                metadata = self.database.get_metadata(idx)
                results.append((idx, dist, metadata))
            return results

        # Hybrid: Find candidates near thermal peak
        # We start with a small radius and expand if not enough candidates found
        radius = 0.5
        candidates_indices = set()
        candidates = []
        
        # Iterative expansion to find enough candidates
        max_search_vectors = len(self.database.vectors)
        
        while len(candidates) < k and radius < 2.0 * self.dimension: # Limit expansion
            candidates = []
            candidates_indices = set()
            
            # This linear scan is O(N). In production, use spatial index (KDTree/LSH).
            for i, vec in enumerate(self.database.vectors):
                # Check soft boundary first (box) to avoid sqrt? 
                # Just euclidean for now.
                dist_to_peak = euclidean_distance(location, vec)
                
                if dist_to_peak < radius:
                    if i not in candidates_indices:
                        # Calculate exact distance to QUERY (not peak)
                        target = query_vector if query_vector is not None else location
                        real_dist = euclidean_distance(target, vec)
                        candidates.append((i, real_dist))
                        candidates_indices.add(i)
            
            if len(candidates) < k:
                print(f"  Radius {radius:.2f}: found {len(candidates)} candidates. Expanding...")
                radius *= 1.5
            else:
                print(f"  Radius {radius:.2f}: found {len(candidates)} candidates.")
                
        # If still not enough, take nearest to peak directly (global sort)
        if len(candidates) < k:
             print("  Not enough near peak. Getting nearest to peak globally.")
             fallback = self.database.get_nearest_to_point(location, k)
             # Recalculate dist to query
             target = query_vector if query_vector is not None else location
             for idx, _ in fallback:
                 if idx not in candidates_indices:
                     vec = self.database.get_vector(idx)
                     d = euclidean_distance(target, vec)
                     candidates.append((idx, d))

        # Sort by real distance
        candidates.sort(key=lambda x: x[1])
        
        # Return top k with metadata
        results = []
        for idx, dist in candidates[:k]:
            metadata = self.database.get_metadata(idx)
            results.append((idx, dist, metadata))
        
        return results

def build_index(vectors, metadata_list=None, 
                dimension=None, resolutions=[64, 256, 1024]):
    """
    Build thermodynamic index from vectors
    
    Args:
        vectors: List of numpy arrays
        metadata_list: Optional list of metadata dicts
        dimension: Vector dimension (inferred if None)
        resolutions: Grid resolutions for multi-scale
    
    Returns:
        ThermodynamicVectorSearch index
    """
    if dimension is None:
        dimension = len(vectors[0])
    
    # Create database
    database = VectorDatabase()
    for i, vec in enumerate(vectors):
        metadata = metadata_list[i] if metadata_list else {'id': i}
        database.insert(vec, metadata)
    
    # Create search index
    index = ThermodynamicVectorSearch(
        dimension=dimension,
        database=database,
        resolutions=resolutions
    )
    
    return index
