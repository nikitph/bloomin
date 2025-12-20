import torch
import torch.nn.functional as F
import numpy as np

# Utility for sparse-dense multiplication
def sparse_dense_mul(s, d):
    return torch.sparse.mm(s, d.unsqueeze(1)).squeeze()

class WaveRetrievalEngine:
    def __init__(self, data_tensor: torch.Tensor, k_neighbors: int = 10):
        self.device = torch.device('cpu') # Force CPU for deterministic behavior in benchmark
        self.data = data_tensor.to(self.device)
        self.k = k_neighbors
        self.L = self._build_laplacian()
        
    def _build_laplacian(self) -> torch.Tensor:
        """
        Constructs the Normalized Graph Laplacian L = I - D^-1/2 A D^-1/2
        or Random Walk L = I - D^-1 A.
        Physics preference: Geometric/Normalized for standard diffusion.
        """
        print("Building Graph Laplacian...")
        # 1. KNN Graph
        # Brute force or FAISS? For benchmark size (e.g. 10k), brute force cdist is fine.
        dists = torch.cdist(self.data, self.data)
        
        # Get top k indices
        # We want to keep k smallest distances (excluding self)
        # torch.topk returns largest, so we negate
        _, indices = torch.topk(-dists, k=self.k + 1, dim=1)
        
        # Build Adjacency Matrix
        n = self.data.shape[0]
        rows = torch.arange(n).view(-1, 1).repeat(1, self.k + 1)
        # Flatten
        indices = indices.flatten()
        rows = rows.flatten()
        
        # Values = 1.0 (unweighted) or exp(-dist)
        # For simplicity in Wave: Unweighted KNN connectivity
        values = torch.ones_like(rows, dtype=torch.float32)
        
        # Create Sparse Matrix A
        A = torch.sparse_coo_tensor(torch.stack([rows, indices]), values, (n, n))
        
        # Calculate Degree Matrix D (diagonal)
        # Sum of A rows
        D_vec = torch.sparse.sum(A, dim=1).to_dense()
        
        # Normalized Laplacian: L = I - D^-1/2 A D^-1/2
        # Or Simple Random Walk: L = I - D^-1 A
        # Let's use simple L = D - A for diffusion (unnormalized) or normalized.
        # Standard: L = I - D^-1 A  (Transition probability version)
        
        # Inverse Degree
        D_inv = 1.0 / (D_vec + 1e-8)
        
        # L = I - D^-1 * A
        # We need sparse multiplication D^-1 * A. 
        # Since D is diagonal, we just scale rows of A.
        
        # Scaling A indices by D_inv
        # A_ij is at rows[k], so multiply by D_inv[rows[k]]
        new_values = values * D_inv[rows]
        
        RW_L = torch.sparse_coo_tensor(torch.stack([rows, indices]), new_values, (n, n))
        
        # Identity
        I_indices = torch.arange(n).repeat(2, 1)
        I_values = torch.ones(n)
        I = torch.sparse_coo_tensor(I_indices, I_values, (n, n))
        
        L = I - RW_L
        print("Graph built.")
        return L.to(self.device).coalesce()

    def wave_step(self, psi_0: torch.Tensor, T_wave: int = 10, c: float = 1.0, dt: float = 0.1, sigma: float = 1.0, mask: torch.Tensor = None):
        """
        Step 1: Wave Equation (Exploration)
        d2psi/dt2 = c^2 * L * psi
        """
        # Initial Injection: localized gaussian or raw
        # Convert query vec to initial field psi
        # Distances from query to all nodes
        dists = torch.cdist(psi_0.unsqueeze(0), self.data).squeeze()
        psi = torch.exp(-dists**2 / (sigma**2))
        
        if mask is not None:
            psi = psi * mask
            
        psi_t = torch.zeros_like(psi)
        
        for t in range(T_wave):
            # Discrete Wave Equation Step (Verlet Integration or similar)
            # psi(t+1) = 2*psi(t) - psi(t-1) + c^2 * dt^2 * (-L * psi(t))
            # Actually simplest Euler for dimensionality:
            # d_psi_t = - c^2 * L * psi
            # psi += psi_t * dt
            # psi_t += d_psi_t * dt
            
            laplacian_psi = sparse_dense_mul(self.L, psi) if hasattr(self, 'sparse_dense_mul') else torch.sparse.mm(self.L, psi.unsqueeze(1)).squeeze()
            
            d_psi_t = - (c**2) * laplacian_psi
            psi = psi + psi_t * dt
            psi_t = psi_t + d_psi_t * dt
            
            if mask is not None:
                psi = psi * mask
                psi_t = psi_t * mask
        
        return psi, psi_t

    def telegrapher_step(self, psi: torch.Tensor, psi_t: torch.Tensor, T_damp: int = 20, gamma: float = 0.5, dt: float = 0.1, mask: torch.Tensor = None):
        """
        Step 2: Telegrapher's Eq (Selection)
        Add damping term: u_tt + gamma*u_t = c^2 * L * u
        """
        u = psi.clone()
        u_t = psi_t.clone()
        
        for t in range(T_damp):
            laplacian_u = sparse_dense_mul(self.L, u) if hasattr(self, 'sparse_dense_mul') else torch.sparse.mm(self.L, u.unsqueeze(1)).squeeze()
            
            # u_tt = -c^2 L u - gamma u_t
            # Euler-Cromer
            u_tt = - laplacian_u - gamma * u_t
            u_t = u_t + u_tt * dt
            u = u + u_t * dt
            
            if mask is not None:
                u = u * mask
                u_t = u_t * mask
                
        return u

    def poisson_solve(self, u: torch.Tensor, percentile: float = 95.0, alpha: float = 0.1, epsilon: float = 1e-4, max_iter: int = 1000):
        """
        Step 3: Screened Poisson Equation (Reconstruction/Basin Finding)
        (Delta - alpha)*phi = -source
        source = ReLU(u - threshold)
        """
        # Define Source from accumulated energy u
        threshold = np.percentile(u.cpu().numpy(), percentile)
        # source = F.relu(u - threshold) # Soft threshold
        # Or stricter:
        source = torch.where(u > threshold, u, torch.zeros_like(u))
        
        # Solving (L + alpha I) phi = source iteratively?
        # Or simply iterating diffusion: phi_new = (source + alpha*phi_old + (1-alpha)*neighbors)
        # Actually (L + alpha) inverse is basically heat kernel smoothing of source.
        # Let's use Iterative Jacobi or simple heat smoothing which approximates inverse Laplacian
        
        phi = source.clone() # Initial guess
        
        # Let's effectively run heat diffusion on source for 'smoothing' to find basins
        # phi_t = -L phi + source?
        # Steady state of (L + alpha) phi = source -> phi = (L+alpha)^-1 source
        
        # Jacobi iteration for solving Ax = b where A = L + alpha I
        # L = I - W_norm (roughly). So A = (1+alpha)I - W_norm
        # x = (b + W_norm x) / (1+alpha)
        
        # Note: self.L is I - Dinv A.
        # So L + alpha I = (1+alpha)I - Dinv A.
        # (1+alpha)I phi - Dinv A phi = source
        # phi = (source + Dinv A phi) / (1+alpha)
        # And Dinv A phi = (I - L) phi
        
        for i in range(max_iter):
             # L_phi = L * phi
             L_phi = sparse_dense_mul(self.L, phi) if hasattr(self, 'sparse_dense_mul') else torch.sparse.mm(self.L, phi.unsqueeze(1)).squeeze()
             
             # (I - (I-DinvA)) = DinvA.  Actually I - L = DinvA.
             # So neighbor_sum = phi - L_phi
             neighbor_sum = phi - L_phi
             
             phi_new = (source + neighbor_sum) / (1.0 + alpha)
             
             diff = torch.norm(phi_new - phi)
             phi = phi_new
             if diff < epsilon:
                 break
                 
        return -phi # Potential wells are negative

    def extract_basins(self, phi: torch.Tensor, top_k: int = 3):
        """
        Extract local minima (basins) from the potential field phi.
        """
        # Simple greedy approach for MVP:
        # 1. Sort nodes by potential (lowest first)
        # 2. Pick lowest as basin center.
        # 3. Suppress neighbors within radius (or just neighbors in graph)
        # 4. Repeat.
        
        sorted_idx = torch.argsort(phi)
        
        basins = []
        visited = torch.zeros_like(phi, dtype=torch.bool)
        
        # Need adjacency for suppression? 
        # For MVP: Suppress nothing, just return top k lowest potential points as representatives
        # Better: Suppress if 'too close' in embedding?
        
        # Actually, let's just return the sorted indices and their "confidence" (mass)
        # Confidence = magnitude of u at that point (source strength)
        
        for idx in sorted_idx:
            if len(basins) >= top_k:
                break
            
            i = idx.item()
            if visited[i]:
                continue
                
            basins.append({
                "center": i,
                "potential": phi[i].item(),
                "representatives": [i] 
            })
            # ideally mark neighbors as visited
            # For now, just taking top distinct
            visited[i] = True 
            
        return basins

    def retrieve_basins(self, query_vec: torch.Tensor, top_k_basins: int = 3, wave_params={}, telegrapher_params={}, poisson_params={}):
        """
        Returns structured basin objects.
        """
        psi, psi_t = self.wave_step(query_vec, **wave_params)
        u = self.telegrapher_step(psi, psi_t, **telegrapher_params)
        
        # Calculate Mass/Confidence
        total_mass = u.sum()
        
        phi = self.poisson_solve(u, **poisson_params)
        basins = self.extract_basins(phi, top_k=top_k_basins)
        
        # Enrich with confidence
        for b in basins:
            idx = b['center']
            b['confidence'] = (u[idx] / total_mass).item()
            
        return basins

    def retrieve_refined_indices(self, query_vec: torch.Tensor, top_k: int = 5, **kwargs) -> torch.Tensor:
        """
        Two-Stage Retrieval (Physics-Based):
        1. Global Wave: Identify Candidate Basins.
        2. Local Wave: Re-run dynamics with Dirichlet BCs (Mask) to sharpen boundaries.
        Returns tensor of indices.
        """
        # 1. Global Wave (Coarse)
        # Use slightly more diffusive params to find *all* relevant areas
        coarse_wave_params = kwargs.get('wave_params', {}).copy()
        coarse_wave_params['T_wave'] = int(coarse_wave_params.get('T_wave', 10) * 1.5) # Broader
        
        # Get Candidate Basins (Top 3)
        candidates = self.retrieve_basins(query_vec, top_k_basins=3, 
                                        wave_params=coarse_wave_params,
                                        telegrapher_params=kwargs.get('telegrapher_params', {}),
                                        poisson_params=kwargs.get('poisson_params', {}))
        
        if not candidates:
            idx, _ = self.retrieve(query_vec, top_k=top_k, **kwargs)
            return idx
            
        # 2. Build Mask (Basin Centers + Neighborhood)
        # Ideally we use adjacency. For MVP without easy adj index, 
        # we'll approximate neighborhood by distance in embedding space OR just trust the basin representatives?
        # User said "restrict graph to those basins".
        # Let's use the 'candidates' representatives.
        # And to allow local diffusion, we need *some* space around them.
        # Let's say we unmask all nodes that are close to these centers?
        # Or simplest: Unmask the top N nodes from the Global Coarse pass potential?
        # Yes. Let's take top 50 nodes from Coarse Phi.
        
        # Extract Coarse Phi (We need to re-run or cache it. `retrieve_basins` calc'd it but didn't return full field.
        # Efficiency Trade-off: Just re-run or modify retrieve_basins.
        # Let's just re-run retrieve to get phi, it's fast.
        _, coarse_phi = self.retrieve(query_vec, top_k=50, 
                                    wave_params=coarse_wave_params, 
                                    telegrapher_params=kwargs.get('telegrapher_params', {}),
                                    poisson_params=kwargs.get('poisson_params', {}))
                                    
        # Create Mask: 1.0 for top 50 lowest potential nodes, 0.0 otherwise.
        # Actually, mask should be spatial. Top 50 in Phi *should* cover the basins.
        mask_indices = torch.argsort(torch.tensor(coarse_phi), descending=False)[:50]
        mask = torch.zeros(self.data.shape[0], device=self.device)
        mask[mask_indices] = 1.0
        
        # 3. Local Wave (Fine)
        # Re-run strict wave with mask
        # Use strict params (original kwargs)
        fine_wave_params = kwargs.get('wave_params', {}).copy()
        fine_wave_params['mask'] = mask
        
        fine_telegrapher_params = kwargs.get('telegrapher_params', {}).copy()
        fine_telegrapher_params['mask'] = mask
        
        # We need to expose retrieval method that accepts mask. `retrieve` calls steps.
        # But `retrieve` doesn't pass mask.
        # We need to update `retrieve` signature or manually call steps here.
        
        psi, psi_t = self.wave_step(query_vec, **fine_wave_params)
        u = self.telegrapher_step(psi, psi_t, **fine_telegrapher_params)
        phi = self.poisson_solve(u, **kwargs.get('poisson_params', {}))
        
        # 4. Extract Final Indices from sharpened field
        # Sort by lowest potential within the mask
        # (Outside mask u=0 -> phi might be flat or high)
        sorted_indices = torch.argsort(phi, descending=False)
        
        # Return top_k
        return sorted_indices[:top_k]

    def retrieve(self, query_vec: torch.Tensor, top_k: int = 10, **kwargs):
        """
        End-to-End Pipeline
        Returns indices and potential values
        """
        # Defaults
        wave_params = kwargs.get('wave_params', {'T_wave': 10, 'c': 1.0, 'sigma': 1.0})
        telegrapher_params = kwargs.get('telegrapher_params', {'T_damp': 20, 'gamma': 0.5})
        poisson_params = kwargs.get('poisson_params', {'alpha': 0.1})
        
        psi, psi_t = self.wave_step(query_vec, **wave_params)
        u = self.telegrapher_step(psi, psi_t, **telegrapher_params)
        phi = self.poisson_solve(u, **poisson_params)
        
        # Sort by potential (lowest (most negative) is best attractor)
        sorted_indices = torch.argsort(phi, descending=False)
        
        return sorted_indices[:top_k], phi
