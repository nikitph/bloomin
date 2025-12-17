import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from .utils import build_knn_graph, sparse_dense_mul

class WaveRetrievalEngine:
    def __init__(self, data: torch.Tensor, k_neighbors: int = 10, use_cuda: bool = False):
        """
        Initializes the retrieval engine.
        
        Args:
            data: (N, D) tensor of embeddings.
            k_neighbors: number of neighbors for graph construction.
            use_cuda: whether to use GPU.
        """
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.N = data.shape[0]
        
        # Build Laplacian (L)
        # Using a normalized graph Laplacian
        print("Building Graph Laplacian...")
        self.L = build_knn_graph(self.data.cpu(), k=k_neighbors).to(self.device)
        print("Graph built.")
        
    def wave_step(self, query_vec: torch.Tensor, T_wave: int = 50, dt: float = 0.1, c: float = 1.0, sigma: float = 10.0):
        """
        Step 1: Wave Equation (Global Scan)
        PDE: partial_tt psi = c^2 * L @ psi
        Equation in discrete form (Leapfrog):
        psi_tt = -c^2 * L * psi  (Note: usually L is positive semi-definite for div-grad, so -L for Laplacian operator in physics?)
        Wait, for Graph Laplacian L_rw = I - D^-1 A.
        The continuous Laplacian is negative definite usually (spectra 0 to negative).
        Standard Graph Laplacian spectra is [0, \lambda_max].
        Physics wave eq: u_tt = c^2 \nabla^2 u.
        Graph analog: u_tt = -c^2 L u.
        """
        # Initial Excitation (Gaussian around query)
        # Find closest node to query to center the Gaussian (or soft-assign)
        # For simplicity, we find the single nearest neighbor in the manifold to inject the pulse
        dists = torch.cdist(query_vec.view(1, -1), self.data)
        closest_idx = torch.argmin(dists)
        
        # Initialize psi
        # psi[v] = exp(-dist(v, query)^2 / sigma^2)
        # In graph measurement, we can just use Euclidean dists for the initial pulse
        psi = torch.exp(-dists.view(-1)**2 / (2 * sigma**2)).to(self.device)
        psi_t = torch.zeros_like(psi).to(self.device)
        
        # Time integration
        for t in range(T_wave):
            # L is positive semi-definite (eigenvalues >= 0)
            # So -L corresponds to the spatial Laplacian \nabla^2
            psi_tt = -(c**2) * sparse_dense_mul(self.L, psi)
            psi_t = psi_t + dt * psi_tt
            psi = psi + dt * psi_t
            
        return psi, psi_t

    def telegrapher_step(self, psi: torch.Tensor, psi_t: torch.Tensor, T_damp: int = 50, dt: float = 0.1, c: float = 1.0, gamma: float = 0.5):
        """
        Step 2: Telegrapher's Equation (Damped Refinement)
        PDE: u_tt + gamma * u_t = c^2 \nabla^2 u
        """
        # Initialize from Wave State
        # u = abs(psi) # energy envelope (magnitude of wave)
        # We can also keep it real signed. The prompt says u = abs(psi).
        u = torch.abs(psi)
        u_t = psi_t # inherit velocity? or zero it? prompt says u_t = psi_t
        
        for t in range(T_damp):
            # u_tt = c^2 \nabla^2 u - gamma * u_t
            laplacian_u = -sparse_dense_mul(self.L, u)
            u_tt = (c**2) * laplacian_u - gamma * u_t
            
            u_t = u_t + dt * u_tt
            u = u + dt * u_t
            
        return u

    def poisson_solve(self, u: torch.Tensor, percentile: float = 95.0, alpha: float = 0.1, epsilon: float = 1e-4, max_iter: int = 1000):
        """
        Step 3 & 4: Source Extraction and Poisson Solve
        Delta phi = rho
        """
        # Step 3: Source Extraction (Soft Sources)
        # Old: threshold = np.percentile(u.cpu().numpy(), percentile)
        # New: rho = relu(u - mean(u))
        u_mean = u.mean()
        rho = F.relu(u - u_mean)
        
        # Step 4: Poisson Solve
        # L phi = rho -> phi = L_inv rho (pseudo-inverse)
        # Or iterative relaxation: phi_new = phi - alpha * (L phi - rho) ?
        # Wait, \Delta phi = rho. On graph, -L phi = rho.
        # So L phi = -rho.
        # Iterative: phi_{k+1} = phi_k - alpha * (L phi_k + rho)
        
        phi = torch.zeros_like(rho)
        rho_neg = -rho
        
        for i in range(max_iter):
            # Residual = L*phi - (-rho) = L*phi + rho
            L_phi = sparse_dense_mul(self.L, phi)
            residual = L_phi - rho_neg 
            
            phi_new = phi - alpha * residual
            
            if torch.norm(phi_new - phi) < epsilon:
                phi = phi_new
                break
            phi = phi_new
            
        return phi

    def retrieve(self, query_vec: torch.Tensor, top_k: int = 10, **kwargs):
        """
        End-to-End Pipeline
        """
        # 1. Wave
        psi, psi_t = self.wave_step(query_vec, **kwargs.get('wave_params', {}))
        
        # 2. Telegrapher
        u = self.telegrapher_step(psi, psi_t, **kwargs.get('telegrapher_params', {}))
        
        # 3. Poisson
        phi = self.poisson_solve(u, **kwargs.get('poisson_params', {}))
        
        # 4. Rank
        # We want "basins" of the potential phi.
        # If -L phi = rho, and rho is positive "mass", then phi should be a "gravity well".
        # High rho means deep potential well (negative value usually) if we define \Delta \phi = \rho.
        # But here we solved L \phi = - \rho. L is positive semi-definite.
        # Let's look at interpretation.
        # If rho is source, phi should be high near rho?
        # Electrostatics: \nabla^2 V = -rho/eps. V is peak (positive) near positive charge.
        # -L V = -rho => L V = rho.
        # Here we used L phi = -rho. So phi should be negative near rho (potential well).
        # We want to retrieve items in the well.
        # argsort(phi) (ascending) gives most negative values first (deepest wells).
        
        sorted_indices = torch.argsort(phi) # Ascending
        # The prompt says: results = argsort(phi)[:k]
        # This implies we are looking for minima of potential.
        
        return sorted_indices[:top_k], phi
