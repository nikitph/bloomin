import torch
import torch.nn.functional as F

class SymbolicReasoningEngine:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device='cpu'):
        """
        Symbolic Engine on Graph G = (V, E)
        edge_index: [2, E] tensor of source-target pairs
        """
        self.device = device
        self.edge_index = edge_index.to(device)
        self.num_nodes = num_nodes
        
        # Precompute degree and adjacency utilities if needed
        # For grad/div, we iterate over edges.
        
    def gradient(self, u: torch.Tensor):
        """
        Compute edge gradient: grad_u_ij = u[j] - u[i]
        Returns: [E] tensor
        """
        src, dst = self.edge_index
        return u[dst] - u[src]

    def divergence(self, edge_flux: torch.Tensor):
        """
        Compute node divergence: div_F_i = sum_{j in N(i)} F_ij
        edge_flux is F_ij (directed from src to dst)
        Returns: [N] tensor
        Note: If graph is undirected (symmetric edges), we just sum incoming?
        Actually divergence is net flow OUT.
        F_ij = flow i -> j.
        div_i = sum_j F_ij.
        We use scatter_add.
        """
        src, dst = self.edge_index
        # Flow leaving src
        out_flow = torch.zeros(self.num_nodes, device=self.device)
        out_flow.scatter_add_(0, src, edge_flux)
        
        # This assumes symmetric edges exist for j->i?
        # Graph Laplacian def usually: sum (u[i] - u[j]) for j in N(i)
        # grad_ij = u[j] - u[i].
        # We want sum of (u[i]-u[j])? That is -grad_ij.
        # Divergence of vector field F: sum_j F_ij?
        # Let's align with TV norm definition.
        # TV(u) = sum |grad_u_ij|.
        # Gradient descent flow: div(grad/|grad|).
        
        # Divergence operator D satisfies <grad u, F> = - <u, div F>.
        # <grad u, F> = sum_{ij} (u[j]-u[i]) * F_ij
        # = sum F_ij u[j] - sum F_ij u[i]
        # = <u, in_flux> - <u, out_flux>
        # = <u, in_flux - out_flux>
        # So div F = -(in_flux - out_flux) = out_flux - in_flux
        
        in_flow = torch.zeros(self.num_nodes, device=self.device)
        in_flow.scatter_add_(0, dst, edge_flux)
        
        return out_flow - in_flow

    def tv_flow_step(self, u: torch.Tensor, dt: float = 0.01, epsilon: float = 1e-5):
        """
        Total Variation Flow:
        du/dt = div( grad u / |grad u| )
        """
        grad_u = self.gradient(u)
        norm = torch.sqrt(grad_u**2 + epsilon**2)
        flux = grad_u / norm
        
        div_flux = self.divergence(flux)
        
        # Update
        return u + dt * div_flux

    def mean_curvature_flow_step(self, u: torch.Tensor, dt: float = 0.01, epsilon: float = 1e-5):
        """
        Mean Curvature Flow:
        du/dt = |grad u| * div( grad u / |grad u| )
        Note: |grad u| is defined on edges. Divergence is on nodes.
        We need a node-based |grad u|.
        Averaging incoming edge norms?
        Or: Standard formulation on graphs often simplifies.
        Strictly: Level Set Equation.
        Let's approximate node gradient norm as mean of incident edge gradient norms.
        """
        grad_u = self.gradient(u)
        norm_edge = torch.sqrt(grad_u**2 + epsilon**2)
        flux = grad_u / norm_edge
        
        curvature = self.divergence(flux) # Node-based curvature (kappa)
        
        # Compute node-based gradient norm |grad u|_i
        # sum(|grad_ij|) / degree? Or max?
        # TV flow is sum_j weights * sign(u_j - u_i) roughly.
        # MCF is usually just moving interface.
        # In image processing: u_t = |grad u| K.
        
        # Scatter add edge norms to nodes, then divide by degree
        src, dst = self.edge_index
        node_grad_sum = torch.zeros(self.num_nodes, device=self.device)
        node_grad_sum.scatter_add_(0, src, norm_edge)
        node_grad_sum.scatter_add_(0, dst, norm_edge) # Add both directions??
        
        # Counts
        degree = torch.zeros(self.num_nodes, device=self.device)
        ones = torch.ones_like(src, dtype=torch.float)
        degree.scatter_add_(0, src, ones)
        degree.scatter_add_(0, dst, ones)
        
        node_grad_norm = node_grad_sum / (degree + 1e-9)
        
        return u + dt * node_grad_norm * curvature

    def obstacle_projection(self, u: torch.Tensor, mask: torch.Tensor):
        """
        Enforce u >= mask (if mask has values).
        Or logic: u can be anything, but MUST be 1.0 where mask=1.0?
        User said: Forbidden forbidden regions -> u=0.
        "if mask[v] == FORBIDDEN: u[v] = 0.0" (Logic Invariant)
        Also seed regions: u[v] = 1.0 (Fixed BCs).
        
        Let's assume 'mask' contains values where u is constrained.
        mask values:
          0: Free
          1: Must be 1 (Seed)
         -1: Must be 0 (Forbidden)
        """
        if mask is None:
            return u
            
        # Hard constraints
        u = torch.where(mask == 1, torch.tensor(1.0, device=self.device), u)
        u = torch.where(mask == -1, torch.tensor(0.0, device=self.device), u)
        return u

    def solve(self, u0: torch.Tensor, mask: torch.Tensor = None, T_tv: int = 50, T_mc: int = 50, dt: float = 0.1):
        u = u0.clone()
        
        # 1. Total Variation (Denoising)
        for t in range(T_tv):
            u = self.tv_flow_step(u, dt=dt)
            u = self.obstacle_projection(u, mask) # Enforce constraints during flow
            
        # 2. Mean Curvature (Simplification)
        for t in range(T_mc):
            u = self.mean_curvature_flow_step(u, dt=dt)
            u = self.obstacle_projection(u, mask)
            
        return u
