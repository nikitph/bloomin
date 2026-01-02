import torch
import torch.nn.functional as F
import numpy as np

class RADSField:
    """
    Reaction-Advection-Diffusion-Scale (RAD-S) Continuous Field.
    Implements the level 1 evolution dynamics for the unified reasoning pipeline.
    """
    def __init__(self, size, d=16, D=0.1, v='none', reaction=None, num_scales=1, **kwargs):
        """
        Args:
            size (int): Size of the spatial domain (1D or 2D).
            d (int): Feature dimension per point.
            D (float or list): Diffusion coefficient(s).
            v (str or float): Advection velocity parameter. 'none', 'rightward', or float value.
            reaction (callable or str): Reaction function f(u, v) or 'gray_scott'.
            num_scales (int): Number of hierarchy scales.
            **kwargs: Extra parameters like D_u, D_v for specific reactions.
        """
        self.size = size
        self.d = d
        self.D = D
        self.v_param = v
        self.reaction = reaction
        self.num_scales = num_scales
        
        # Store extra params (e.g., D_u, D_v)
        for k, val in kwargs.items():
            setattr(self, k, val)

        # Initialize field phi
        # Default to 2D (1, d, size, size)
        self.two_d = True
        self.phi = torch.randn(1, d, size, size) * 0.1 
        
        # Determine 1D vs 2D heuristic based on size or velocity
        # If size is small and v is rightward, maybe user wanted 1D?
        # But 'rightward' works in 2D (x-axis).
        # We will expose a way to force 1D if needed, but default 2D is safer for patterns.
        
        if reaction == 'gray_scott':
             # We need at least 2 channels. 
             # Initialize specific Gray-Scott state:
             # u=1 (everywhere), v=0 (everywhere) + small noise in center
             if d < 2:
                 raise ValueError("Gray-Scott requires d>=2")
                 
             self.phi = torch.zeros(1, d, size, size)
             self.phi[:, 0, :, :] = 1.0
             
             # Perturb center for v
             r = int(size * 0.1) if size > 20 else 2
             c = size // 2
             self.phi[:, 1, c-r:c+r, c-r:c+r] = 0.25
             
             # Add small noise
             self.phi[:, 0] += torch.randn(size, size) * 0.01
             self.phi[:, 1] += torch.randn(size, size) * 0.01

    @property
    def u(self):
        return self.phi[0, 0].detach().cpu().numpy()

    def laplacian(self, tensor):
        """
        Compute Laplacian using convolution.
        """
        if self.two_d:
            kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=tensor.device).view(1, 1, 3, 3)
            # Apply per channel
            d = tensor.shape[1]
            kernel = kernel.repeat(d, 1, 1, 1)
            return F.conv2d(tensor, kernel, padding=1, groups=d)
        else:
            # 1D Laplacian: [1, -2, 1]
            kernel = torch.tensor([1., -2., 1.], device=tensor.device).view(1, 1, 3)
            d = tensor.shape[1]
            kernel = kernel.repeat(d, 1, 1)
            return F.conv1d(tensor, kernel, padding=1, groups=d)

    def gradient_x(self, tensor):
        """
        Compute x-gradient (for advection/energy).
        """
        if self.two_d:
            kernel = torch.tensor([[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]], device=tensor.device).view(1, 1, 3, 3)
            d = tensor.shape[1]
            kernel = kernel.repeat(d, 1, 1, 1)
            return F.conv2d(tensor, kernel, padding=1, groups=d)
        else:
            kernel = torch.tensor([-0.5, 0., 0.5], device=tensor.device).view(1, 1, 3)
            d = tensor.shape[1]
            kernel = kernel.repeat(d, 1, 1)
            return F.conv1d(tensor, kernel, padding=1, groups=d)
            
    def gradient_y(self, tensor):
        if not self.two_d:
            return torch.zeros_like(tensor)
        kernel = torch.tensor([[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]], device=tensor.device).view(1, 1, 3, 3)
        d = tensor.shape[1]
        kernel = kernel.repeat(d, 1, 1, 1)
        return F.conv2d(tensor, kernel, padding=1, groups=d)

    def evolve_step(self, dt=0.01, field_override=None):
        """
        Perform one step of RAD-S evolution.
        φ_t+1 = φ_t + dt * (D∇²φ - v·∇φ + f(φ))
        """
        phi = field_override if field_override is not None else self.phi
        
        # 1. Diffusion: D∇²φ
        diff_term = self.laplacian(phi)
        
        if self.reaction == 'gray_scott':
            D_tensor = torch.ones_like(phi) * 0.1
            if hasattr(self, 'D_u') and hasattr(self, 'D_v'):
                D_tensor[:, 0] = self.D_u
                D_tensor[:, 1] = self.D_v
            diff_term = diff_term * D_tensor
        else:
            diff_term = diff_term * self.D

        # 2. Advection: -v·∇φ
        adv_term = torch.zeros_like(phi)
        if self.v_param == 'rightward': 
             v_mag = 1.0
             adv_term = -v_mag * self.gradient_x(phi)
             
        # 3. Reaction: f(φ)
        react_term = torch.zeros_like(phi)
        if self.reaction == 'gray_scott':
            u = phi[:, 0:1]
            v = phi[:, 1:2]
            F_rate = 0.055
            k_rate = 0.062
            
            uv2 = u * (v ** 2)
            du = -uv2 + F_rate * (1.0 - u)
            dv = uv2 - (F_rate + k_rate) * v
            
            react_term = torch.cat([du, dv] + [torch.zeros_like(phi[:, 2:])], dim=1)
        
        # Update
        new_phi = phi + dt * (diff_term + adv_term + react_term)
        
        if field_override is None:
            self.phi = new_phi
            
        return new_phi

    def step(self, dt=0.01):
        self.evolve_step(dt)

    def find_stable_regions(self, threshold=0.1):
        """
        Identify regions where local variance is high (patterns).
        """
        if self.reaction == 'gray_scott':
             v_map = self.phi[0, 1]
             mask = v_map > threshold
             return mask.nonzero() 
        return []

    def build_scale_hierarchy(self):
        """
        Create coarser scales using avg pooling (R_lambda).
        """
        scales = []
        current = self.phi
        scales.append(current[0, 0].detach().cpu().numpy()) # Store finest
        
        for _ in range(self.num_scales - 1):
            if current.shape[-1] < 2:
                break
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            scales.append(current[0, 0].detach().cpu().numpy())
            
        return scales
