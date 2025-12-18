import torch
from torch.optim import Optimizer
import torch.nn.functional as F
import math

class NavierSchrodinger(Optimizer):
    """
    Navier-Schrödinger Flow Optimizer (NSFO) - Final Architecture
    
    Phases:
    1. 'exploration' (Schrödinger): Quantum tunneling.
    2. 'exploitation' (Closed System): 
       - Stokes Flow
       - Poisson Anchor
       - Manifold Constraint (Gradient Projection)
       - Support Masking (Vacuum Reflection)
    """

    def __init__(self, params, 
                 nu_max=2.0, 
                 hbar_max=2.0, 
                 tau=200, 
                 dt=0.05,
                 epsilon=1e-8,
                 use_convection=False,
                 k_anchor=0.2,
                 gamma_normal=5.0,
                 use_manifold_constraint=False, # Upgrade #5
                 epsilon_support=1e-4,          # Upgrade #6
                 damping=0.1):                  # New damping param
        defaults = dict(nu_max=nu_max, hbar_max=hbar_max, tau=tau, dt=dt, 
                       epsilon=epsilon, use_convection=use_convection, 
                       k_anchor=k_anchor, gamma_normal=gamma_normal,
                       use_manifold_constraint=use_manifold_constraint,
                       epsilon_support=epsilon_support,
                       damping=damping)
        super(NavierSchrodinger, self).__init__(params, defaults)

    def _get_laplacian(self, tensor):
        if tensor.numel() <= 0:
            return torch.zeros_like(tensor)
        flat = tensor.view(1, 1, -1)
        avg = F.avg_pool1d(flat, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        laplace = flat - avg
        return laplace.view_as(tensor)

    def capture_anchor(self):
        """Captures the current parameter state as the basin anchor point."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['anchor'] = p.detach().clone()

    @torch.no_grad()
    def step(self, closure=None, phase='exploration'):
        """
        Performs a single optimization step.
        phase: 'exploration' or 'exploitation'
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt = group['dt']
            nu = group['nu_max'] 
            hbar = group['hbar_max']
            use_convection = group['use_convection']
            k_anchor = group['k_anchor']
            gamma_normal = group['gamma_normal']
            use_manifold_constraint = group['use_manifold_constraint']
            epsilon_support = group['epsilon_support']
            damping = group['damping']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                grad_norm = grad.norm()
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['u'] = torch.zeros_like(p)
                    # Initialize psi opposite to gradient to start moving downhill
                    state['psi'] = -torch.sign(grad) if grad_norm > 0 else torch.ones_like(p)
                    state['psi'].div_(state['psi'].norm() + 1e-10)

                u = state['u']
                psi = state['psi']
                
                state['step'] += 1
                
                # --- PHASE A: EXPLORATION (Schrödinger) ---
                if phase == 'exploration':
                    lap_psi = self._get_laplacian(psi)
                    delta_psi = dt * (lap_psi - grad * psi)
                    psi.add_(delta_psi)
                    psi.div_(psi.norm() + 1e-10)
                    
                    p.add_(dt * hbar * psi)
                    u.zero_()

                # --- PHASE B: EXPLOITATION (Closed System) ---
                elif phase == 'exploitation':
                    
                    # 4. Support Masking (Upgrade #6)
                    # Check if we are on valid manifold support
                    if grad_norm < epsilon_support:
                        # Vacuum / boundary reflection
                        u.mul_(-0.5) 
                        # Stop processing forces for this step to freeze/reflect
                        # But we still apply the velocity update below?
                        # If u became -0.5*old_u, we apply that.
                    else:
                        lap_u = self._get_laplacian(u)
                        
                        # 1. Stokes Flow + Anchor Force
                        force = -grad + nu * lap_u
                        
                        # Anchor Logic
                        if 'anchor' in state:
                            displacement = p - state['anchor']
                            anchor_force = k_anchor * displacement
                            force -= anchor_force
                            
                            dist = displacement.norm()
                            if dist > 1e-6:
                                n = displacement / dist
                            else:
                                n = torch.zeros_like(displacement)
                        else:
                            n = torch.zeros_like(p)
                        
                        if use_convection:
                            convection = u * grad
                            force -= convection
                        
                        # Update Velocity
                        delta_u = dt * force
                        u.add_(delta_u)
                        
                        # 2. Velocity Projection (Normal Damping)
                        if 'anchor' in state and n.norm() > 0:
                            u_dot_n = torch.sum(u * n)
                            u_normal = u_dot_n * n
                            u_tangent = u - u_normal
                            
                            decay_normal = math.exp(-gamma_normal * dt)
                            u_normal.mul_(decay_normal)
                            # Dynamic damping
                            u_tangent.mul_(1.0 - damping * dt)
                            
                            u.copy_(u_tangent + u_normal)
                        else:
                            # Dynamic damping fallback
                            u.mul_(1.0 - damping * dt)

                        # 3. Manifold Constraint (Gradient Projection)
                        if use_manifold_constraint:
                            if grad_norm > 1e-10:
                                g_unit = grad / grad_norm
                                u_parallel = torch.sum(u * g_unit) * g_unit
                                u.copy_(u_parallel)
                    
                    # Clamp velocity
                    u_norm = u.norm()
                    if u_norm > 10.0:
                        u.mul_(10.0 / (u_norm + 1e-6))
                    
                    p.add_(dt * u)

        return loss
