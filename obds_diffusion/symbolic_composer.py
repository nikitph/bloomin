import torch
import sympy as sp
import numpy as np
from .model import OBDSDiffusion

class SymbolicComposer:
    """
    Compose polynomials symbolically for single-step sampling.
    Includes degree truncation to maintain O(1) sampling complexity relative to N.
    """
    def __init__(self, model: OBDSDiffusion):
        self.model = model
        self.max_degree = model.poly_score.max_degree
        self.n_dims = model.poly_score.n_dims
        self.device = next(model.parameters()).device
        self.composed_funcs = []
        
    def compose_steps(self, num_steps=10, truncate=True):
        """
        Compose N diffusion steps into a single set of polynomials.
        Args:
            num_steps: Number of diffusion steps to compose
            truncate: If True, truncates the polynomial to max_degree after each step.
                      This restricts the operator to the L3 polynomial ring, 
                      preventing exponential degree growth.
        """
        print(f"Composing {num_steps} steps symbolically (truncate={truncate})...")
        
        x_sym = sp.Symbol('x', real=True)
        
        # Get coefficients
        coeffs = self.model.poly_score.coeffs.detach().cpu().numpy() # [degree+1, n_dims]
        betas = self.model.betas.detach().cpu().numpy()
        
        dims_to_compose = min(self.n_dims, 50) # Optimization for PoC
        
        composed_funcs = []
        for d in range(dims_to_compose):
            # Start with identity: P(x) = x
            expr = x_sym
            
            step_indices = np.linspace(self.model.n_timesteps-1, 0, num_steps).astype(int)
            
            for t_idx in step_indices:
                t_norm = t_idx / self.model.n_timesteps
                beta = betas[t_idx]
                
                # Construct score polynomial for this step
                score_expr = 0
                for degree in range(self.max_degree + 1):
                    c = float(np.real(coeffs[degree, d])) # Take real part
                    score_expr += c * (x_sym ** degree) * (t_norm ** degree)
                
                # Compose: x_{t-1} = x_t + beta * score(x_t, t)
                # expr represents the map from x_T to x_t
                # We need x_{t-1}(x_T) = x_t(x_T) + beta * score(x_t(x_T), t)
                # So we substitute the *accumulated* expr into the score
                
                # Substitute: score(expr)
                current_score = score_expr.subs(x_sym, expr)
                
                # Update accumulated expression
                expr = expr + beta * current_score
                
                if truncate:
                    # Project back to degree K: truncate higher order terms
                    # Expand is needed to identify terms
                    expr = sp.expand(expr)
                    # Remove O(x^{d+1})
                    expr = expr + sp.O(x_sym**(self.max_degree + 1))
                    expr = expr.removeO()
            
            # Final simplifiction based on real assumption
            expr = sp.simplify(expr)
            
            # Convert to lambda
            func = sp.lambdify(x_sym, expr, modules='numpy')
            composed_funcs.append(func)
            
            if d % 10 == 0:
                print(f"Composed {d}/{dims_to_compose} dims...")
                
        self.composed_funcs = composed_funcs
        return composed_funcs

    @torch.no_grad()
    def sample(self, x_T):
        """
        Evaluate composed polynomials on a batch of noise.
        """
        x_np = x_T.cpu().numpy()
        batch_size = x_np.shape[0]
        x_0_np = np.zeros_like(x_np)
        
        dims_composed = len(self.composed_funcs)
        
        for d in range(dims_composed):
            x_0_np[:, d] = self.composed_funcs[d](x_np[:, d])
            
        return torch.tensor(x_0_np, dtype=torch.float32, device=self.device)
