import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperbolicSpace:
    """
    Poincaré ball model of hyperbolic space.
    Curvature c is -1/K^2 (where K is the radius).
    We use the convention c = -k where k > 0.
    Standard Poincaré ball has curvature -1.
    """
    
    def __init__(self, c=1.0, eps=1e-5):
        """
        Args:
            c: Curvature constant (c = -curvature). c=1 means curvature -1.
            eps: Numerical stability constant.
        """
        self.c = c
        self.eps = eps
        
    def _lambda_x(self, x):
        """Compute the conformal factor lambda_x = 2 / (1 - c * ||x||^2)."""
        x_sqnorm = torch.sum(x.pow(2), dim=-1, keepdim=True)
        return 2.0 / (1.0 - self.c * x_sqnorm).clamp(min=self.eps)

    def mobius_add(self, x, y):
        """
        Möbius addition in Poincaré ball: x (+) y.
        Formula:
        (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y
        -------------------------------------------
        1 + 2c<x,y> + c^2||x||^2||y||^2
        """
        x2 = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y2 = torch.sum(y.pow(2), dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        
        return num / denom.clamp(min=self.eps)

    def exp_map(self, x, v):
        """
        Exponential map at point x: Map vector v from T_x M to M.
        Exp_x(v) = x (+) ( tanh(sqrt(c)*lambda_x*||v||/2) * v / (sqrt(c)*||v||) )
        if x = 0: Exp_0(v) = tanh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = self.c ** 0.5
        
        # Tangent vector at 0 (simplified)
        # Using the property that we can map v to 0 then exp then map back via mobius add?
        # Standard formula: Exp_x(v) = x (+) (tanh(lambda_x * ||v|| / 2) * v / ||v||) for c=1
        
        # General formula:
        # u = tanh(sqrt(c) * lambda_x * ||v|| / 2) * v / (sqrt(c) * ||v||)
        # res = x (+) u
        
        lambda_x = self._lambda_x(x)
        scale = torch.tanh(sqrt_c * lambda_x * v_norm / 2) / (sqrt_c * v_norm)
        u = scale * v
        
        return self.mobius_add(x, u)
        
    def exp_map0(self, v):
        """
        Exponential map at origin (x=0).
        Exp_0(v) = tanh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = self.c ** 0.5
        
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return scale * v

    def log_map(self, x, y):
        """
        Logarithmic map at x: Map point y from M to T_x M.
        Log_x(y) = 2/ (sqrt(c)*lambda_x) * atanh(sqrt(c)*||-x (+) y||) * (-x (+) y) / ||-x (+) y||
        """
        sub = self.mobius_add(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = self.c ** 0.5
        lambda_x = self._lambda_x(x)
        
        scale = 2 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * sub_norm) / sub_norm
        return scale * sub
        
    def log_map0(self, y):
        """
        Logarithmic map at origin (x=0).
        Log_0(y) = atanh(sqrt(c)*||y||) * y / (sqrt(c)*||y||)
        """
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = self.c ** 0.5
        
        scale = torch.atanh(sqrt_c * y_norm) / (sqrt_c * y_norm)
        return scale * y

    def distance(self, x, y):
        """
        Hyperbolic distance d(x, y).
        d(x, y) = 2/sqrt(c) * atanh(sqrt(c) * ||-x (+) y||)
        Alternately: d(x, y) = 1/sqrt(c) * acosh(1 + 2c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))
        """
        # Using Mobius addition formula which is numerically more stable for autodiff?
        # d(x, y) = 2 arctanh(||-x (+) y||) if c=1
        
        sqrt_c = self.c ** 0.5
        sub = self.mobius_add(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp(min=self.eps, max=1.0 - self.eps)
        
        dist = 2 / sqrt_c * torch.atanh(sqrt_c * sub_norm)
        return dist

    def project(self, x, max_norm=None):
        """
        Project points to be within the Poincaré ball (norm < 1/sqrt(c)).
        Ideally norm < 1.0 for c=1.
        """
        if max_norm is None:
            max_norm = (1.0 / self.c ** 0.5) - self.eps
            
        norm = x.norm(dim=-1, keepdim=True)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)
