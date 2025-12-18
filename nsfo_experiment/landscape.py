import torch
import torch.nn as nn
import numpy as np

class Landscape(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, xy):
        raise NotImplementedError

class MultiWell(Landscape):
    """
    A 2D landscape with multiple local minima and a global funnel.
    V(x, y) = 0.1 * (x^2 + y^2) - 2.0 * (cos(3x) + cos(3y))
    """
    def __init__(self):
        super().__init__()
        # Initialize at a point that is NOT the global min
        # Global min is near (0,0). 
        # We start at (3, 3) to force traversal over barriers.
        self.start_point = torch.tensor([3.0, 3.0])

    def forward(self, x):
        # x is shape (..., dim)
        
        # Quadratic bowl: 0.5 * sum(x_i^2)
        bowl = 0.5 * torch.sum(x**2, dim=-1)
        
        # Egg crate: -2.0 * sum(cos(3*x_i))
        ripples = -2.0 * torch.sum(torch.cos(3 * x), dim=-1)
        
        return bowl + ripples

class Saddle(Landscape):
    """
    A classic saddle point function.
    V(x, y) = x^2 - y^2 (unbounded) -> lets use something bounded or standard Monkey Saddle.
    Standard: V(x,y) = x^3 - 3xy^2 (Monkey Saddle)
    Or simply: x^2 - y^2 but we want a minimum eventually.
    
    Let's use a function that has a saddle at (0,0) leading to minima.
    V(x, y) = (x^2 - 1)^2 + (y^2 - 1)^2 ? No that's 4 minima.
    
    Let's use the 'Beale' function or similar, but simplified.
    Two Gaussians?
    
    Let's stick to the "MultiWell" as the primary test as requested.
    """
    pass
