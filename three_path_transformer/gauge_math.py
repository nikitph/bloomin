import torch
import numpy as np

class GaugeMath:
    """
    Implements Differential Geometry on the Semantic Manifold.
    Manifold: Probability Simplex with Fisher Information Metric.
    Isometry: Sqrt map -> Positive Orthant of Sphere S^{n-1}.
    """
    
    @staticmethod
    def to_sphere(p):
        """Map probability p to sphere coords psi = sqrt(p)"""
        # Ensure normalization
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-10)
        return torch.sqrt(p)
    
    @staticmethod
    def to_prob(psi):
        """Map sphere coords psi back to probability p = psi^2"""
        return psi ** 2
        
    @staticmethod
    def distance(p1, p2):
        """
        Fisher-Rao distance on simplex = Geodesic distance on sphere.
        d(p, q) = 2 arccos( <sqrt(p), sqrt(q)> )
        """
        psi1 = GaugeMath.to_sphere(p1)
        psi2 = GaugeMath.to_sphere(p2)
        
        dot = (psi1 * psi2).sum(dim=-1)
        dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        return 2 * torch.arccos(dot)
        
    @staticmethod
    def geodesic(p1, p2, t):
        """
        Point at time t along geodesic connecting p1 to p2.
        SLERP (Spherical Linear Interpolation) on psi1, psi2.
        """
        psi1 = GaugeMath.to_sphere(p1)
        psi2 = GaugeMath.to_sphere(p2)
        
        dot = (psi1 * psi2).sum(dim=-1)
        dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        omega = torch.arccos(dot)
        
        sin_omega = torch.sin(omega)
        # Avoid division by zero if points are identical
        if sin_omega < 1e-6:
            return p1
            
        term1 = torch.sin((1-t)*omega) / sin_omega
        term2 = torch.sin(t*omega) / sin_omega
        
        psi_t = term1 * psi1 + term2 * psi2
        return GaugeMath.to_prob(psi_t)
        
    @staticmethod
    def parallel_transport(v, p_start, p_end):
        """
        Parallel transport tangent vector v from T_{p_start} to T_{p_end}.
        Uses spherical parallel transport formula.
        
        v: Vector in Tangent Space of Sphere (orthogonal to psi_start)
        p_start, p_end: Points on simplex
        
        Note: The vector v provided by the user (embeddings['Blue'] - embeddings['Red']) 
        is a chord in the embedding space, NOT strictly a tangent vector.
        We must first project it to the tangent space of the sphere at psi_start.
        """
        psi_start = GaugeMath.to_sphere(p_start)
        psi_end = GaugeMath.to_sphere(p_end)
        
        # 1. Project v to tangent space at psi_start
        # T_x S = {v | <v, x> = 0}
        # v_tan = v - <v, psi_start> * psi_start
        v_tan = v - (v * psi_start).sum() * psi_start
        
        # 2. Transport formula
        # Transport v along geodesic from x to y on unit sphere:
        # P(v) = v - <v, y> / (1 + <x, y>) * (x + y)
        # Wait, this matches the reflection formula I recalled earlier.
        # Let's verify with simpler one:
        # P(v) = v - 2 * <v, k> * k, where k = (y+x)/|y+x| ? No.
        
        # Standard formula:
        # P_{x->y}(v) = v - \frac{\langle v, y \rangle}{1 + \langle x, y \rangle} (x + y)
        # Check: P(v) should be orthogonal to y.
        # <P(v), y> = <v, y> - <v, y>/(1+<x,y>) * (<x,y> + <y,y>)
        # Since <y,y>=1, <x,y>+1 cancels denominator.
        # <P(v), y> = <v, y> - <v, y> = 0. Correct.
        
        dot = (psi_start * psi_end).sum()
        denom = 1 + dot
        
        if denom < 1e-6:
            # Antipodal points? Transport undefined/ambiguous
            return v_tan # Fallback
            
        factor = (v_tan * psi_end).sum() / denom
        v_transported = v_tan - factor * (psi_start + psi_end)
        
        return v_transported

    @staticmethod
    def angle_between(v1, v2):
        """Angle between two vectors"""
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        cos_theta = (v1 * v2).sum() / (norm1 * norm2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return torch.arccos(cos_theta).item()
        
    @staticmethod
    def triangle_area(p1, p2, p3):
        """
        Area of spherical triangle defined by 3 probability distributions.
        Using Girard's Theorem: Area = A + B + C - pi
        where A, B, C are internal angles.
        """
        # We need tangent vectors along geodesics to compute angles
        # Tangent from p1 to p2: Proj_{p1}(psi2 - psi1)?
        # Initial velocity of geodesic starting at p1 towards p2
        
        def initial_tangent(u, v):
            # u, v are sphere points
            # Tangent at u towards v
            # t = v - <v,u>u, then normalize
            tangent = v - (v * u).sum() * u
            return tangent / (torch.norm(tangent) + 1e-9)
            
        psi1, psi2, psi3 = [GaugeMath.to_sphere(p) for p in [p1, p2, p3]]
        
        # Angles at vertices
        # Angle at p1: between t_12 and t_13
        t12 = initial_tangent(psi1, psi2)
        t13 = initial_tangent(psi1, psi3)
        angle_A = GaugeMath.angle_between(t12, t13)
        
        # Angle at p2: between t_23 and t_21
        t23 = initial_tangent(psi2, psi3)
        t21 = initial_tangent(psi2, psi1)
        angle_B = GaugeMath.angle_between(t23, t21)
        
        # Angle at p3: between t_31 and t_32
        t31 = initial_tangent(psi3, psi1)
        t32 = initial_tangent(psi3, psi2)
        angle_C = GaugeMath.angle_between(t31, t32)
        
        area = angle_A + angle_B + angle_C - np.pi
        return max(0.0, area)
