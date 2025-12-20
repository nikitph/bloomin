import numpy as np
from typing import List, Dict, Tuple, Callable

def constitutional_metric(theta_vec: np.ndarray, boundaries: List) -> np.ndarray:
    """
    g_ij(theta) = delta_ij + sum_k alpha_k (grad phi_k outer grad phi_k) / phi_k^2
    """
    n = len(theta_vec)
    g = np.eye(n)
    
    # We map theta_vec index 0 to 'child_safety_risk' and index 1 to 'time_until_deadline'
    state = {'child_safety_risk': theta_vec[0], 'time_until_deadline': theta_vec[1]}
    
    for b in boundaries:
        phi = b.distance(state)
        grad_phi = b.gradient(state)
        alpha = b.strength
        
        if phi > 0:
            barrier = alpha * np.outer(grad_phi, grad_phi) / (phi ** 2)
            g += barrier
            
    return g

def numerical_derivative_metric(theta: np.ndarray, boundaries: List, eps: float = 1e-4) -> np.ndarray:
    """
    dg[i,j,k] = d g_ij / d theta_k
    """
    n = len(theta)
    dg = np.zeros((n, n, n))
    
    for k in range(n):
        plus = theta.copy()
        plus[k] += eps
        g_plus = constitutional_metric(plus, boundaries)
        
        minus = theta.copy()
        minus[k] -= eps
        g_minus = constitutional_metric(minus, boundaries)
        
        dg[:, :, k] = (g_plus - g_minus) / (2 * eps)
    return dg

def christoffel_symbols(g: np.ndarray, dg: np.ndarray) -> np.ndarray:
    n = g.shape[0]
    g_inv = np.linalg.inv(g)
    Gamma = np.zeros((n, n, n))
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    Gamma[i, j, k] += 0.5 * g_inv[i, m] * (dg[m, j, k] + dg[m, k, j] - dg[j, k, m])
    return Gamma

def numerical_derivative_christoffel(theta: np.ndarray, boundaries: List, eps: float = 1e-4) -> np.ndarray:
    """
    dGamma[i,j,k,l] = d Gamma^i_jk / d theta_l
    """
    n = len(theta)
    dGamma = np.zeros((n, n, n, n))
    
    for l in range(n):
        plus = theta.copy()
        plus[l] += eps
        g_p = constitutional_metric(plus, boundaries)
        dg_p = numerical_derivative_metric(plus, boundaries)
        G_p = christoffel_symbols(g_p, dg_p)
        
        minus = theta.copy()
        minus[l] -= eps
        g_m = constitutional_metric(minus, boundaries)
        dg_m = numerical_derivative_metric(minus, boundaries)
        G_m = christoffel_symbols(g_m, dg_m)
        
        dGamma[:, :, :, l] = (G_p - G_m) / (2 * eps)
    return dGamma

def riemann_tensor(Gamma: np.ndarray, dGamma: np.ndarray) -> np.ndarray:
    """
    R^i_jkl = d_k Gamma^i_jl - d_l Gamma^i_jk + sum_m (Gamma^i_mk Gamma^m_jl - Gamma^i_ml Gamma^m_jk)
    """
    n = Gamma.shape[0]
    R = np.zeros((n, n, n, n))
    
    for i in range(n):
      for j in range(n):
        for k in range(n):
          for l in range(n):
            R[i, j, k, l] = dGamma[i, j, l, k] - dGamma[i, j, k, l]
            for m in range(n):
              R[i, j, k, l] += (Gamma[i, m, k] * Gamma[m, j, l] - Gamma[i, m, l] * Gamma[m, j, k])
    return R

def ricci_tensor(R: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    Ric = np.zeros((n, n))
    for j in range(n):
        for l in range(n):
            for i in range(n):
                Ric[j, l] += R[i, j, i, l]
    return Ric

def ricci_scalar(Ric: np.ndarray, g: np.ndarray) -> float:
    g_inv = np.linalg.inv(g)
    return float(np.sum(g_inv * Ric))

def role_tension_tensor(theta: np.ndarray, roles: List) -> np.ndarray:
    """
    R_ij = sum w_r F_i F_j - (sum w_r F_i)(sum w_s F_j)
    """
    n = len(theta)
    R = np.zeros((n, n))
    forces = []
    weights = []
    
    for r in roles:
        F = r['gradient'](theta)
        w = r['weight']
        forces.append(F)
        weights.append(w)
        R += w * np.outer(F, F)
        
    F_total = np.zeros(n)
    W_total = sum(weights)
    for f, w in zip(forces, weights):
        F_total += w * f
    F_mean = F_total / max(0.001, W_total)
    R -= np.outer(F_mean, F_mean)
    return R

def solve_field_equations(theta: np.ndarray, boundaries: List, roles: List, kappa: float = 1.0) -> np.ndarray:
    """
    C_ij = kappa * R_ij
    Returns the residual C_ij - kappa * R_ij as a measure of equilibrium.
    """
    g = constitutional_metric(theta, boundaries)
    dg = numerical_derivative_metric(theta, boundaries)
    Gamma = christoffel_symbols(g, dg)
    dGamma = numerical_derivative_christoffel(theta, boundaries)
    R_t = riemann_tensor(Gamma, dGamma)
    Ric = ricci_tensor(R_t)
    R_s = ricci_scalar(Ric, g)
    
    C_ij = Ric - 0.5 * g * R_s
    R_ij = role_tension_tensor(theta, roles)
    
    return C_ij - kappa * R_ij
