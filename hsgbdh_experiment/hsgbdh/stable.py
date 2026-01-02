import torch
import torch.nn as nn
import torch.nn.functional as F

class LogSpaceTropicalSemiring:
    """
    Work in log-space to prevent overflow
    
    Instead of: a (+) b = max(a, b)
                a (x) b = a * b
    
    Use:        log(a) (+) log(b) = max(log(a), log(b))
                log(a) (x) log(b) = log(a) + log(b)
    """
    def __init__(self, base=10.0):
        self.base = float(base)
    
    def to_log_space(self, G):
        """Convert raw weights to log-space"""
        # G must be non-negative.
        # Handle zeros safely.
        return torch.log(G + 1e-18) / torch.log(torch.tensor(self.base))
    
    def from_log_space(self, G_log):
        """Convert back from log-space"""
        # base^x
        return torch.pow(self.base, G_log)
    
    def compose(self, G_log_a, G_log_b):
        """
        Composition in log-space
        log(a*b) = log(a) + log(b)
        Matrix Multiplication in (Max, +) semiring.
        C_ij = Max_k (A_ik + B_kj)
        """
        n = G_log_a.shape[0]
        
        # Optimized implementation using broadcasting
        # (N, N) (x) (N, N)
        # Expand A: (N, 1, N)
        # Expand B: (1, N, N)
        # Sum: (N, N, N) -> A_ik + B_kj
        # Max over k (axis 2 of A / axis 0 of B? Wait.)
        # A_expanded: (N, 1, N) -> indices (i, 1, k)
        # B_expanded: (1, N, N) -> indices (1, j, k) -- Wait, standard matmul sums over middle dim.
        # A[i,k], B[k,j].
        # Let's align k.
        # A: (N, N, 1) -> (i, k, 1)
        # B: (1, N, N) -> (1, k, j)
        # Sum: (N, N, N) -> (i, k, j) with value A_ik + B_kj
        # Max over dim 1 (k): result (N, N) -> (i, j)
        
        A_metrics = G_log_a.unsqueeze(2) # (N, N, 1) -> i,k
        B_metrics = G_log_b.unsqueeze(0) # (1, N, N) -> k,j
        
        # We want to broadcast k against k?
        # A: (N, N) -> i, k
        # B: (N, N) -> k, j
        
        # C_ij = max_k (A_ik + B_kj)
        # sum shape: (N, N, N)
        sum_tensor = A_metrics + B_metrics 
        
        # Max over k (dimension 1)
        G_result, _ = torch.max(sum_tensor, dim=1)
        
        return G_result
    
    def max_op(self, G_log_a, G_log_b):
        """Max operation in log-space (stays the same)"""
        return torch.maximum(G_log_a, G_log_b)

def stable_transitive_closure(G, max_hops=5, temperature=1.0, decay=0.9):
    """
    Compute transitive closure without numerical explosion
    
    Args:
        G: Initial graph (n x n)
        max_hops: Maximum reasoning depth
        temperature: Softmax temperature for normalization
        decay: Per-hop weight decay
    
    Returns:
        G_star: Transitive closure (stable, normalized)
    """
    # Debug G range
    # print(f"G min: {G.min()}, max: {G.max()}, mean: {G.mean()}")
    
    # Pre-scale G to avoid numerical underflow in log space if too small
    # But log space handles small numbers well (-Inf).
    
    semiring = LogSpaceTropicalSemiring(base=10.0)
    
    # Check for NaNs in input
    if torch.isnan(G).any():
        print("NaN in input G!")
    
    G_log = semiring.to_log_space(G)
    
    # print(f"G_log range: {G_log.min()}, {G_log.max()}")
    
    G_star_log = G_log.clone()
    
    # Store penalty pattern
    decay_log = torch.log(torch.tensor(decay + 1e-18)) / torch.log(torch.tensor(10.0))
    
    for hop in range(1, max_hops):
        # Compose
        G_new_log = semiring.compose(G_star_log, G_log)
        
        # Apply length penalty
        G_new_log = G_new_log + decay_log
        
        # Normalize?
        # If we normalize every step, we might kill the growing path if it's not dominant yet.
        # Softmax normalization makes the row sum to 1. 
        # But we are tracking "connectivity strength", not probability.
        # If we force sum constant, then 10 paths = 0.1 each. 1 path = 1.0.
        # Removing per-step normalization to see if it fixes the signal drop.
        # User's code had: G_new_log = G_new_log - max_val * T. 
        # This keeps the MAX value unchanged? No, log(exp(x-max)) = x-max.
        # max_val = logsumexp(x). x_new = x - max_val.
        # sum(exp(x_new)) = sum(exp(x) / sum(exp(x))) = 1.
        # Normalize (prevent explosion even in log-space)
        if temperature > 0:
            # Normalize so that the MAX outgoing edge in each row is 1.0 (log 0)
            # This preserves the strength of the "best path" and prevents decay due to branching.
            # Using max instead of logsumexp.
            max_val, _ = torch.max(G_new_log, dim=-1, keepdim=True)
            G_new_log = G_new_log - max_val
        
        # Update (max operation)
        G_star_log = semiring.max_op(G_star_log, G_new_log)
        
    # Convert back to regular space
    G_star = semiring.from_log_space(G_star_log)
    
    # Final normalization (0-1 range)
    # Check for infs
    if torch.isinf(G_star).any():
        # If inf, set to 1.0 (max confidence)?
        # Or just clamp.
        G_star[torch.isinf(G_star)] = 1e38 # Float max proxy
        
    global_max = G_star.max()
    if global_max > 0:
        G_star = G_star / global_max
    else:
        # If all zero, remain zero
        pass
    
    return G_star

def damped_closure(G, max_hops=10, self_loop_penalty=0.5, temperature=1.0, decay=0.9):
    """
    Explicitly separate self-loops from transitions and penalize self-loops
    before computing closure, to emphasize reasoning chains over persistence.
    """
    # Extract transition matrix (off-diagonal)
    G_transition = G.clone()
    G_transition.fill_diagonal_(0)  # Remove self-loops
    
    # Extract self-loops (diagonal)
    G_self = torch.diag(torch.diag(G))
    
    # Penalize self-loops
    G_self = G_self * self_loop_penalty
    
    # Reconstruct
    G_damped = G_transition + G_self
    
    return stable_transitive_closure(G_damped, max_hops=max_hops, temperature=temperature, decay=decay)

def multi_timescale_closure(G, max_hops=10, temperature=1.0):
    """
    Apply distance-dependent decay using iterative power method.
    Decay schedule: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70...]
    """
    semiring = LogSpaceTropicalSemiring(base=10.0)
    
    # Check inputs
    if torch.isnan(G).any():
        print("NaN in input G (multiscale)!")
    
    G_log = semiring.to_log_space(G)
    
    # G_power tracks paths of length 'hop' exactly
    G_power_log = G_log.clone()
    
    # G_star tracks best path found so far (of any length)
    G_star_log = G_log.clone()
    
    # Hardcoded schedule for the experiment
    # decay_schedule = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    # Log decay values
    schedule = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    log_schedule = [torch.log(torch.tensor(d + 1e-18)) / torch.log(torch.tensor(10.0)) for d in schedule]
    last_decay = log_schedule[-1]
    
    # Pre-apply decay to 1-hop? 
    # User's code applies decay inside loop (starts hop=1).
    # If hop=1 (initial G), decay is 0.95.
    # So G_star should be decayed too? 
    # Let's apply decay to G_star initially for consistency? 
    # Or just let G_star be raw G (hop 1) and assume G_power loop starts from hop 2?
    # User loop: for hop in range(1, max_hops).
    # hop=1 implies we are computing G^2 (from G^1 * G).
    # decay index: min(hop-1, ...). If hop=1, index 0 (0.95).
    # So G^2 gets 0.95 decay. G^1 (initial) has NO decay.
    # This preserves immediate signals.
    
    for hop in range(1, max_hops):
        # Compose G_power * G -> Next Power
        # G_power represents G^hop. We want G^{hop+1}.
        G_next_log = semiring.compose(G_power_log, G_log)
        
        # Apply decay
        s_idx = min(hop - 1, len(log_schedule) - 1)
        decay = log_schedule[s_idx]
        G_next_log = G_next_log + decay
        
        # Normalize G_next?
        # User code: G_star = G_star / max.
        # Stabilizing the POWER matrix prevents it from exploding/vanishing.
        # If we effectively track probability mass, we should normalize.
        if temperature > 0:
             max_val, _ = torch.max(G_next_log, dim=-1, keepdim=True)
             G_next_log = G_next_log - max_val
        
        # Update Accumulator
        G_star_log = semiring.max_op(G_star_log, G_next_log)
        
        # Move forward
        G_power_log = G_next_log
        
    # Reconstruct
    G_star = semiring.from_log_space(G_star_log)
    
    # Final Norm
    if torch.isinf(G_star).any():
        G_star[torch.isinf(G_star)] = 1e38
    global_max = G_star.max()
    if global_max > 0:
        G_star = G_star / global_max
        
    return G_star

