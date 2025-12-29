import torch
import torch.nn as nn
import numpy as np

class SymplecticSATSolver:
    def __init__(self, n_vars, clauses, device="cpu"):
        self.n_vars = n_vars
        self.clauses = clauses
        self.device = device
        
        # Position z in [0, 1]
        self.z = torch.rand(n_vars, device=device, requires_grad=True)
        # Momentum p
        self.p = torch.randn(n_vars, device=device) * 0.1
        
    def compute_potential_energy(self, z_val):
        """
        P_C(z) = (1 - L1)(1 - L2)(1 - L3)
        Total potential V(z) = sum |P_C|^2 + reg
        """
        # Constrain z to [0, 1] using clamp during energy calc or sigmoid
        # But for Hamiltonian, we want the raw space or reflected space
        # We'll use the raw z and apply reflection at boundaries
        
        total_energy = 0.0
        for clause in self.clauses:
            clause_term = 1.0
            for lit in clause:
                idx = abs(lit) - 1
                if lit > 0:
                    val = 1.0 - z_val[idx]
                else:
                    val = z_val[idx]
                clause_term *= val
            total_energy += clause_term**2
            
        # Regulator to push towards 0/1
        reg_weight = 0.01
        reg_energy = torch.sum((z_val * (1.0 - z_val))**2)
        
        return total_energy + reg_weight * reg_energy

    def solve(self, max_steps=2000, dt=0.01):
        """
        Leapfrog Integration:
        p(t + dt/2) = p(t) - (dt/2) * grad V(z(t))
        z(t + dt) = z(t) + dt * p(t + dt/2)
        p(t + dt) = p(t + dt/2) - (dt/2) * grad V(z(t + dt))
        """
        energies = []
        z_curr = self.z.detach().clone().requires_grad_(True)
        p_curr = self.p.detach().clone()
        
        for step in range(max_steps):
            # 1. First half-step for momentum
            v = self.compute_potential_energy(z_curr)
            if z_curr.grad is not None:
                z_curr.grad.zero_()
            v.backward()
            grad = z_curr.grad.detach()
            
            p_half = p_curr - 0.5 * dt * grad
            
            # 2. Full-step for position
            z_next = (z_curr + dt * p_half).detach().requires_grad_(True)
            
            # 3. Boundary reflection (Symplectic reflection)
            # If z > 1: z_new = 2 - z, p_new = -p
            # If z < 0: z_new = -z, p_new = -p
            mask_upper = z_next > 1.0
            mask_lower = z_next < 0.0
            
            z_next.data[mask_upper] = 2.0 - z_next.data[mask_upper]
            p_half[mask_upper] *= -1.0
            
            z_next.data[mask_lower] = -z_next.data[mask_lower]
            p_half[mask_lower] *= -1.0
            
            # 4. Final half-step for momentum
            v_next = self.compute_potential_energy(z_next)
            if z_next.grad is not None:
                z_next.grad.zero_()
            v_next.backward()
            grad_next = z_next.grad.detach()
            
            p_next = p_half - 0.5 * dt * grad_next
            
            # Update state
            z_curr = z_next
            p_curr = p_next
            
            e_val = v_next.item()
            energies.append(e_val)
            
            if e_val < 1e-6:
                print(f"Symplectic converged at step {step}, Energy: {e_val:.8f}")
                break
                
            if step % 500 == 0:
                print(f"Step {step}, Energy: {e_val:.8f}")

        return z_curr.detach().cpu().numpy(), energies

    def check_solution(self, z_final):
        assignment = (z_final > 0.5).astype(int)
        satisfied_count = 0
        for clause in self.clauses:
            is_satisfied = False
            for lit in clause:
                idx = abs(lit) - 1
                val = assignment[idx]
                if lit > 0 and val == 1:
                    is_satisfied = True
                    break
                if lit < 0 and val == 0:
                    is_satisfied = True
                    break
            if is_satisfied:
                satisfied_count += 1
        return satisfied_count == len(self.clauses), (assignment, satisfied_count)

if __name__ == "__main__":
    # Test
    n_vars = 3
    clauses = [[1, 2, -3], [-1, -2, 3]]
    solver = SymplecticSATSolver(n_vars, clauses)
    z_final, _ = solver.solve()
    success, (_, count) = solver.check_solution(z_final)
    print(f"Success: {success}, Clauses: {count}/{len(clauses)}")
