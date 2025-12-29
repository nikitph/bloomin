import torch
import torch.nn as nn
import numpy as np

class BraidingProteinSolver:
    def __init__(self, sequence, singularity_pos=None, device="cpu"):
        self.sequence = sequence
        self.device = device
        self.n_aa = len(sequence)
        
        # 1. THE LIFT: Sequence -> Topological Charges
        self.charges = self.map_sequence_to_braid_charges(sequence)
        
        # 2. INITIAL STATE: Randomly entangled 3D path
        self.positions = torch.randn(self.n_aa, 3, device=device) * 3.0
        self.positions.requires_grad_(True)
        self.velocities = torch.zeros(self.n_aa, 3, device=device)
        
        # Optional Singularity (Zinc, or Hydrophobic Center-of-Mass)
        self.singularity_pos = singularity_pos if singularity_pos is not None else torch.zeros(3, device=device)

    def map_sequence_to_braid_charges(self, sequence):
        charges = []
        for aa in sequence:
            if aa in ['C', 'H']: charges.append('ANCHOR')
            elif aa in ['L', 'I', 'V', 'F', 'W', 'Y', 'M']: charges.append(1.0) # Hydrophobic
            elif aa in ['S', 'T', 'N', 'Q', 'D', 'E', 'K', 'R', 'P']: charges.append(-1.0) # Polar/Surface
            else: charges.append(0.0) # Flexible
        return charges

    def compute_topological_potential(self, pos):
        # 1. BONDING: Strong local constraints
        diffs = pos[1:] - pos[:-1]
        bond_energy = torch.sum((torch.norm(diffs, dim=1) - 3.8)**2)
        
        # 2. LOCAL TOPOLOGY (Secondary Structure)
        # Res i and i+4 (Alpha), i and i+10 (Beta loops)
        helix_energy = 0.0
        if self.n_aa > 4:
            h_diffs = pos[4:] - pos[:-4]
            helix_energy += torch.sum((torch.norm(h_diffs, dim=1) - 5.4)**2)
        
        if self.n_aa > 10:
            l_diffs = pos[10:] - pos[:-10]
            helix_energy += 0.5 * torch.sum((torch.norm(l_diffs, dim=1) - 12.0)**2)
            
        # 3. EXCLUDED VOLUME (Pauli Wall)
        dist_matrix = torch.cdist(pos, pos)
        mask = torch.eye(self.n_aa, device=self.device) == 0
        repulsive_energy = torch.sum(torch.exp(-2.0 * (dist_matrix[mask] - 3.0)))
        
        # 4. LONG-RANGE TOPOLOGICAL COLLAPSE
        # Hydrophobic core should be global
        hydro_indices = [i for i, c in enumerate(self.charges) if c == 1.0]
        if hydro_indices:
            core_pos = torch.mean(pos[hydro_indices], dim=0)
            hydro_energy = 0.0
            for i in hydro_indices:
                # Quadratic pull to the global center of mass
                hydro_energy += torch.sum((pos[i] - core_pos)**2)
        else:
            hydro_energy = 0.0
            
        # 5. SINGULARITY ANCHOR (Zinc, etc.)
        anchor_energy = 0.0
        for i, char in enumerate(self.charges):
            if char == 'ANCHOR':
                anchor_energy += torch.sum((pos[i] - self.singularity_pos)**2)
        
        return 5.0 * bond_energy + helix_energy + 0.1 * repulsive_energy + 1.5 * hydro_energy + 10.0 * anchor_energy

    def solve(self, max_steps=2000, dt=0.01):
        energies = []
        pos_curr = self.positions.detach().clone().requires_grad_(True)
        vel_curr = self.velocities.detach().clone()
        
        for step in range(max_steps):
            v = self.compute_topological_potential(pos_curr)
            if pos_curr.grad is not None: pos_curr.grad.zero_()
            v.backward()
            grad = pos_curr.grad.detach()
            
            # Hamiltonian Leapfrog
            vel_half = vel_curr - 0.5 * dt * grad
            pos_next = (pos_curr + dt * vel_half).detach().requires_grad_(True)
            
            v_next = self.compute_topological_potential(pos_next)
            if pos_next.grad is not None: pos_next.grad.zero_()
            v_next.backward()
            grad_next = pos_next.grad.detach()
            
            vel_next = (vel_half - 0.5 * dt * grad_next) * 0.98 # Annealing
            
            pos_curr, vel_curr = pos_next, vel_next
            energies.append(v_next.item())
                
        return pos_curr.detach().cpu().numpy(), energies

if __name__ == "__main__":
    # Zinc Finger Sequence
    seq = "PYKCPDCGKSFSQKSDLRRHQRTH"
    solver = BraidingProteinSolver(seq)
    final_pos, _ = solver.solve(max_steps=100)
    print("Folded positions (first 5):")
    print(final_pos[:5])
