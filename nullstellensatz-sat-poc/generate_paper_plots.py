import torch
import numpy as np
import matplotlib.pyplot as plt
from protein_solver import BraidingProteinSolver

def generate_plots():
    # Sequence: Trp-Cage
    seq = "NLYIQWLKDGGPSSGRPPPS"
    solver = BraidingProteinSolver(seq)
    
    print("Running solver for plotting...")
    final_pos, energies = solver.solve(max_steps=2000, dt=0.01)
    
    # 1. Energy Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(energies, color='#00aaff', linewidth=2, label='Topological Energy (H)')
    plt.yscale('log')
    plt.title('Hamiltonian Energy Convergence (Topological Snap)', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Energy (log scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig('/Users/truckx/PycharmProjects/bloomin/nullstellensatz-sat-poc/energy_convergence_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 3D Structure Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = final_pos[:, 0], final_pos[:, 1], final_pos[:, 2]
    
    # Color coding based on residue type (Hydrophobic in red, Polar in blue)
    colors = []
    for char in solver.charges:
        if char == 1.0: colors.append('red')
        elif char == -1.0: colors.append('blue')
        else: colors.append('gray')
        
    ax.plot(x, y, z, color='black', alpha=0.4, linewidth=1)
    ax.scatter(x, y, z, c=colors, s=100, edgecolors='black')
    
    # Add indices
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], f'{i}', size=8)
        
    ax.set_title(f'Folded State: {seq}', fontsize=14)
    ax.set_axis_off()
    
    plt.savefig('/Users/truckx/PycharmProjects/bloomin/nullstellensatz-sat-poc/folded_3d_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
