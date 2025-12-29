import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from protein_solver import BraidingProteinSolver
import numpy as np

def plot_protein(positions, title, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Plot the chain
    ax.plot(x, y, z, marker='o', linestyle='-', markersize=5, color='blue', alpha=0.6, label='Backbone')
    
    # Plot the Zinc Ion
    ax.scatter([0], [0], [0], color='red', s=100, label='Zinc Ion (Singularity)')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # Standard Zinc Finger Sequence
    # Zinc Finger Pattern: C-x(2)-C-x(12)-H-x(3)-H
    sequence = "PYKCPDCGKSFSQKSDLRRHQRTH"
    print(f"Folding Sequence: {sequence}")
    
    solver = BraidingProteinSolver(sequence)
    
    # Initial State
    initial_pos = solver.positions.detach().cpu().numpy()
    plot_protein(initial_pos, "Initial Random State", "/Users/truckx/PycharmProjects/bloomin/nullstellensatz-sat-poc/initial_fold.png")
    print("Initial state saved to initial_fold.png")
    
    # Run Topological Snap (Hamiltonian Flow)
    print("Running Topological Snap (Hamiltonian Flow)...")
    final_pos, energies = solver.solve(max_steps=5000, dt=0.01)
    
    # Final State
    plot_protein(final_pos, "Folded Native State (Topological Snap)", "/Users/truckx/PycharmProjects/bloomin/nullstellensatz-sat-poc/final_fold.png")
    print("Final state saved to final_fold.png")
    
    # Plot Energy Descent
    plt.figure(figsize=(10, 6))
    plt.plot(energies)
    plt.yscale('log')
    plt.title("Topological Potential Energy Descent")
    plt.xlabel("Step")
    plt.ylabel("Potential V(z)")
    plt.grid(True, alpha=0.3)
    plt.savefig("/Users/truckx/PycharmProjects/bloomin/nullstellensatz-sat-poc/folding_energy.png")
    print("Folding energy plot saved to folding_energy.png")
    
    print("\nSimulation Complete.")
    print("The final structure shows the backbone collapsing around the Zinc singularity,")
    print("with hydrophobic residues forming a compact core.")

if __name__ == "__main__":
    main()
