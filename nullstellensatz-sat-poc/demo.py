from generator import generate_random_3sat
from solver import NullstellensatzSolver
from eigen_solver import EigenSatSolver
from symplectic_solver import SymplecticSATSolver
import matplotlib.pyplot as plt
import numpy as np
import time

def run_experiment(N, M, name):
    print(f"\n=== Experiment: {name} (N={N}, M={M}) ===")
    clauses = generate_random_3sat(N, M)
    
    # 1. Nullstellensatz Solver
    print("Running Nullstellensatz Solver...")
    start_n = time.time()
    ns_solver = NullstellensatzSolver(N, clauses)
    z_ns, ns_energies = ns_solver.solve(max_iters=1000)
    ns_success, (_, ns_count) = ns_solver.check_solution(z_ns)
    ns_time = time.time() - start_n
    
    # 2. Eigen-SAT Solver
    print("Running Eigen-SAT Solver...")
    start_e = time.time()
    eigen_solver = EigenSatSolver(N, clauses)
    e_assignment, l_min = eigen_solver.solve()
    ei_success, (_, ei_count) = eigen_solver.check_solution(e_assignment)
    ei_time = time.time() - start_e
    
    # 3. Symplectic SAT Solver
    print("Running Symplectic SAT Solver...")
    start_s = time.time()
    symp_solver = SymplecticSATSolver(N, clauses)
    z_symp, symp_energies = symp_solver.solve(max_steps=2000, dt=0.01)
    sy_success, (_, sy_count) = symp_solver.check_solution(z_symp)
    sy_time = time.time() - start_s
    
    print(f"Nullstellensatz: {'PASS' if ns_success else 'FAIL'} ({ns_count}/{M}) in {ns_time:.2f}s")
    print(f"Eigen-SAT:      {'PASS' if ei_success else 'FAIL'} ({ei_count}/{M}) in {ei_time:.4f}s")
    print(f"Symplectic:     {'PASS' if sy_success else 'FAIL'} ({sy_count}/{M}) in {sy_time:.2f}s")
    
    return ns_energies, symp_energies, ns_success, sy_success

def main():
    # Run experiments with varying complexity
    configs = [
        (20, 85, "Phase Transition"),
        (50, 213, "Hard Large (Target: 213/213)")
    ]
    
    plt.figure(figsize=(12, 8))
    
    for N, M, name in configs:
        ns_e, sy_e, ns_s, sy_s = run_experiment(N, M, name)
        plt.plot(ns_e, label=f"NS-{name} ({'P' if ns_s else 'F'})", linestyle="--")
        plt.plot(sy_e, label=f"Symp-{name} ({'P' if sy_s else 'F'})")
    
    plt.yscale("log")
    plt.xlabel("Step / Iteration")
    plt.ylabel("Energy (V)")
    plt.title("Nullstellensatz vs Symplectic Hamiltonian Flow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("/Users/truckx/PycharmProjects/bloomin/nullstellensatz-sat-poc/energy_flow_v3.png")
    
    print("\nBenchmark complete. Plot saved to energy_flow_v3.png")

if __name__ == "__main__":
    main()
