import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

class EigenSatSolver:
    def __init__(self, n_vars, clauses):
        self.n_vars = n_vars
        self.clauses = clauses

    def build_signed_laplacian(self):
        """
        Constructs the Signed SAT-Laplacian L = D - W.
        W_ij = sum_{clauses C} sign(L_i, C) * sign(L_j, C)
        where sign(L_k, C) is +1 if x_k in C, -1 if -x_k in C, and 0 otherwise.
        """
        # Adjacency matrix W
        W = np.zeros((self.n_vars, self.n_vars))
        
        for clause in self.clauses:
            for i in range(len(clause)):
                for j in range(i + 1, len(clause)):
                    lit_i = clause[i]
                    lit_j = clause[j]
                    
                    idx_i = abs(lit_i) - 1
                    idx_j = abs(lit_j) - 1
                    
                    # Contribution to W_ij
                    # If signs are same (both pos or both neg), they "want" to be same: +1
                    # If signs are opposite, they "want" to be different: -1
                    # Note: x_i v x_j -> (1-x_i)(1-x_j)=0 -> x_i+x_j-x_i*x_j=0? 
                    # Actually, for spectral clustering/Cut logic:
                    # Same sign means they should have same assignment to satisfy at least one.
                    # Wait, if x1 v x2: (T, T), (T, F), (F, T) satisfy. (F, F) fails.
                    # If they have same sign in a clause, they are "cooperating".
                    # If opposite signs (x1 v -x2), (F, T) fails.
                    
                    val = 1 if np.sign(lit_i) == np.sign(lit_j) else -1
                    W[idx_i, idx_j] += val
                    W[idx_j, idx_i] += val
        
        # Diagonal matrix D contains absolute row-sums
        D = np.diag(np.sum(np.abs(W), axis=1))
        L = D - W
        return L

    def solve(self):
        L = self.build_signed_laplacian()
        
        # Ensure matrix is not all zeros (happens for 0 clauses)
        if np.all(L == 0):
            return np.zeros(self.n_vars), 0.0

        # Smallest eigenvalue and corresponding eigenvector
        # Use shift-invert mode to find smallest eigenvalue effectively
        try:
            # k=1 for smallest eigenvalue
            # which='SA' means smallest algebraic
            eigenvalues, eigenvectors = sla.eigsh(L, k=1, which='SA')
            v_truth = eigenvectors[:, 0]
            l_min = eigenvalues[0]
            
            # Assignment is the sign of the eigenvector
            # Project back to {0, 1}
            assignment = (v_truth > 0).astype(int)
            return assignment, l_min
        except Exception as e:
            print(f"Eigen solver error: {e}")
            return np.zeros(self.n_vars), 0.0

    def check_solution(self, assignment):
        satisfied_count = 0
        for clause in self.clauses:
            is_satisfied = False
            for lit in clause:
                var = abs(lit) - 1
                val = assignment[var]
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
    # Test on a simple 3-SAT instance
    n_vars = 3
    clauses = [[1, 2, -3], [-1, -2, 3]]
    solver = EigenSatSolver(n_vars, clauses)
    assignment, l_min = solver.solve()
    success, (_, count) = solver.check_solution(assignment)
    print(f"Success: {success}, Assignment: {assignment}, L_min: {l_min:.4f}, Clauses: {count}/{len(clauses)}")
