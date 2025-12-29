import random

def generate_random_3sat(n_vars, n_clauses):
    """
    Generates a random 3-SAT instance.
    n_vars: Number of variables (1 to n_vars)
    n_clauses: Number of clauses
    Returns: List of lists, each sublist is a clause containing 3 literals.
    """
    clauses = []
    for _ in range(n_clauses):
        vars_sampled = random.sample(range(1, n_vars + 1), 3)
        clause = []
        for v in vars_sampled:
            # Randomly negate with 0.5 probability
            literal = v if random.random() > 0.5 else -v
            clause.append(literal)
        clauses.append(clause)
    return clauses

def to_dimacs(n_vars, clauses):
    """
    Converts clauses to DIMACS CNF format string.
    """
    lines = [f"p cnf {n_vars} {len(clauses)}"]
    for clause in clauses:
        lines.append(f"{' '.join(map(str, clause))} 0")
    return "\n".join(lines)

if __name__ == "__main__":
    # Example: Near phase transition N=20, M=85
    N = 20
    M = 85
    clauses = generate_random_3sat(N, M)
    print(to_dimacs(N, clauses))
