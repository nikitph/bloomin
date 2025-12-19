def synthesize(witnesses: set[str], with_proof=False) -> str:
    """
    Synthesizes an artifact (Python code) from a set of witnesses.
    If with_proof=True, appends a witness trace (metadata proof).
    """
    parts = []
    trace = []
    
    if "optimization" in witnesses:
        parts.append("# [LOGIC_FIELD] Optimized for small-input sequences")
        trace.append("optimization -> Found bits in Sketch")
        
    if "recursion" in witnesses:
        parts.append("""def factorial(n):
    if n <= 1: return 1
    return n * factorial(n-1)""")
        trace.append("recursion -> Found bits in Sketch")
        
    if "loop" in witnesses:
        parts.append("""def sum_list(items):
    total = 0
    for x in items:
        total += x
    return total""")
        trace.append("loop -> Found bits in Sketch")
        
    if "conditional" in witnesses:
        parts.append("""def is_positive(x):
    if x > 0:
        return True
    return False""")
        trace.append("conditional -> Found bits in Sketch")
    
    output = "\n\n".join(parts) if parts else "pass  # Empty semantic footprint"
    
    if with_proof:
        proof_header = "# [WITNESS_TRACE]\n" + "\n".join(f"# - {t}" for t in trace)
        output = proof_header + "\n\n" + output
        
    return output
