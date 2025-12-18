from witness_arithmetic_mvp.ast_extractor import extract_witnesses
import ast

code = "x - 1"
print(f"Code: {code}")
w = extract_witnesses(code)
print(f"Witnesses: {w}")

code2 = "def foo(x): return foo(x-1)"
print(f"Code: {code2}")
w2 = extract_witnesses(code2)
print(f"Witnesses: {w2}")

# Debug internal AST structure
tree = ast.parse(code)
print(ast.dump(tree, indent=2))
