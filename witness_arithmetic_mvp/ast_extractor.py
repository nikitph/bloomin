import ast
import ast as py_ast

def extract_witnesses(code: str) -> set[str]:
    """
    Extracts semantic witnesses from code using robust AST parsing.
    Stable against formatting and variable renaming.
    """
    witnesses = set()
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to heuristics if AST parsing fails (e.g. incomplete code)
        print("Warning: AST parsing failed, returning empty set for this chunk.")
        return witnesses

    class WitnessVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None
            self.function_calls = set()
            
        def visit_If(self, node):
            witnesses.add("conditional")
            
            # Check for small input optimization pattern: len(x) < N
            if isinstance(node.test, ast.Compare):
                # Check for len() call
                has_len = False
                if isinstance(node.test.left, ast.Call) and isinstance(node.test.left.func, ast.Name):
                    if node.test.left.func.id == "len":
                        has_len = True
                
                # Check for < or <= operators
                has_lt = False
                for op in node.test.ops:
                    if isinstance(op, (ast.Lt, ast.LtE)):
                        has_lt = True
                        
                if has_len and has_lt:
                    witnesses.add("small_input_optimization")
            
            self.generic_visit(node)

        def visit_For(self, node):
            witnesses.add("loop")
            self.generic_visit(node)

        def visit_While(self, node):
            witnesses.add("loop")
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Identify algorithms by name
            if "quicksort" in node.name:
                witnesses.add("quicksort")
            if "mergesort" in node.name:
                witnesses.add("mergesort")
            if "insertion_sort" in node.name:
                witnesses.add("insertion_sort")
            if "binary_search" in node.name:
                witnesses.add("binary_search")

            old_function = self.current_function
            self.current_function = node.name
            
            self.generic_visit(node)
            self.current_function = old_function

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                
                # Check for recursion
                if self.current_function and func_name == self.current_function:
                    witnesses.add("recursion")
                
                # Check for known algorithms called
                if "quicksort" in func_name:
                    witnesses.add("quicksort")
                if "mergesort" in func_name:
                    witnesses.add("mergesort")
                if "insertion_sort" in func_name:
                    witnesses.add("insertion_sort")
            
            self.generic_visit(node)

        def visit_Return(self, node):
             self.generic_visit(node)
             
        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Add): witnesses.add("add")
            elif isinstance(node.op, ast.Sub): witnesses.add("sub")
            elif isinstance(node.op, ast.Mult): witnesses.add("mul")
            elif isinstance(node.op, ast.Div): witnesses.add("div")
            elif isinstance(node.op, ast.Mod): witnesses.add("mod")
            elif isinstance(node.op, ast.Pow): witnesses.add("pow")
            self.generic_visit(node)

        def visit_Compare(self, node):
            for op in node.ops:
                if isinstance(op, ast.Eq): witnesses.add("eq")
                elif isinstance(op, ast.NotEq): witnesses.add("neq")
                elif isinstance(op, ast.Lt): witnesses.add("lt")
                elif isinstance(op, ast.Gt): witnesses.add("gt")
                elif isinstance(op, ast.LtE): witnesses.add("lte")
                elif isinstance(op, ast.GtE): witnesses.add("gte")
            self.generic_visit(node)
            
        def visit_BoolOp(self, node):
             if isinstance(node.op, ast.And): witnesses.add("and")
             elif isinstance(node.op, ast.Or): witnesses.add("or")
             elif isinstance(node.op, ast.Not): witnesses.add("not")
             self.generic_visit(node)

    visitor = WitnessVisitor()
    visitor.visit(tree)
    
    return witnesses
