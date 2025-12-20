from witness_arithmetic_mvp.ast_extractor import extract_witnesses
import unittest

class TestWitnessInvariance(unittest.TestCase):
    
    def test_formatting_invariance(self):
        """Test that whitespace/formatting changes don't affect witnesses."""
        code_a = """
def func(x):
    if x > 10:
        return x
"""
        code_b = """
def func(   x   ):
    if x>10: return x
"""
        w1 = extract_witnesses(code_a)
        w2 = extract_witnesses(code_b)
        
        self.assertEqual(w1, w2, f"Witnesses differed for formatting: {w1} vs {w2}")
        self.assertIn("conditional", w1)
        # self.assertIn("gt", w1) # Assuming we added 'gt' to output
        
    def test_variable_renaming_invariance(self):
        """Test that renaming variables doesn't affect structural witnesses."""
        code_a = """
def sort(arr):
    for i in range(len(arr)):
        pass
"""
        code_b = """
def sort(my_list):
    for k in range(len(my_list)):
        pass
"""
        w1 = extract_witnesses(code_a)
        w2 = extract_witnesses(code_b)
        
        self.assertEqual(w1, w2, f"Witnesses differed for renaming: {w1} vs {w2}")
        self.assertIn("loop", w1)

    def test_comment_invariance(self):
        """Test that comments are ignored."""
        code_a = "x = x + 1"
        code_b = "x = x + 1 # increment"
        
        w1 = extract_witnesses(code_a)
        w2 = extract_witnesses(code_b)
        self.assertEqual(w1, w2)

if __name__ == '__main__':
    unittest.main()
