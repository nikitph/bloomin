import argparse
import sys
from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

def main():
    parser = argparse.ArgumentParser(description="Witness Arithmetic Semantic Diff")
    parser.add_argument("file_a", help="Path to first file (Before)")
    parser.add_argument("file_b", help="Path to second file (After)")
    
    args = parser.parse_args()
    
    with open(args.file_a, "r") as f:
        code_a = f.read()
        
    with open(args.file_b, "r") as f:
        code_b = f.read()
        
    result = semantic_diff(code_a, code_b)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
