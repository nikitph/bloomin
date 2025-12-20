import argparse
import sys
from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

def main():
    parser = argparse.ArgumentParser(description="Witness Arithmetic Semantic Diff")
    parser.add_argument("file_a", help="Path to first file (Before)")
    parser.add_argument("file_b", help="Path to second file (After)")
    parser.add_argument("--domain", choices=["code", "iam", "k8s", "api"], default="code", 
                        help="Domain for semantic extraction (default: code)")
    
    args = parser.parse_args()
    
    with open(args.file_a, "r") as f:
        content_a = f.read()
        
    with open(args.file_b, "r") as f:
        content_b = f.read()
        
    result = semantic_diff(content_a, content_b, domain=args.domain)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
