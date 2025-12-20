from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

class ProductFormatter:
    @staticmethod
    def print_diff(title, result):
        print(f"\n📢 {title}")
        distance = result["semantic_distance"]
        added = result["added_semantics"]
        removed = result["removed_semantics"]
        
        if distance == 0:
            print(f"   [=] No semantic change (Pure Refactor)")
        else:
            for w in added:
                print(f"   [+] Added: {w}")
            for w in removed:
                print(f"   [-] Removed: {w}")

def run_product_demo():
    print("🚀 Witness Arithmetic Product Demo")
    
    # Scene 1: The "No-Op" Refactor
    code_a = "def foo(x):\n  return x + 1"
    code_b = "def foo( x ) :\n    # formatting change\n    return x+1"
    res1 = semantic_diff(code_a, code_b, domain="code")
    ProductFormatter.print_diff("Scenario 1: Code Reformatting", res1)

    # Scene 2: The Security Regression
    k8s_safe = """
apiVersion: v1
kind: PersistentVolumeClaim
spec:
  encryption: true
"""
    k8s_risk = """
apiVersion: v1
kind: PersistentVolumeClaim
spec:
  encryption: false
"""
    res2 = semantic_diff(k8s_safe, k8s_risk, domain="k8s")
    ProductFormatter.print_diff("Scenario 2: Infra Security Regression", res2)
    
    # Scene 3: The API Breaking Change
    api_v1 = """
{
  "paths": {
    "/users": {
        "post": {
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "id": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
  }
}
"""
    api_v2 = """
{
  "paths": {
    "/users": {
        "post": {
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "id": {"type": "string"}
                            },
                            "required": ["id"]
                        }
                    }
                }
            }
        }
    }
  }
}
"""
    res3 = semantic_diff(api_v1, api_v2, domain="api")
    ProductFormatter.print_diff("Scenario 3: API Breaking Change", res3)

if __name__ == "__main__":
    run_product_demo()
