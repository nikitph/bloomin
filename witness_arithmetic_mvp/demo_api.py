from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

def run_demo_api():
    # Public Read-Only API
    api_v1 = """
{
  "openapi": "3.0.0",
  "paths": {
    "/users": {
      "get": {
        "summary": "List users"
      }
    }
  }
}
"""

    # Secure Private API with PII
    api_v2 = """
{
  "openapi": "3.0.0",
  "security": [{"ApiKeyAuth": []}],
  "paths": {
    "/users": {
        "post": {
            "summary": "Create user",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "email": {"type": "string"},
                                "password": {"type": "string"}
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

    print("--- Comparing API Schemas ---")
    print("Schema A: Public Read-Only")
    print("Schema B: Secure Write API with PII")
    
    result = semantic_diff(api_v1, api_v2, domain="api")
    
    print("\nResult:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_demo_api()
