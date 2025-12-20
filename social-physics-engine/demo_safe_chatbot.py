import numpy as np
from constitutional_sdk import Boundary, ConstitutionalLayer

def demo_safe_chatbot():
    print("\nDEMO 2: Safe LLM / Secret Protection")
    
    # Simplified semantic space: 1D distance to secret.
    # Secret is at semantic_pos = 1.0. Lower is safer.
    
    # Boundary: No leaks. Strength = 5.0 -> r_s = 0.89
    secret_boundary = Boundary(
        name="no_secrets",
        threshold=10.0, # Secret semantic point
        strength=5.0, 
        gradient_fn=lambda s, get_distance=False: ( (10.0 - s[0]) if get_distance else np.array([-1.0]) )
    )
    
    sdk = ConstitutionalLayer([secret_boundary])
    
    # Mock LLM generation: tries to move semantic state toward 10.0
    current_semantic_state = np.array([0.0])
    print(f"Initial State: {current_semantic_state}")
    
    # Token generation loop
    for i in range(10):
        # LLM "wants" to output something risky (e.g. answering a prompt about the secret)
        intent = np.array([1.5]) 
        
        safe_action = sdk.project_action(current_semantic_state, intent)
        
        if not np.allclose(safe_action, intent):
            print(f"  TOKEN {i}: REDACTED. Semantic Pressure hit Event Horizon at {current_semantic_state[0]:.2f}")
        else:
            print(f"  TOKEN {i}: Safe output generated.")
            
        current_semantic_state += safe_action
        
    print(f"Final Semantic State: {current_semantic_state[0]:.2f} (Target was 10.0+)")

if __name__ == "__main__":
    demo_safe_chatbot()
