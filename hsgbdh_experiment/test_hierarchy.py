import torch
from hsgbdh.model import HSGBDH

def test_hierarchy():
    n = 64
    d = 16
    K = 3 # Levels: 64, 32, 16 neurons
    
    model = HSGBDH(n, d, K, block_size=16)
    
    # Input: (1, d)
    x = torch.randn(1, d)
    
    # Forward pass
    output = model(x)
    
    print("Output shape:", output.shape)
    # Output should be (1, n) - base level activations
    # Or (1, d)? 
    # HSGBDHLevel returns y_t (1, n_k).
    # So output should be (1, n_0) = (1, 64).
    
    assert output.shape == (1, n)
    print("SUCCESS: Hierarchy forward pass complete.")

if __name__ == "__main__":
    test_hierarchy()
