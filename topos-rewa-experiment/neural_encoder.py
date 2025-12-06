"""
Neural encoder for learning witness representations
"""

import torch
import torch.nn as nn


class NeuralEncoder(nn.Module):
    """
    Simple neural encoder that maps concept embeddings to witness distributions
    
    Input: Concept embedding (DIM_INPUT)
    Output: Witness distribution (N_WITNESSES)
    """
    
    def __init__(self, dim_input, n_witnesses, hidden_dim=128):
        super().__init__()
        
        self.dim_input = dim_input
        self.n_witnesses = n_witnesses
        
        # 2-layer MLP
        self.network = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_witnesses),
            nn.Softmax(dim=-1)  # Output is a probability distribution
        )
    
    def forward(self, x):
        """
        Args:
            x: Concept embedding (batch_size, dim_input) or (dim_input,)
        
        Returns:
            Witness distribution (batch_size, n_witnesses) or (n_witnesses,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            output = self.network(x)
            return output.squeeze(0)
        return self.network(x)
    
    def get_embedding_layer(self):
        """Get the first layer for visualization"""
        return self.network[0].weight.data
