import torch
import torch.nn as nn
import torch.nn.functional as F

class TropicalMatrixMul(nn.Module):
    """
    Implements Tropical (Min-Plus) Matrix Multiplication.
    y = M @ x  =>  y_i = min_j (M_ij + x_j)
    """
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # M is the weight matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * init_scale)

    def forward(self, x):
        # x: [batch, in_features]
        # weight: [out_features, in_features]
        
        # Expand x to [batch, 1, in_features]
        x_expanded = x.unsqueeze(1)
        
        # Expand weight to [1, out_features, in_features]
        w_expanded = self.weight.unsqueeze(0)
        
        # Sum: [batch, out_features, in_features]
        sum_term = x_expanded + w_expanded
        
        # Min over in_features: [batch, out_features]
        out, _ = torch.min(sum_term, dim=2)
        
        return out

class WitnessExtractor(nn.Module):
    def __init__(self, input_dim, witness_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, witness_dim),
            nn.Sigmoid() # Witnesses are "activations" or "probabilities"
        )
        
    def forward(self, x):
        return self.net(x)

class TacitReasoner(nn.Module):
    def __init__(self, num_nodes, embedding_dim=32, witness_dims=[16, 32, 64], code_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # Hierarchical Extractors
        # Level 1: Input -> W1
        self.extractor_l1 = WitnessExtractor(embedding_dim * 2, witness_dims[0])
        
        # Level 2: Input + W1 -> W2
        self.extractor_l2 = WitnessExtractor(embedding_dim * 2 + witness_dims[0], witness_dims[1])
        
        # Level 3: Input + W2 -> W3
        self.extractor_l3 = WitnessExtractor(embedding_dim * 2 + witness_dims[1], witness_dims[2])
        
        # Tropical Encoder
        total_witness_dim = sum(witness_dims)
        self.tropical_encoder = TropicalMatrixMul(total_witness_dim, code_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output distance
        )

    def forward(self, u, v):
        u_emb = self.embedding(u)
        v_emb = self.embedding(v)
        
        # Base input: concatenation of start and end node embeddings
        base_input = torch.cat([u_emb, v_emb], dim=1)
        
        # Level 1
        w1 = self.extractor_l1(base_input)
        
        # Level 2 (conditioned on W1)
        l2_input = torch.cat([base_input, w1], dim=1)
        w2 = self.extractor_l2(l2_input)
        
        # Level 3 (conditioned on W2 - skipping W1 for direct dependency on previous level, 
        # or could include all. Let's just chain them as per user sketch)
        l3_input = torch.cat([base_input, w2], dim=1)
        w3 = self.extractor_l3(l3_input)
        
        # Aggregate Witnesses
        all_witnesses = torch.cat([w1, w2, w3], dim=1)
        
        # Implicit Reasoning via Tropical Encoding
        code = self.tropical_encoder(all_witnesses)
        
        # Decode
        out = self.decoder(code)
        return out

class DirectModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, u, v):
        u_emb = self.embedding(u)
        v_emb = self.embedding(v)
        x = torch.cat([u_emb, v_emb], dim=1)
        return self.net(x)
