import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TropicalMatrixMul, WitnessExtractor

class TacitReasonerWithLosses(nn.Module):
    """
    Enhanced TacitReasoner with auxiliary losses to force structural learning.
    """
    def __init__(self, num_nodes, embedding_dim=32, witness_dims=[16, 32, 64], code_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # Hierarchical Extractors
        self.extractor_l1 = WitnessExtractor(embedding_dim * 2, witness_dims[0])
        self.extractor_l2 = WitnessExtractor(embedding_dim * 2 + witness_dims[0], witness_dims[1])
        self.extractor_l3 = WitnessExtractor(embedding_dim * 2 + witness_dims[1], witness_dims[2])
        
        # Tropical Encoder
        total_witness_dim = sum(witness_dims)
        self.tropical_encoder = TropicalMatrixMul(total_witness_dim, code_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Store witnesses for consistency loss
        self.last_witnesses = None
        self.last_nodes = None

    def forward(self, u, v, return_witnesses=False):
        u_emb = self.embedding(u)
        v_emb = self.embedding(v)
        
        base_input = torch.cat([u_emb, v_emb], dim=1)
        
        # Level 1
        w1 = self.extractor_l1(base_input)
        
        # Level 2
        l2_input = torch.cat([base_input, w1], dim=1)
        w2 = self.extractor_l2(l2_input)
        
        # Level 3
        l3_input = torch.cat([base_input, w2], dim=1)
        w3 = self.extractor_l3(l3_input)
        
        # Aggregate
        all_witnesses = torch.cat([w1, w2, w3], dim=1)
        
        # Store for auxiliary losses
        self.last_witnesses = all_witnesses
        self.last_nodes = (u, v)
        
        # Encode
        code = self.tropical_encoder(all_witnesses)
        
        # Decode
        out = self.decoder(code)
        
        if return_witnesses:
            return out, (w1, w2, w3)
        return out
    
    def get_node_witness(self, node_id):
        """Get the witness representation for a single node (using self-loop)."""
        # For witness consistency, we need node-level witnesses
        # We'll use the embedding as a proxy, or extract from (node, node) pair
        node_tensor = torch.tensor([node_id], device=self.embedding.weight.device)
        emb = self.embedding(node_tensor)
        
        # Extract W1 from self-pair
        base_input = torch.cat([emb, emb], dim=1)
        w1 = self.extractor_l1(base_input)
        return w1

def triangle_inequality_loss(model, u, v, w):
    """
    Enforce triangle inequality: d(u,w) <= d(u,v) + d(v,w)
    """
    d_uv = model(u, v).squeeze()
    d_vw = model(v, w).squeeze()
    d_uw = model(u, w).squeeze()
    
    # Violation of triangle inequality
    violation = F.relu(d_uw - d_uv - d_vw)
    return violation.mean()

def witness_consistency_loss(model, node_to_cluster):
    """
    Force nodes in the same cluster to have similar W1 witnesses.
    """
    device = model.embedding.weight.device
    
    # Group nodes by cluster
    clusters = {}
    for node, cluster in node_to_cluster.items():
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(node)
    
    total_loss = 0
    count = 0
    
    for cluster_id, nodes in clusters.items():
        if len(nodes) < 2:
            continue
            
        # Get witnesses for all nodes in cluster
        witnesses = []
        for node in nodes[:10]:  # Limit to 10 nodes per cluster for efficiency
            w = model.get_node_witness(node)
            witnesses.append(w)
        
        if len(witnesses) < 2:
            continue
            
        witnesses = torch.cat(witnesses, dim=0)
        
        # Compute centroid
        centroid = witnesses.mean(dim=0, keepdim=True)
        
        # Compute similarity to centroid (negative distance)
        distances = torch.norm(witnesses - centroid, dim=1)
        total_loss += distances.mean()
        count += 1
    
    return total_loss / max(count, 1)

if __name__ == "__main__":
    # Quick test
    model = TacitReasonerWithLosses(100)
    u = torch.tensor([0, 1, 2])
    v = torch.tensor([5, 6, 7])
    w = torch.tensor([10, 11, 12])
    
    out = model(u, v)
    print("Output shape:", out.shape)
    
    tri_loss = triangle_inequality_loss(model, u, v, w)
    print("Triangle loss:", tri_loss.item())
    
    node_to_cluster = {i: i // 20 for i in range(100)}
    cons_loss = witness_consistency_loss(model, node_to_cluster)
    print("Consistency loss:", cons_loss.item())
