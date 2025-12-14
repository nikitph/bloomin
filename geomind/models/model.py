import torch
import torch.nn as nn
from geomind.core.geometry import HyperbolicSpace
from geomind.core.ricci import RicciFlowLayer
from geomind.attention.witness_attn import WitnessAttention

class GeoMindBlock(nn.Module):
    """
    A single block of the Geometric Reasoning Machine.
    Composition:
    1. Witness-Based Attention (Sparse, Discrete-to-Continuous)
    2. Residual connection (in tangent space / via Mobius add)
    3. LayerNorm (Riemannian adaptation or standard)
    4. Ricci Flow (Geometric smoothing / FFN equivalent)
    5. Norm
    """
    def __init__(self, dim, num_witnesses=64, dropout=0.1):
        super().__init__()
        self.attn = WitnessAttention(dim, num_witnesses=num_witnesses)
        self.ricci = RicciFlowLayer(dim)
        
        # Using standard norms for prototype simplicity
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 1. Attention
        # x is currently treated as vector in Euclidean space for operations (Tangent space approx)
        # Ideally we should use Mobius addition for residuals: x (+) attn_out
        
        attn_out, _ = self.attn(x, x, x, mask=mask)
        x = x + self.dropout(attn_out) # Residual
        x = self.norm1(x)
        
        # 2. Ricci Flow (replaces FFN)
        flow_out = self.ricci(x)
        x = x + self.dropout(flow_out - x) # flow_out is new x, so residual is implied or explicit?
        # RicciFlowLayer returns x_new. 
        # If we utilize residual around it: x + (flow(x) - x) -> flow(x).
        # Let's rely on RicciFlowLayer's internal update logic.
        
        x = self.norm2(flow_out)
        
        return x

class GeoMind(nn.Module):
    """
    Full GeoMind Model.
    """
    def __init__(self, vocab_size, dim=256, depth=6, num_witnesses=32, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.hyperbolic = HyperbolicSpace()
        
        # 1. Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim) # Standard pos emb for now
        
        # 2. Layers
        self.layers = nn.ModuleList([
            GeoMindBlock(dim, num_witnesses=num_witnesses) for _ in range(depth)
        ])
        
        # 3. Head
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, x):
        # x: (B, S) - tokens
        B, S = x.shape
        
        # Embed
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, S, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        x = tok_emb + pos_emb
        
        # Map to Hyperbolic (just before first layer? OR operate in tangent space mainly?)
        # For this prototype, we treat the vectors as living in the tangent space of the origin
        # and only project or use hyperbolic logic where explicitly needed (like attention).
        # But WitnessAttention currently uses vector math.
        # So we stay in "pseudo-Euclidean" representational space that approximates the manifold.
        
        # Mask for causal attention
        # (B, S, S)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(B, -1, -1) # True where masked
        # Attention expects mask where True means KEEP? Or standard PyTorch?
        # My WitnessAttention: scores.masked_fill(mask == 0, -inf) -> 1 is keep, 0 is mask.
        
        # Create causal mask where 1 is visible, 0 is future.
        causal_mask = torch.tril(torch.ones(S, S, device=x.device))
        
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
            
        x = self.norm_f(x)
        logits = self.head(x)
        
        return logits
