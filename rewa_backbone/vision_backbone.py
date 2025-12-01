import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helpers ----------
def pairwise_l2(a, b):
    return torch.cdist(a, b, p=2)

def spherical_distance(a, b):
    # a, b: (B, H, N, D)
    # einsum is convenient but let's be careful with shapes
    # The prompt used: sim = torch.einsum('bnd,bmd->bnm', a, b) which implies (B, N, D) inputs or similar
    # But inside attention we have (B, H, N, D).
    # Let's use standard matmul for (B, H, N, D)
    # a: (B, H, N, D), b: (B, H, N, D) -> (B, H, N, N)
    sim = torch.matmul(a, b.transpose(-1, -2))
    sim = torch.clamp(sim, -1+1e-6, 1-1e-6)
    return torch.acos(sim)

def poincare_distance(u, v):
    # Numerically stable Poincaré distance
    # u, v: (B, H, N, D)
    diff = u.unsqueeze(-2) - v.unsqueeze(-3)  # (B, H, N, 1, D) - (B, H, 1, N, D) -> (B, H, N, N, D)
    diff_norm_sq = diff.pow(2).sum(-1)
    
    u_norm = u.pow(2).sum(-1).unsqueeze(-1) # (B, H, N, 1)
    v_norm = v.pow(2).sum(-1).unsqueeze(-2) # (B, H, 1, N)
    
    denom = (1 - u_norm) * (1 - v_norm)
    arg = 1 + 2 * diff_norm_sq / (denom + 1e-8)
    arg = torch.clamp(arg, min=1.0 + 1e-6)
    return torch.acosh(arg)


# ================================================================
#  A.3 — Core REWA-Space Attention (Vision)
# ================================================================
class REWAVisionAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dim_e=64,    # Euclidean witness dimension
        dim_s=64,    # Spherical witness dimension
        dim_h=32,    # Hyperbolic witness dimension
        temperature_init=1.0
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # Standard QKV
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj_out = nn.Linear(inner_dim, dim)

        # Witness projections
        self.WE = nn.Linear(dim_head, dim_e)
        self.WS = nn.Linear(dim_head, dim_s)
        self.WH = nn.Linear(dim_head, dim_h)

        # Learnable geometric mixture weights (positive via softplus)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Euclidean
        self.beta  = nn.Parameter(torch.tensor(1.0))  # Spherical
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Hyperbolic

        self.temperature = nn.Parameter(torch.tensor(temperature_init))

    def forward(self, x):
        """
        x: (B, N, C), patches or tokens
        """
        B, N, C = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, heads, N, dim_head)
        q = q.view(B, N, self.heads, -1).transpose(1, 2)
        k = k.view(B, N, self.heads, -1).transpose(1, 2)
        v = v.view(B, N, self.heads, -1).transpose(1, 2)

        # ====================================================
        # 1. Create Witness Embeddings in 3 Geometries
        # ====================================================

        # (B, H, N, dim_e)
        WE_q = self.WE(q)
        WE_k = self.WE(k)

        # Spherical → normalize
        WS_q = F.normalize(self.WS(q), dim=-1)
        WS_k = F.normalize(self.WS(k), dim=-1)

        # Hyperbolic → squashed into Poincaré ball
        WH_q = torch.tanh(self.WH(q))
        WH_k = torch.tanh(self.WH(k))

        # ====================================================
        # 2. Compute pairwise witness distances
        # ====================================================
        dE = pairwise_l2(WE_q, WE_k)             # Euclidean
        dS = spherical_distance(WS_q, WS_k)      # Spherical (arccos)
        dH = poincare_distance(WH_q, WH_k)       # Hyperbolic

        # ====================================================
        # 3. Combine distances (positive weights)
        # ====================================================
        wE = F.softplus(self.alpha)
        wS = F.softplus(self.beta)
        wH = F.softplus(self.gamma)

        D = (wE * dE + wS * dS + wH * dH)

        # Convert distance → similarity
        logits = -D / F.softplus(self.temperature)

        # ====================================================
        # 4. Attention softmax
        # ====================================================
        attn = F.softmax(logits, dim=-1)

        # ====================================================
        # 5. Weighted sum
        # ====================================================
        out = torch.matmul(attn, v) # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, -1)

        return self.proj_out(out)


# ================================================================
# A.4 — REWA-ViT Block (MLP + Attention + Residuals)
# ================================================================
class REWAViTBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, heads=8, dim_head=64):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = REWAVisionAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dim_e=dim_head//2,
            dim_s=dim_head//2,
            dim_h=dim_head//4
        )

        hidden_dim = dim * mlp_ratio

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ================================================================
# A.5 — Full REWA-ViT Backbone
# ================================================================
class REWAVisionBackbone(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        heads=12,
        mlp_ratio=4
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2

        # CLS token
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            REWAViTBlock(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                heads=heads,
                dim_head=embed_dim // heads
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)

        # Patchify
        x = self.patch_embed(x)          # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1,2)  # (B, N, C)

        # prepend CLS
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positions
        x = x + self.pos

        # Transformer backbone
        for blk in self.layers:
            x = blk(x)

        # CLS embedding
        return self.norm(x[:,0])
