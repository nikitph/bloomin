import torch
import torch.nn as nn
import torch.nn.functional as F
from .witnesses import WitnessEuclidean, WitnessSpherical, WitnessHyperbolic, WitnessFunctionalRFF, REWADistance

# ============================================================
# 1. 3D VIDEO PATCHIFIER
# ============================================================

class Patchify3D(nn.Module):
    """
    Splits video into (T, H, W) patches and produces patch embeddings.
    Input:  (B, C, T, H, W)
    Output: (B, N, D) where N = num_patches
    """
    def __init__(self, in_channels=3, patch_size=(2,16,16), embed_dim=768):
        super().__init__()
        # patch_size is (pt, ph, pw)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)   # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ============================================================
# 2. REWA MULTI-HEAD ATTENTION (VIDEO)
# ============================================================

class REWAMultiHeadAttention(nn.Module):
    """
    Multi-Head REWA Attention for Video Tokens
    Includes:
        - Euclidean witness
        - Spherical witness
        - Hyperbolic witness
        - Functional RFF witness
    """

    def __init__(
        self,
        dim,                # input dim
        num_heads=8,
        dim_euc=64,
        dim_sph=64,
        dim_hyp=64,
        dim_rff=128,
        rff_sigma=1.0,
        dropout=0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # --------------------------------------------------------
        # VALUE PROJECTIONS (Shared across REWA distance)
        # --------------------------------------------------------
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # --------------------------------------------------------
        # Witness extractors (one per head)
        # --------------------------------------------------------
        self.E_heads = nn.ModuleList([
            WitnessEuclidean(self.head_dim, dim_euc) for _ in range(num_heads)
        ])

        self.S_heads = nn.ModuleList([
            WitnessSpherical(self.head_dim, dim_sph) for _ in range(num_heads)
        ])

        self.H_heads = nn.ModuleList([
            WitnessHyperbolic(self.head_dim, dim_hyp) for _ in range(num_heads)
        ])

        self.F_heads = nn.ModuleList([
            WitnessFunctionalRFF(self.head_dim, dim_rff, sigma=rff_sigma)
            for _ in range(num_heads)
        ])

        # --------------------------------------------------------
        # REWA Distance per head
        # --------------------------------------------------------
        self.rewa_heads = nn.ModuleList([
            REWADistance(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, temperature=1.0)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------
    def forward(self, x, mask=None):
        """
        x:    (B, N, D)
        mask: (B, N) optional
        """

        B, N, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # --------------------------------------------------------
        # STEP 1: Split into heads
        # --------------------------------------------------------
        x_heads = x.view(B, N, H, Hd).transpose(1, 2)  # (B, H, N, Hd)

        # For collecting per-head outputs
        head_outputs = []

        # --------------------------------------------------------
        # STEP 2: Per-head REWA Attention
        # --------------------------------------------------------
        for h in range(H):
            x_h = x_heads[:, h]  # (B, N, Hd)

            # --- Witness extraction ---
            w_E = self.E_heads[h](x_h)
            w_S = self.S_heads[h](x_h)
            w_H = self.H_heads[h](x_h)
            w_F = self.F_heads[h](x_h)

            # --- REWA Distance ---
            dist = self.rewa_heads[h](w_E, w_S, w_H, w_F)  # (B, N, N)

            # --- Attention logits ---
            logits = -dist

            # Apply mask if available
            if mask is not None:
                # mask: (B, N)
                attn_mask = mask.unsqueeze(1).repeat(1, N, 1)  # (B, N, N)
                logits = logits.masked_fill(attn_mask == 0, float('-inf'))

            # --- Softmax ---
            attn = F.softmax(logits, dim=-1)  # (B, N, N)
            attn = self.dropout(attn)

            # --- Values ---
            v_full = self.value(x)  # (B, N, D)
            v_h = v_full.view(B, N, H, Hd)[:, :, h]  # (B, N, Hd)

            # --- Apply attention ---
            out_h = torch.bmm(attn, v_h)  # (B, N, Hd)

            head_outputs.append(out_h)

        # --------------------------------------------------------
        # STEP 3: Combine heads
        # --------------------------------------------------------
        out = torch.stack(head_outputs, dim=2)  # (B, N, H, Hd)
        out = out.reshape(B, N, D)             # (B, N, D)

        # Final projection
        out = self.out_proj(out)
        return out


# ============================================================
# 3. REWA VIDEO BLOCK (Inferred)
# ============================================================

class REWAVideoBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = REWAMultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 4. FULL REWA-VIDEO BACKBONE (Inferred)
# ============================================================

class REWAVideoBackbone(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=(2, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=16,
        img_size=224,
        num_classes=400
    ):
        super().__init__()
        
        self.patch_embed = Patchify3D(in_channels, patch_size, embed_dim)
        
        # Calculate number of patches
        # patch_size is (t, h, w)
        t_patch, h_patch, w_patch = patch_size
        n_t = num_frames // t_patch
        n_h = img_size // h_patch
        n_w = img_size // w_patch
        num_patches = n_t * n_h * n_w
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            REWAVideoBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: (B, C, T, H, W)
        x = self.patch_embed(x)  # (B, N, D)
        
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
