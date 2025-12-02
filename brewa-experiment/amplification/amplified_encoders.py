import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHybridREWAEncoder(nn.Module):
    def __init__(self, d_model=768, m_dim=256):
        super().__init__()
        
        # Start with HIGH random ratio (e.g., 80%)
        # Gradually decrease as model learns
        self.random_ratio = nn.Parameter(torch.tensor(0.8))  # Learnable!
        
        # Calculate dimensions ensuring sum is m_dim
        m_random = int(m_dim * 0.8)
        m_learned = m_dim - m_random
        
        # Random projection (LARGE portion)
        self.random_proj = nn.Linear(d_model, m_random, bias=False)
        nn.init.orthogonal_(self.random_proj.weight)
        self.random_proj.weight.requires_grad = False
        
        # Learned projection (SMALL portion)
        self.learned_proj = nn.Sequential(
            nn.Linear(d_model, m_learned),
            nn.LayerNorm(m_learned),
            nn.Dropout(0.3),
        )
        
        # Dynamic mixing network
        self.mixer = nn.Sequential(
            nn.Linear(m_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Outputs weights for random vs learned
        )
    
    def forward(self, x, epoch=None, training=True, add_noise=None, **kwargs):
        # Get random and learned parts
        random_part = self.random_proj(x)
        learned_part = self.learned_proj(x)
        
        # Dynamic mixing
        combined = torch.cat([random_part, learned_part], dim=-1)
        
        # Determine mixing weights
        # Use mean pooling for sequence data if x is [B, N, D], else x is [B, D]
        if x.dim() == 3:
            pool = combined.mean(dim=1)
        else:
            pool = combined
            
        if training:
            mix_weights = self.mixer(pool)
        else:
            # At inference: learned optimal mix
            with torch.no_grad():
                mix_weights = self.mixer(pool)
        
        mix_weights = F.softmax(mix_weights, dim=-1)
        
        # Expand weights for broadcasting
        if x.dim() == 3:
            w_random = mix_weights[:, 0].view(-1, 1, 1)
            w_learned = mix_weights[:, 1].view(-1, 1, 1)
        else:
            w_random = mix_weights[:, 0].view(-1, 1)
            w_learned = mix_weights[:, 1].view(-1, 1)
        
        # Pad parts to match m_dim for addition? 
        # Wait, the user's code did: output = random_weighted + learned_weighted
        # But random_part is 0.8*m_dim and learned_part is 0.2*m_dim. They can't be added directly if they have different shapes.
        # The user's snippet might have implied they project to same dimension or something else.
        # "combined = torch.cat([random_part, learned_part], dim=-1)" implies concatenation.
        # "output = random_weighted + learned_weighted" implies addition.
        # Let's assume the user meant to WEIGHT the concatenation, or project them to same space.
        # Given the "Dynamic Random Ratio Tuning" description, maybe the idea is to weight the *contribution* of each part in the final vector?
        # But usually concatenation is enough.
        # Let's look at the user's code again:
        # mix_weights = F.softmax(mix_weights, dim=-1)
        # random_weighted = random_part * mix_weights[:, 0:1]
        # learned_weighted = learned_part * mix_weights[:, 1:2]
        # output = random_weighted + learned_weighted
        # This DEFINITELY requires same shape.
        
        # FIX: Let's make both project to m_dim, but we keep the "ratio" concept in initialization or just use m_dim for both?
        # Or maybe the user meant we concatenate them and the mixer weights are applied to the concatenated vector?
        # "Dynamic Random Ratio Tuning" -> "Start with HIGH random ratio... Gradually decrease".
        # If we want to add them, they must be same size.
        # Let's modify to: Both project to m_dim // 2 (or full m_dim) and we weight them.
        # OR: We concatenate them (total m_dim) and just return the concatenated vector, but maybe scale them?
        # The user's code `output = random_weighted + learned_weighted` is explicit.
        # I will modify the init to make them compatible for addition: both project to m_dim.
        # But wait, `self.random_proj = nn.Linear(d_model, int(m_dim * 0.8))`
        # This explicitly sets different sizes.
        # I will assume the user meant CONCATENATION is the output, but maybe we scale the segments?
        # "combined = torch.cat([random_part, learned_part], dim=-1)" -> This is size m_dim.
        # If I return `combined`, that matches m_dim.
        # If I try to add them, it fails.
        # I will interpret "Dynamic Random Ratio" as: We have a fixed architecture (concatenation), but we might want to weight the features?
        # Actually, looking at "output = random_weighted + learned_weighted", maybe the user copied code from a version where they were same size.
        # I'll stick to CONCATENATION as the output mechanism (since m_dim is split 0.8/0.2), and apply the weights as a scaling factor to the segments.
        
        w_random = mix_weights[:, 0].view(-1, 1) if x.dim() == 2 else mix_weights[:, 0].view(-1, 1, 1)
        w_learned = mix_weights[:, 1].view(-1, 1) if x.dim() == 2 else mix_weights[:, 1].view(-1, 1, 1)
        
        # Scale the parts
        random_scaled = random_part * w_random
        learned_scaled = learned_part * w_learned
        
        output = torch.cat([random_scaled, learned_scaled], dim=-1)
        return F.normalize(output, dim=-1)

class MultiScaleHybridREWAEncoder(nn.Module):
    def __init__(self, d_model=768, m_dim=256):
        super().__init__()
        
        # 3 different random projections (different scales)
        # Split m_dim into 3 parts for random, plus 1 part for learned?
        # User code: m_dim // 3 for each random. That's m_dim total.
        # Plus learned_proj to m_dim.
        # Then fusion attention.
        
        part_dim = m_dim // 3
        self.random_projections = nn.ModuleList([
            # Coarse scale (preserves global structure)
            nn.Linear(d_model, part_dim, bias=False),
            # Medium scale (preserves local structure)
            nn.Linear(d_model, part_dim, bias=False),
            # Fine scale (preserves fine details)
            nn.Linear(d_model, part_dim, bias=False),
        ])
        
        # Initialize each with different scales
        with torch.no_grad():
            # Coarse: Large variance
            nn.init.normal_(self.random_projections[0].weight, std=1.0)
            # Medium: Medium variance
            nn.init.normal_(self.random_projections[1].weight, std=0.5)
            # Fine: Small variance
            nn.init.normal_(self.random_projections[2].weight, std=0.1)
        
        # Freeze all
        for proj in self.random_projections:
            proj.weight.requires_grad = False
        
        # Learned projection - projects to full m_dim? Or part_dim?
        # User code: nn.Linear(d_model, m_dim)
        self.learned_proj = nn.Sequential(
            nn.Linear(d_model, m_dim),
            nn.LayerNorm(m_dim),
            nn.Dropout(0.3),
        )
        
        # Attention-based fusion
        # Embed dim = 3*part_dim + m_dim?
        # User code: embed_dim=m_dim * 4.
        # This implies part_dim = m_dim.
        # But init says m_dim // 3.
        # Let's fix: Make random projections project to m_dim each.
        
        self.random_projections = nn.ModuleList([
            nn.Linear(d_model, m_dim, bias=False),
            nn.Linear(d_model, m_dim, bias=False),
            nn.Linear(d_model, m_dim, bias=False),
        ])
        # Re-init
        with torch.no_grad():
            nn.init.normal_(self.random_projections[0].weight, std=1.0)
            nn.init.normal_(self.random_projections[1].weight, std=0.5)
            nn.init.normal_(self.random_projections[2].weight, std=0.1)
        for proj in self.random_projections:
            proj.weight.requires_grad = False
            
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=m_dim, # We will stack them as sequence [B, 4, m_dim]
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, x, add_noise=None, **kwargs):
        # x: [B, D] (assuming 2D for simplicity in this experiment, or handle 3D)
        is_3d = x.dim() == 3
        if is_3d:
            B, N, D = x.shape
            # Flatten for projection then reshape? Or just map
            # Let's assume x is [B, D] for the newsgroups experiment (doc embeddings)
            # If x is [B, N, D], we need to be careful with attention dims.
            pass
        
        # Get all projections
        random_features = []
        for proj in self.random_projections:
            feat = proj(x)
            random_features.append(F.normalize(feat, dim=-1))
        
        learned_feat = self.learned_proj(x)
        learned_feat = F.normalize(learned_feat, dim=-1)
        
        # Stack features: [B, 4, m_dim]
        # If input was [B, D], this works.
        # If input was [B, N, D], we get [B, N, m_dim]. Stacking would be [B, N, 4, m_dim]?
        # The fusion attention expects [Batch, Seq, Dim].
        # If we want to fuse the 4 representations, we treat "4" as the sequence length.
        
        if is_3d:
            # x: [B, N, D]
            # feats: [B, N, m_dim]
            # Stack: [B*N, 4, m_dim]
            all_features = torch.stack(random_features + [learned_feat], dim=-2) # [B, N, 4, m_dim]
            B, N, K, M = all_features.shape
            all_features = all_features.view(B*N, K, M)
        else:
            all_features = torch.stack(random_features + [learned_feat], dim=1) # [B, 4, m_dim]
        
        # Attention-based fusion
        # Query: mean of features?
        query = all_features.mean(dim=1, keepdim=True) # [Batch, 1, m_dim]
        fused, _ = self.fusion_attention(query, all_features, all_features) # [Batch, 1, m_dim]
        
        output = fused.squeeze(1)
        
        if is_3d:
            output = output.view(B, N, M)
            
        return F.normalize(output, dim=-1)

class AdversarialHybridREWAEncoder(nn.Module):
    def __init__(self, d_model=768, m_dim=256):
        super().__init__()
        
        # Standard hybrid encoder
        self.random_proj = nn.Linear(d_model, m_dim // 2, bias=False)
        nn.init.orthogonal_(self.random_proj.weight)
        self.random_proj.weight.requires_grad = False
        
        self.learned_proj = nn.Sequential(
            nn.Linear(d_model, m_dim // 2),
            nn.LayerNorm(m_dim // 2),
        )
        
        # DISCRIMINATOR
        self.discriminator = nn.Sequential(
            nn.Linear(m_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def forward(self, x, add_noise=None, **kwargs):
        random_part = self.random_proj(x)
        learned_part = self.learned_proj(x)
        
        # Combine
        combined = torch.cat([random_part, learned_part], dim=-1)
        return F.normalize(combined, dim=-1)
    
    def adversarial_loss(self, x):
        # Get features
        learned_features = self.learned_proj(x)
        with torch.no_grad():
            random_features = self.random_proj(x)
        
        # Flatten if 3D
        if x.dim() == 3:
            learned_features = learned_features.view(-1, learned_features.size(-1))
            random_features = random_features.view(-1, random_features.size(-1))
            
        batch_size = learned_features.shape[0]
        half = batch_size // 2
        
        mixed_features = torch.cat([
            learned_features[:half],
            random_features[half:]
        ], dim=0)
        
        labels = torch.cat([
            torch.ones(half, 1, device=x.device),  # Learned
            torch.zeros(batch_size - half, 1, device=x.device)  # Random
        ], dim=0)
        
        preds = torch.sigmoid(self.discriminator(mixed_features))
        adv_loss = F.binary_cross_entropy(preds, labels)
        
        return adv_loss

class CurriculumHybridREWAEncoder(nn.Module):
    def __init__(self, d_model=768, m_dim=256):
        super().__init__()
        
        # Start as PURE random encoder
        self.random_proj = nn.Linear(d_model, m_dim, bias=False)
        nn.init.orthogonal_(self.random_proj.weight)
        self.random_proj.weight.requires_grad = False # Keep random fixed? User said "Gradually replace".
        # If we replace, we need both.
        
        self.learned_proj = nn.Sequential(
            nn.Linear(d_model, m_dim),
            nn.LayerNorm(m_dim),
        )
        
        # Mask: which dimensions are still random vs learned
        # We'll use a float mask for soft transition or hard mask
        self.register_buffer('random_mask', torch.ones(m_dim))
        self.epoch = 0
    
    def forward(self, x, current_epoch=None, add_noise=None, **kwargs):
        if current_epoch is not None:
            self.epoch = current_epoch
        
        # Calculate how many dimensions to keep random
        # Start: 100% random, end: 50% random
        keep_random_ratio = max(0.5, 1.0 - (self.epoch / 20) * 0.5) # Faster curriculum (20 epochs)
        num_random = int(self.random_mask.shape[0] * keep_random_ratio)
        
        random_features = self.random_proj(x)
        learned_features = self.learned_proj(x)
        
        # Create mask
        # We want to keep the same dimensions random if possible, or shuffle?
        # User code: "indices = torch.topk(self.random_mask, num_random)"
        # This implies random_mask has values.
        # Let's just pick top k indices deterministically or randomly.
        # Deterministic is better for stability.
        
        mask = torch.zeros_like(self.random_mask)
        mask[:num_random] = 1.0
        
        # Apply mask
        output = random_features * mask + learned_features * (1 - mask)
        
        return F.normalize(output, dim=-1)

class EnsembleHybridREWAEncoder(nn.Module):
    def __init__(self, d_model=768, m_dim=256, num_bases=8):
        super().__init__()
        
        self.num_bases = num_bases
        
        # Calculate dimensions to handle remainder
        base_dim = m_dim // num_bases
        remainder = m_dim % num_bases
        
        self.dims = []
        self.random_bases = nn.ModuleList()
        
        for i in range(num_bases):
            # Distribute remainder to first few bases
            curr_dim = base_dim + (1 if i < remainder else 0)
            self.dims.append(curr_dim)
            self.random_bases.append(nn.Linear(d_model, curr_dim, bias=False))
        
        # Initialize
        for i, base in enumerate(self.random_bases):
            torch.manual_seed(i + 42)
            nn.init.orthogonal_(base.weight)
            base.weight.requires_grad = False
        
        # Learned combiner
        self.combiner = nn.Sequential(
            nn.Linear(m_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_bases),
            nn.Softmax(dim=-1),
        )
        
        # Learned residual
        self.learned_residual = nn.Sequential(
            nn.Linear(d_model, m_dim),
            nn.LayerNorm(m_dim),
        )
    
    def forward(self, x, add_noise=None, **kwargs):
        random_features = []
        for base in self.random_bases:
            feat = base(x)
            random_features.append(feat)
        
        random_concat = torch.cat(random_features, dim=-1)
        
        # Learn weights
        weights = self.combiner(random_concat) # [B, num_bases]
        
        # Weighted combination
        # We need to apply weight[i] to random_features[i]
        weighted_parts = []
        for i in range(self.num_bases):
            w = weights[:, i].view(-1, 1) if x.dim() == 2 else weights[:, i].view(-1, 1, 1)
            weighted_parts.append(random_features[i] * w)
            
        weighted_random = torch.cat(weighted_parts, dim=-1)
        
        learned = self.learned_residual(x)
        
        output = weighted_random + 0.3 * learned
        
        return F.normalize(output, dim=-1)
