import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from transformers import AutoModel, AutoConfig

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class REWAModel(nn.Module):
    def __init__(self, backbone_name, mode='rewa', out_dim=128, pretrained=True):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        
        # --- Backbone ---
        if 'resnet' in backbone_name:
            self.modality = 'vision'
            if backbone_name == 'resnet18':
                base = resnet18(weights='DEFAULT' if pretrained else None)
                feat_dim = 512
            else:
                base = resnet50(weights='DEFAULT' if pretrained else None)
                feat_dim = 2048
            
            # Remove FC layer
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.feat_dim = feat_dim
            
        elif 'bert' in backbone_name:
            self.modality = 'text'
            self.backbone = AutoModel.from_pretrained(backbone_name)
            self.feat_dim = self.backbone.config.hidden_size
            
            # Freeze BERT (optional, usually better to fine-tune but expensive)
            # For this exp, we fine-tune properly.
        
        # --- Heads ---
        # 1. Angular Head (u) - Used by all
        self.proj_u = MLPHead(self.feat_dim, out_dim)
        
        # 2. Radial Head (s) - Used only by REWA/Hybrid
        # Can be scalar or vector? User: "I recommend r scalar for simplicity"
        self.proj_r = nn.Linear(self.feat_dim, 1) 
        # Init radial bias for median r=1
        self.proj_r.bias.data.fill_(0.54) # softplus(0.54) approx 1.0
        
    def forward(self, x):
        if self.modality == 'vision':
            x = self.backbone(x)
            x = torch.flatten(x, 1)
        else: # Text
            # x is input_ids, attention_mask dictionary? No, standard API expects tensor/dict
            outputs = self.backbone(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
            x = outputs.last_hidden_state[:, 0, :] # [CLS] token
            
        # --- Projections ---
        # 1. Angular part (u_raw)
        u_raw = self.proj_u(x)
        
        # 2. Radial part (r)
        s = self.proj_r(x).squeeze(1) # [B]
        r = F.softplus(s) + 1e-3
        
        if self.mode == 'normalized':
            # Baseline A: Standard Cosine SimCLR
            v = F.normalize(u_raw, dim=1)
            return v, v, torch.ones_like(r) # r is dummy 1
            
        elif self.mode == 'unnormalized':
            # Baseline B: Raw vectors
            # Note: For fair comparison, should we use u_raw directly?
            # Or use u * r but without normalizing u?
            # Standard "unnormalized" usually just means regular embedding.
            # Let's use u_raw.
            v = u_raw
            return v, F.normalize(v, dim=1), v.norm(dim=1)
            
        elif self.mode == 'rewa':
            # REWA: v = r * u_hat
            u_hat = F.normalize(u_raw, dim=1)
            v = u_hat * r.unsqueeze(1)
            return v, u_hat, r
            
        elif self.mode == 'hybrid':
            # Hybrid: Normalized + Norm as feature
            u_hat = F.normalize(u_raw, dim=1)
            v = torch.cat([u_hat, r.unsqueeze(1)], dim=1)
            return v, u_hat, r
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.state_dict(torch.load(path))
