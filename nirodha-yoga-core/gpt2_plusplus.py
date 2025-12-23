import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config

class NirodhaTransformerBlock(nn.Module):
    """
    GPT-2 style transformer block with Nirodha regulation.
    Regulation is applied functionally to weights relative to an anchor.
    """
    def __init__(self, config, beta=10000.0):
        super().__init__()
        # Standard GPT-2 Block components
        hidden_size = config.n_embd
        num_heads = config.n_head
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        # Identity Initialization: Ensure blocks start as identity mappings
        # Final linear layers in each sub-block are zeroed.
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        # For MultiheadAttention, we zero the out_proj
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        
        self.beta = beta
        self.anchor = None
        self.is_res = True # Default to residual for depth expansion
        self.last_drift = 0.0

    def set_anchor(self):
        """Snapshots current weights as the stable anchor."""
        self.anchor = {k: v.clone().detach() for k, v in self.named_parameters()}

    def nirodha_op(self, x):
        """N_beta(x) = x / (1 + beta*|x|)"""
        return x / (1.0 + self.beta * torch.abs(x))

    def get_regulated_params(self):
        """Returns parameters regulated by the anchor."""
        regulated = {}
        for name, param in self.named_parameters():
            if self.anchor and name in self.anchor:
                anchor_val = self.anchor[name]
                delta = param - anchor_val
                regulated[name] = anchor_val + self.nirodha_op(delta)
            else:
                regulated[name] = param
        return regulated

    def forward(self, x, attention_mask=None):
        params = self.get_regulated_params()
        
        # 1. Attention (Regulated)
        residual = x
        x = F.layer_norm(x, self.ln_1.normalized_shape, params['ln_1.weight'], params['ln_1.bias'], self.ln_1.eps)
        
        # Multi-head attention functional
        # We need to map our params to the format expected by F.multi_head_attention_forward
        # For simplicity, we'll use the attribute weights directly if we can, 
        # but since we want to be functional, let's extract them.
        
        qkv_w = params['attn.in_proj_weight']
        qkv_b = params['attn.in_proj_bias']
        out_w = params['attn.out_proj.weight']
        out_b = params['attn.out_proj.bias']
        
        attn_output, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=self.attn.embed_dim,
            num_heads=self.attn.num_heads,
            in_proj_weight=qkv_w,
            in_proj_bias=qkv_b,
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=out_w,
            out_proj_bias=out_b,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=attention_mask
        )
        
        x = residual + attn_output
        
        # 2. MLP (Regulated)
        residual = x
        x = F.layer_norm(x, self.ln_2.normalized_shape, params['ln_2.weight'], params['ln_2.bias'], self.ln_2.eps)
        
        x = F.linear(x, params['mlp.0.weight'], params['mlp.0.bias'])
        x = F.gelu(x)
        x = F.linear(x, params['mlp.2.weight'], params['mlp.2.bias'])
        
        if self.is_res:
            output = residual + x
        else:
            output = x
            
        # Log drift: ||h_post - h_pre|| / ||h_pre||
        with torch.no_grad():
            self.last_drift = torch.norm(output - residual) / (torch.norm(residual) + 1e-6)
            
        return output

class GPT2PlusPlus(nn.Module):
    """
    GPT-2 Wrapper with Nirodha Depth Expansion.
    Base model is frozen; only added layers are trainable.
    """
    def __init__(self, model_id="gpt2"):
        super().__init__()
        self.base = GPT2LMHeadModel.from_pretrained(model_id)
        self.config = self.base.config
        
        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False
            
        self.nirodha_blocks = nn.ModuleList()
        self.total_base_layers = len(self.base.transformer.h)
        
    def add_layers(self, n=6, beta=10000.0):
        print(f"📈 Adding {n} Nirodha reasoning layers (beta={beta})")
        for _ in range(n):
            self.nirodha_blocks.append(NirodhaTransformerBlock(self.config, beta=beta))
            
    def set_anchor(self):
        """Anchors all current Nirodha blocks."""
        print("🔒 Anchoring reasoning layers...")
        for block in self.nirodha_blocks:
            block.set_anchor()
            
    def update_beta(self, new_beta):
        """Updates beta for all nirodha blocks."""
        for block in self.nirodha_blocks:
            block.beta = new_beta
            
    def forward(self, input_ids, attention_mask=None, loop_k=1):
        # 1. Base Embeddings
        hidden_states = self.base.transformer.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device)
        position_embeds = self.base.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        
        # 2. Base Transformer Layers (Frozen)
        # Handle attention mask for GPT2 (needs to be 4D etc, but let's keep it simple for PoC)
        for block in self.base.transformer.h:
            hidden_states = block(hidden_states)[0]
            
        # 3. Nirodha Layers (Trainable)
        self.drifts = []
        for _ in range(loop_k):
            for block in self.nirodha_blocks:
                hidden_states = block(hidden_states)
                self.drifts.append(block.last_drift.item())
            
        # 4. Final Output
        hidden_states = self.base.transformer.ln_f(hidden_states)
        logits = self.base.lm_head(hidden_states)
        
        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

if __name__ == "__main__":
    model = GPT2PlusPlus()
    print(f"Base GPT-2 Params: {model.count_parameters()[0]/1e6:.1f}M")
    
    model.add_layers(n=6)
    total, trainable = model.count_parameters()
    print(f"GPT-2++ Total Params: {total/1e6:.1f}M")
    print(f"Trainable Params (Added Depth): {trainable/1e6:.1f}M")
    
    dummy_input = torch.randint(0, 50257, (1, 32))
    output = model(dummy_input)
    print(f"Forward pass success. Output shape: {output.shape}")
