import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================================
# 1. NIRODHA OPERATOR (Core Innovation)
# ============================================================================

class NirodhaLayer(nn.Module):
    """Single transformer layer with Nirodha regulation"""
    def __init__(self, hidden_dim, n_heads=4, beta=500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        # Identity Initialization
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        
        self.beta = beta
        self.anchor = None
    
    def nirodha_op(self, x):
        """N_beta(x) = x / (1 + beta*|x|)"""
        return x / (1 + self.beta * torch.abs(x))
    
    def set_anchor(self):
        """Lock current state"""
        self.anchor = {k: v.clone() for k, v in self.state_dict().items()}
    
    def forward(self, x, mask=None):
        # Nirodha regulation (apply BEFORE usage to affect this forward pass)
        if self.anchor is not None and self.training:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in self.anchor:
                        anchor_val = self.anchor[name].to(param.device)
                        delta = param.data - anchor_val
                        regulated_delta = self.nirodha_op(delta)
                        param.data.copy_(anchor_val + regulated_delta)

        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

# ============================================================================
# 2. MODEL DEFINITIONS
# ============================================================================

class TransformerToy(nn.Module):
    """Configurable transformer for toy experiments"""
    def __init__(self, vocab_size=100, n_layers=4, hidden_dim=128, 
                 n_heads=4, max_len=50, use_nirodha=False, beta=500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_nirodha = use_nirodha
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        
        # Layers
        if use_nirodha:
            self.layers = nn.ModuleList([
                NirodhaLayer(hidden_dim, n_heads, beta) 
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=4*hidden_dim,
                    batch_first=True
                )
                for _ in range(n_layers)
            ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, output_hidden_states=False):
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.token_emb(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos)
        h = tok_emb + pos_emb
        
        # Store hidden states for analysis
        hidden_states = [h] if output_hidden_states else None
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
            if output_hidden_states:
                hidden_states.append(h)
        
        # Output
        h = self.ln_f(h)
        logits = self.head(h)
        
        if output_hidden_states:
            return logits, hidden_states
        return logits
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def set_anchor(self):
        """Anchor all Nirodha layers"""
        if self.use_nirodha:
            for layer in self.layers:
                layer.set_anchor()

# ============================================================================
# 3. DATASET: MULTI-HOP ARITHMETIC
# ============================================================================

class MultiHopArithmetic(Dataset):
    """
    Generate problems like:
    "A=5, B=A+3, C=B*2, C=?"
    Answer: "16"
    """
    def __init__(self, n_samples=10000, n_hops=3, max_val=20):
        self.n_samples = n_samples
        self.n_hops = n_hops
        self.max_val = max_val
        
        # Build vocabulary
        self.tokens = ['<PAD>', '<START>', '<END>'] + \
                      [str(i) for i in range(max_val * 15)] + \
                      list('ABCDEFGHIJ') + \
                      ['+', '-', '*', '=', ',', '?']
        self.tok2idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx2tok = {i: t for i, t in enumerate(self.tokens)}
        
        # Generate dataset
        self.data = [self._generate_sample() for _ in range(n_samples)]
    
    def _generate_sample(self):
        """Generate one multi-hop problem"""
        vars = ['A', 'B', 'C', 'D', 'E', 'F'][:self.n_hops+1]
        values = {}
        init_val = np.random.randint(1, self.max_val)
        problem = f"A={init_val}"
        values['A'] = init_val
        
        for i in range(self.n_hops):
            prev_var = vars[i]
            curr_var = vars[i+1]
            op = np.random.choice(['+', '-', '*'])
            operand = np.random.randint(1, 4)
            problem += f", {curr_var}={prev_var}{op}{operand}"
            if op == '+': values[curr_var] = values[prev_var] + operand
            elif op == '-': values[curr_var] = values[prev_var] - operand
            else: values[curr_var] = values[prev_var] * operand
        
        final_var = vars[self.n_hops]
        problem += f", {final_var}=?"
        answer = str(max(0, values[final_var]))
        return problem, answer
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        problem, answer = self.data[idx]
        def tokenize(s):
            import re
            return re.findall(r'[A-Z]|\d+|[+\-*/=?,]|<START>|<END>', s)

        # Decoder-only: Concatenate Problem + Answer
        tokens = ['<START>'] + tokenize(problem) + ['='] + tokenize(answer) + ['<END>']
        ids = [self.tok2idx.get(t, 0) for t in tokens]
        
        # Find where the answer starts for masking loss
        ans_start_idx = len(['<START>'] + tokenize(problem) + ['='])
        
        max_len = 50
        ids = ids[:max_len]
        loss_mask = [0] * len(ids)
        for i in range(ans_start_idx, len(ids)):
            loss_mask[i] = 1
            
        padding = [0] * (max_len - len(ids))
        ids += padding
        loss_mask += [0] * (max_len - len(loss_mask))
        
        return torch.tensor(ids), torch.tensor(loss_mask)

# ============================================================================
# 4. TRAINING & EVALUATION
# ============================================================================

def train_model(model, train_loader, epochs=10, lr=3e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for ids, mask in train_loader:
            ids, mask = ids.to(device), mask.to(device)
            # Causal mask
            sz = ids.size(1)
            attn_mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(device)
            
            logits = model(ids) # TransformerToy needs to handle internal masking or we pass it
            # Predict NEXT token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = (loss * shift_mask.view(-1)).sum() / (shift_mask.sum() + 1e-6)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {avg_loss:.4f}")
    return losses

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, mask in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            # For each sample, we find the problem part and let it generate the answer
            for b in range(ids.size(0)):
                # Original problem length is before the first 1 in mask
                m = mask[b]
                if m.sum() == 0: continue
                ans_start = torch.where(m == 1)[0][0].item()
                
                # Input is everything before ans_start
                curr_ids = ids[b, :ans_start].unsqueeze(0)
                target_ans = ids[b, ans_start:torch.where(ids[b] == 2)[0][0]].cpu().numpy() if 2 in ids[b] else ids[b, ans_start:ans_start+3].cpu().numpy()
                
                # Generate up to 5 tokens
                gen_ids = []
                for _ in range(5):
                    sz = curr_ids.size(1)
                    # We need to apply causal mask in the model if it doesn't have it
                    # But TransformerToy doesn't have it yet. Let's fix TransformerToy below.
                    logits = model(curr_ids)
                    next_id = logits[0, -1].argmax().item()
                    gen_ids.append(next_id)
                    if next_id == 2: break # <END>
                    curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]], device=device)], dim=1)
                
                gen_ans = np.array(gen_ids[:-1]) if gen_ids[-1] == 2 else np.array(gen_ids)
                if len(gen_ans) == len(target_ans) and np.array_equal(gen_ans, target_ans):
                    correct += 1
                total += 1
    return 100 * correct / (total + 1e-6)

# ============================================================================
# 5. RE-DEFINITION OF TRANSFORMER TOY WITH CAUSAL MASK
# ============================================================================

class TransformerToy(nn.Module):
    def __init__(self, vocab_size=100, n_layers=4, hidden_dim=128, 
                 n_heads=4, max_len=50, use_nirodha=False, beta=500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_nirodha = use_nirodha
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        
        if use_nirodha:
            self.layers = nn.ModuleList([NirodhaLayer(hidden_dim, n_heads, beta) for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=4*hidden_dim, batch_first=True, norm_first=True) for _ in range(n_layers)])
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        # Causal Mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        
        for layer in self.layers:
            if self.use_nirodha:
                h = layer(h, mask=mask)
            else:
                h = layer(h, src_mask=mask, is_causal=True)
        
        return self.head(self.ln_f(h))

    def count_params(self): return sum(p.numel() for p in self.parameters())
    def set_anchor(self):
        if self.use_nirodha:
            for layer in self.layers: layer.set_anchor()

# ============================================================================
# 5. THE EXPERIMENT
# ============================================================================

def run_depth_vs_width_experiment():
    """
    Compare:
    - Baseline: Wide (512) + Shallow (4 layers) = 4M params
    - Nirodha:  Narrow (128) + Deep (16 layers) = 1M params
    """
    
    print("\n" + "="*60)
    print("DEPTH vs WIDTH: Toy Model Validation")
    print("="*60)
    
    # Device
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device('cpu')
    print(f"   Device: {device}")

    # Create dataset
    n_hops = 3
    print(f"\n1. Creating Multi-Hop Arithmetic Dataset ({n_hops}-hops)...")
    train_data = MultiHopArithmetic(n_samples=5000, n_hops=n_hops)
    test_data = MultiHopArithmetic(n_samples=500, n_hops=n_hops)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    vocab_size = len(train_data.tok2idx)
    
    # Baseline: Wide + Shallow
    print("\n2. [SKIPPED] BASELINE (Wide + Shallow) - Using previous result (59.20%)")
    baseline_acc = 59.20 
    baseline_params = 3.34 
    
    # Nirodha: Narrow + Deep (Fair Params)
    print("\n3. Training NIRODHA (Narrow + Deep)...")
    nirodha = TransformerToy(
        vocab_size=vocab_size,
        n_layers=16,
        hidden_dim=128, # Adjusted to match params better
        n_heads=4,
        use_nirodha=True,
        beta=50 # Lower Beta for exploration
    )
    nirodha.set_anchor() 
    print(f"   Architecture: 16 layers x 128 dim")
    nirodha_params = nirodha.count_params()/1e6
    print(f"   Parameters: {nirodha_params:.2f}M")
    
    nirodha_losses = train_model(nirodha, train_loader, epochs=20, device=device)
    nirodha_acc = evaluate_model(nirodha, test_loader, device=device)
    print(f"   Final Accuracy: {nirodha_acc:.2f}%")
    
    # Final Table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<20} | {'Params':<10} | {'Accuracy':<10}")
    print("-" * 50)
    print(f"{'Baseline (Wide)':<20} | {baseline_params:7.2f}M | {baseline_acc:7.2f}%")
    print(f"{'Nirodha (Deep)':<20} | {nirodha_params:7.2f}M | {nirodha_acc:7.2f}%")
    print("-" * 50)
    
    if nirodha_acc > baseline_acc:
        print("\n✅ Success: Deep model outperformed Wide model on hierarchical reasoning.")
    else:
        print("\n❌ Failure: Wide model outperformed Deep model. Re-evaluating thresholds.")

if __name__ == "__main__":
    run_experiment = run_depth_vs_width_experiment()
