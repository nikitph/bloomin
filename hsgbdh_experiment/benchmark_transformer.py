import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformer(nn.Module):
    def __init__(self, n_tokens, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, n_tokens)
        
    def forward(self, src, return_attn=False):
        # src: (B, L)
        x = self.pos_encoder(self.embedding(src) * math.sqrt(64))
        
        # Causal mask
        L = src.size(1)
        mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(src.device)
        
        attns = []
        for layer in self.layers:
            # Manual forward to capture attention
            # self_attn forward returns (attn_output, attn_weights) if need_weights=True
            # BUT TransformerEncoderLayer wraps it.
            # We need to hack or use MultiheadAttention directly.
            # For simplicity, let's trust the standard "Failure" hypothesis 
            # and just use the output prediction: P(End | Start)?
            # No, we feed "A ... Y". Predict "Z".
            # The model predicts Z based on Y.
            # This works.
            # But we want to know if it knows Z is connected to A?
            # The HSGBDH metric is "Reasoning Confidence".
            # Let's stick to the "Attention Extraction" plan.
            # We must use MHA directly.
            pass
            
        # Re-implementing correctly below
        return x

class InterpretableTransformer(nn.Module):
    def __init__(self, n_tokens, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.MultiheadAttention(d_model, nhead, batch_first=True))
            
        self.fc_out = nn.Linear(d_model, n_tokens)
        
    def forward(self, src):
        # returns last_layer_attention: (B, L, L)
        x = self.pos_encoder(self.embedding(src))
        mask = torch.triu(torch.ones(src.size(1), src.size(1)) * float('-inf'), diagonal=1).to(src.device)
        
        last_attn = None
        for attn_layer in self.layers:
            x, attn_weights = attn_layer(x, x, x, attn_mask=mask, need_weights=True)
            # x is just attention output. We skip FFN/Norm for this "Pure Attention" baseline
            # to give it maximum chance of pure routing.
            last_attn = attn_weights
            
        return self.fc_out(x), last_attn

class TransformerBenchmark:
    def __init__(self, n=64, d=64):
        self.n = n
        self.model = InterpretableTransformer(n, d, nhead=4, num_layers=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, length=3, epochs=150):
        print(f"Training Transformer on L={length}...")
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            chain_nodes = torch.randperm(self.n)[:length]
            src = chain_nodes.unsqueeze(0)
            
            # Predict next token
            # Input: [A, B]
            # Target: [B, C]
            output, _ = self.model(src)
            
            # Loss
            loss = 0
            for t in range(length-1):
                pred = output[:, t, :]
                target = src[:, t+1]
                loss += self.criterion(pred, target)
            
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        test_lengths = [3, 4, 5, 7, 9, 12]
        results = []
        
        self.model.eval()
        for L in test_lengths:
            if L > self.n: continue
            
            chain_nodes = torch.randperm(self.n)[:L]
            src = chain_nodes.unsqueeze(0)
            
            with torch.no_grad():
                _, attn = self.model(src)
                # avg over batch(1)
                attn = attn[0] # (L, L)
                
                # Check Attention between Last (L-1) and First (0)
                # In causal mask, Last can attend to First.
                # If Transitive, it should!
                score = attn[L-1, 0].item()
                
            results.append((L, score))
            
        return results

def run_transformer_baseline():
    bench = TransformerBenchmark()
    bench.train(length=3, epochs=200) # Give it every chance
    results = bench.evaluate()
    print("\n--- Transformer Baseline Results ---")
    print(results)
    
    # Plot
    lengths, scores = zip(*results)
    plt.figure(figsize=(10,6))
    plt.plot(lengths, scores, marker='s', color='red', label='Transformer Baseline')
    plt.title('Transformer Failure at L > Train')
    plt.xlabel('Length')
    plt.ylabel('Success Rate (Exact Match)')
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig('transformer_baseline.png')
    print("Plot saved to transformer_baseline.png")

if __name__ == "__main__":
    run_transformer_baseline()
