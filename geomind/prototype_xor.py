import torch
import torch.nn as nn
import torch.optim as optim
from geomind.core.geometry import HyperbolicSpace
from geomind.attention.witness_attn import WitnessAttention

def check_gradients(model):
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.norm().item())
    return grads

class ToyGeoModel(nn.Module):
    def __init__(self, vocab_size, dim, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim) # Start with Euclidean for simplicity? Or use Hyperbolic?
        # User proposed Hyperbolic Embedding Layer:
        # x = embedding(tokens)
        # x = exp_map(x)
        self.hyp = HyperbolicSpace()
        self.witness_attn = WitnessAttention(dim, num_witnesses=16) # Small witnesses for toy
        self.proj = nn.Linear(dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2) # Binary classification
        
    def forward(self, x):
        # 1. Embed
        e = self.embed(x)
        
        # 2. Map to Hyperbolic (just for testing the class, though attention assumes vector input currently)
        # Note: WitnessAttention currently takes "vectors" and sparsifies them. 
        # If we feed hyperbolic points, they are just vectors in the ball.
        h = self.hyp.exp_map0(e) # Map from tangent at 0
        
        # 3. Attention
        # Self-attention
        out, weights = self.witness_attn(h, h, h)
        
        # 4. Project and Classify
        # Currently output of attention is weighted sum of V (h).
        # We might want to map back to tangent space?
        # Let's just treat it as a vector for now.
        out = out.mean(dim=1) # Pooling
        out = self.proj(out)
        logits = self.head(out)
        return logits

def run_toy_experiment():
    print("Running Toy XOR Experiment...")
    vocab_size = 100
    dim = 32
    batch_size = 8
    seq_len = 10
    
    model = ToyGeoModel(vocab_size, dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Fake data
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,))
    
    print("Initial Gradient Check:")
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, labels)
    loss.backward()
    
    grads = check_gradients(model)
    print("Gradient norms: {}".format(grads))
    
    has_grads = any(g > 0 for g in grads)
    if not has_grads:
        print("WARNING: No gradients flowing! WitnessAttention might be non-differentiable.")
    else:
        print("SUCCESS: Gradients are flowing.")
        
    print("\nTraining Loop (5 steps):")
    for i in range(5):
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Step {i}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    run_toy_experiment()
