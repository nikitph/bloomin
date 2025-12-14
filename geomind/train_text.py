import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from geomind.models.model import GeoMind

# Hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 128
MAX_ITERS = 500
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
DEVICE = "cpu"
# if torch.backends.mps.is_available():
#     DEVICE = "mps"

print(f"Using device: {DEVICE}")

with open('geomind/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer (Character level)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10) # Eval 10 batches
        for k in range(10):
            X, Y = get_batch(split)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    model = GeoMind(vocab_size=vocab_size, max_seq_len=BLOCK_SIZE, dim=128, depth=4, num_witnesses=32)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    start_time = time.time()
    
    for iter in range(MAX_ITERS):
        # Eval
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Train
        xb, yb = get_batch('train')
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f}s")
    
    # Generate
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print("\nGenerating text:")
    print(decode(generate(model, context, max_new_tokens=200)[0].tolist()))

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    # idx is (B, T)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] # (B, C)
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == "__main__":
    train()
