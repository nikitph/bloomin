import torch
import torch.nn as nn
import torch.optim as optim
import time
from geomind.models.model import GeoMind

# Hyperparameters for MVR
BATCH_SIZE = 64
BLOCK_SIZE = 256 # Longer context
MAX_ITERS = 3000
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
DEVICE = "cpu"
# if torch.backends.mps.is_available():
#     DEVICE = "mps"
# if torch.cuda.is_available():
#     DEVICE = "cuda"

print(f"Using device: {DEVICE}")

# Load data
with open('geomind/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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
        losses = torch.zeros(20) # Eval 20 batches
        for k in range(20):
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

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] 
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def train():
    # Model Config for ~2.5M Params
    # 6 layers, 256 dim -> ~2.5M params
    model = GeoMind(vocab_size=vocab_size, max_seq_len=BLOCK_SIZE, dim=256, depth=6, num_witnesses=32)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"Model parameters: {num_params:.2f}M")

    start_time = time.time()
    
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            dt = time.time() - start_time
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (time: {dt:.2f}s)")

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
    
    # Save model
    torch.save(model.state_dict(), "geomind_mvr.pt")
    
    # Geneate Text
    print("\n--- Generated Samples ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = decode(generate(model, context, max_new_tokens=500)[0].tolist())
    print(generated)

if __name__ == "__main__":
    train()
