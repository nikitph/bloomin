import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import WitnessEncoder, WitnessDecoder, ReverseDiffusionModel
from diffusion import Diffusion
from utils import BPETokenizer, TextDataset
import os

def train():
    # Hyperparameters
    vocab_size = 1000  # BPE vocab size
    embed_dim = 128
    hidden_dim = 256
    max_seq_len = 32
    batch_size = 16
    epochs = 150
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Expanded training data with more diverse text
    texts = [
        "hello world",
        "diffusion models are cool",
        "rewa semantic witness",
        "generative ai is the future",
        "text diffusion pipeline",
        "continuous witness space",
        "neural networks learn patterns",
        "deep learning rocks",
        "machine learning transforms data",
        "artificial intelligence advances rapidly",
        "natural language processing",
        "transformer architectures are powerful",
        "attention mechanisms improve performance",
        "gradient descent optimizes parameters",
        "backpropagation computes gradients",
        "embeddings capture semantic meaning",
        "tokenization splits text efficiently",
        "byte pair encoding reduces vocabulary",
        "language models generate coherent text",
        "semantic similarity measures relatedness",
    ] * 30  # Duplicate to make a larger dataset
    
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train_from_texts(texts)
    print(f"BPE tokenizer trained. Vocab size: {tokenizer.vocab_size}")
    
    # Setup dataset
    dataset = TextDataset(texts, tokenizer, max_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    encoder = WitnessEncoder(vocab_size, embed_dim, hidden_dim).to(device)
    decoder = WitnessDecoder(vocab_size, embed_dim, hidden_dim, max_seq_len).to(device)
    reverse_model = ReverseDiffusionModel(hidden_dim).to(device)
    diffusion = Diffusion(num_timesteps=100, device=device)
    
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(reverse_model.parameters()), 
        lr=lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Starting training with BPE tokenization...")
    for epoch in range(epochs):
        total_loss = 0
        total_diff_loss = 0
        total_recon_loss = 0
        
        for batch in dataloader:
            tokens = batch.to(device)
            
            # 1. Encode to witness z0
            z0 = encoder(tokens)  # [batch, hidden_dim]
            
            # 2. Forward Diffusion
            t = torch.randint(0, diffusion.num_timesteps, (z0.size(0),), device=device).long()
            zt, eps = diffusion.q_sample(z0, t)
            
            # 3. Reverse Model Prediction
            t_in = t.float() / diffusion.num_timesteps  # Normalize for model
            eps_hat = reverse_model(zt, t_in.view(-1, 1))
            
            # 4. Reconstruction Loss
            logits = decoder(z0)
            recon_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), 
                tokens.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # 5. Diffusion Loss
            diff_loss = torch.nn.functional.mse_loss(eps_hat, eps)
            
            # Combined loss with weighting
            loss = diff_loss + 0.5 * recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()) + list(reverse_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_recon_loss += recon_loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_diff = total_diff_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} (Diff: {avg_diff:.4f}, Recon: {avg_recon:.4f})")
            
    # Save models
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'reverse_model': reverse_model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'max_seq_len': max_seq_len,
    }, "rewa_model_bpe.pt")
    
    # Save tokenizer separately
    tokenizer.save("bpe_tokenizer.json")
    
    print("Training complete. Model saved to rewa_model_bpe.pt")
    print("Tokenizer saved to bpe_tokenizer.json")

if __name__ == "__main__":
    train()
