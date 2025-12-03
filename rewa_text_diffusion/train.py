import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import WitnessEncoder, WitnessDecoder, ReverseDiffusionModel
from diffusion import Diffusion
from utils import SimpleTokenizer, TextDataset
import os

def train():
    # Hyperparameters
    vocab_size = 100 # Approx from SimpleTokenizer
    embed_dim = 64
    hidden_dim = 128
    max_seq_len = 32
    batch_size = 16
    epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy Data
    texts = [
        "hello world",
        "diffusion models are cool",
        "rewa semantic witness",
        "generative ai is the future",
        "text diffusion pipeline",
        "continuous witness space",
        "neural networks learn",
        "deep learning rocks",
    ] * 50 # Duplicate to make a "dataset"
    
    # Setup
    tokenizer = SimpleTokenizer()
    dataset = TextDataset(texts, tokenizer, max_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    encoder = WitnessEncoder(tokenizer.vocab_size, embed_dim, hidden_dim).to(device)
    decoder = WitnessDecoder(tokenizer.vocab_size, embed_dim, hidden_dim, max_seq_len).to(device)
    reverse_model = ReverseDiffusionModel(hidden_dim).to(device)
    diffusion = Diffusion(num_timesteps=100, device=device)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(reverse_model.parameters()), 
        lr=lr
    )
    
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            tokens = batch.to(device)
            
            # 1. Encode to witness z0
            z0 = encoder(tokens) # [batch, hidden_dim]
            
            # 2. Forward Diffusion
            t = torch.randint(0, diffusion.num_timesteps, (z0.size(0),), device=device).long()
            zt, eps = diffusion.q_sample(z0, t)
            
            # 3. Reverse Model Prediction
            t_in = t.float() / diffusion.num_timesteps # Normalize for model
            eps_hat = reverse_model(zt, t_in.view(-1, 1))
            
            # 4. Reconstruction Loss (Auxiliary)
            # We also want the decoder to learn to map z0 back to text
            # In a real pipeline, this might be pre-trained or trained jointly
            logits = decoder(z0)
            recon_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, tokenizer.vocab_size), 
                tokens.view(-1)
            )
            
            # 5. Diffusion Loss
            diff_loss = torch.nn.functional.mse_loss(eps_hat, eps)
            
            loss = diff_loss + recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    # Save models
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'reverse_model': reverse_model.state_dict(),
        'tokenizer': tokenizer
    }, "rewa_model.pt")
    print("Training complete. Model saved to rewa_model.pt")

if __name__ == "__main__":
    train()
