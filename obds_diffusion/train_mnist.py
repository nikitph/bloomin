import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from .model import OBDSDiffusion

def train_obds(epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = OBDSDiffusion(data_dim=784, max_degree=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x_0, _) in enumerate(train_loader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, model.n_timesteps, (batch_size,), device=device)
            
            # Add noise (forward diffusion)
            noise = torch.randn_like(x_0)
            alpha_cumprod = model.alphas_cumprod[t].view(-1, 1)
            x_t = torch.sqrt(alpha_cumprod) * x_0 + torch.sqrt(1 - alpha_cumprod) * noise
            
            # Predict score (Shadow Theory approach predicts the score directly)
            # The true score for Gaussian noise is -(x_t - sqrt(alpha)*x_0) / (1-alpha)
            score_pred = model(x_t, t)
            score_true = -(x_t - torch.sqrt(alpha_cumprod) * x_0) / (1 - alpha_cumprod)
            
            loss = nn.MSELoss()(score_pred, score_true)
            
            # L1 Regularization on polynomial coefficients for sparsity
            l1_lambda = 0.01
            l1_norm = sum(p.norm(1) for p in model.poly_score.parameters())
            loss += l1_lambda * l1_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        
        # Periodic sampling to check quality
        if epoch % 1 == 0:
            print("Sampling...")
            samples = model.sample(batch_size=4, num_steps=20)
            
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/obds_diffusion_mnist.pt')
    print("Model saved to checkpoints/obds_diffusion_mnist.pt")
            
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_obds(epochs=2, device=device)
