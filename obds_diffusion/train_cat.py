import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from obds_diffusion.production_model import OBDSDiffusionProduction

def train_cat_single_shot():
    print("Initializing Shadow Cat Experiment...")
    device = torch.device('cpu') # CPU is sufficient for single image
    
    # 1. Load Target Cat
    image_path = "/Users/truckx/.gemini/antigravity/brain/6b93fd07-54f8-4597-9a26-5af653a312a5/shadow_cat_target_1767121058005.png"
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(), # [0, 1]
    ])
    
    try:
        img_pil = Image.open(image_path)
        x_target = transform(img_pil).to(device) # (1, 28, 28)
        x_target = (x_target - 0.5) * 2.0 # [-1, 1]
        x_target = x_target.view(1, 784)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Initialize Model
    # KEY INSIGHT: For a single image x*, the score is roughly -(x - x*) / sigma^2.
    # This is LINEAR in x. So degree=1 should suffice! 
    # Using degree=3 to allow for some curvature but avoiding Runge's oscillation (degree 8).
    model = OBDSDiffusionProduction(data_dim=784, manifold_dim=15, max_degree=3).to(device)
    
    # Disable RD and Riem for pure polynomial test
    with torch.no_grad():
        model.layer_weights.data = torch.tensor([10.0, -100.0, -100.0]) # Force Poly only
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training on Shadow Cat (Stability Mode: Degree 3)...")
    model.train()
    
    losses = []
    
    # Train
    for step in range(1000):
        optimizer.zero_grad()
        
        # Sample random t - bias towards smaller t for fine details? 
        # No, standard uniform is fine, but let's avoid t=1000 pure noise
        t = torch.randint(0, 1000, (1,), device=device)
        
        alpha_t = model.alphas_cumprod[t].sqrt().view(-1, 1)
        sigma_t = (1 - alpha_t**2).sqrt()
        
        noise = torch.randn_like(x_target)
        x_t = alpha_t * x_target + sigma_t * noise
        
        # Predict score
        score_pred = model(x_t, t)
        score_true = -noise / sigma_t
        
        loss = F.mse_loss(score_pred, score_true)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 200 == 0:
            print(f"Step {step}: Loss {loss.item():.6f} | ScoreMean {score_pred.mean().item():.4f} | Std {score_pred.std().item():.4f}")
            
            # Debug: Visualize what the model thinks x0 is at this step
            with torch.no_grad():
                # pred_x0 = (x_t - sigma*score) / alpha
                pred_x0 = (x_t - sigma_t * score_pred) / alpha_t
                pred_x0 = pred_x0.clamp(-1, 1)
                
                # Save debug image
                plt.imsave(f'results/debug_step_{step}.png', 
                          pred_x0.view(28, 28).cpu().numpy(), cmap='gray')

    # 3. Final Sampling (Robust Deterministic)
    print("Sampling final cat (Deterministic with Clamping)...")
    model.eval()
    
    # Custom Sampling Loop for Stability
    num_steps = 50
    # Start closer to signal? No, start from pure noise
    x = torch.randn(1, 784, device=device)
    
    step_indices = torch.linspace(999, 0, num_steps).long().to(device)
    
    with torch.no_grad():
        for idx in step_indices:
            t = torch.full((1,), idx, device=device, dtype=torch.long)
            
            # Predict
            score = model(x, t)
            
            alpha_t = model.alphas_cumprod[idx]
            prev_idx = max(0, idx - (1000 // num_steps))
            alpha_prev = model.alphas_cumprod[prev_idx] if idx > 0 else torch.tensor(1.0).to(device)
            
            # DDIM formulation (eta=0)
            sigma_t = (1 - alpha_t).sqrt()
            pred_x0 = (x - sigma_t * score) / alpha_t.sqrt()
            
            # KEY FIX: Dynamic Stability
            # 1. Clamp prediction to keep it in valid image range
            pred_x0 = pred_x0.clamp(-1.2, 1.2) 
            
            # 2. Reconstruct direction
            sigma_prev = (1 - alpha_prev).sqrt()
            dir_xt = sigma_prev * score # Re-use score as direction
            
            # Recalculate x_{t-1}
            # DDIM: x_{t-1} = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev - sigma^2) * noise
            # Deterministic: sigma=0
            x = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * score
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1); plt.title("Target"); plt.imshow(x_target.view(28, 28).cpu().numpy(), cmap='gray')
    plt.subplot(1, 4, 2); plt.title("Step 200 Recon"); plt.imshow(plt.imread('results/debug_step_200.png'))
    plt.subplot(1, 4, 3); plt.title("Final Dream (Fixed)"); plt.imshow(x.view(28, 28).detach().cpu().numpy(), cmap='gray')
    plt.subplot(1, 4, 4); plt.title("Loss"); plt.plot(losses)
    plt.tight_layout()
    plt.savefig('results/shadow_cat_result.png')
    plt.close()
    
    print("Cat generated! Saved to results/shadow_cat_result.png")

if __name__ == "__main__":
    train_cat_single_shot()
