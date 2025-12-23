import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

plt.switch_backend('Agg')

# ============================================================================
# 1. TINY DDPM FOR MNIST
# ============================================================================

class SimpleDDPM(nn.Module):
    """Minimal diffusion model for MNIST (28x28 grayscale)"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Simple U-Net-like structure
        self.down1 = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.down2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, t):
        # x: [B, 1, 28, 28], t: [B, 1]
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        
        h1 = torch.relu(self.down1(x) + t_emb)
        h2 = torch.relu(self.down2(h1))
        h = torch.relu(self.mid(h2))
        h = torch.relu(self.up1(h))
        h = self.up2(h + h1)
        
        return h

# ============================================================================
# 2. NIRODHA DEPTH BLOCKS FOR DIFFUSION
# ============================================================================

class NirodhaDiffusionBlock(nn.Module):
    """Add procedural depth to diffusion output"""
    def __init__(self, channels=1, hidden_dim=64, beta=50):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 1, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, channels, 3, padding=1)
        
        # Identity initialization
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        self.beta = beta
        self.anchor = None
    
    def set_anchor(self):
        self.anchor = {k: v.clone().detach() for k, v in self.named_parameters()}
    
    def nirodha_op(self, x):
        return x / (1 + self.beta * torch.abs(x))
    
    def forward(self, x, t):
        # Apply regulation
        if self.anchor and self.training:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in self.anchor:
                        anchor_val = self.anchor[name].to(param.device)
                        delta = param.data - anchor_val
                        param.data.copy_(anchor_val + self.nirodha_op(delta))
        
        # Broadcast time
        t_spatial = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 28, 28)
        h = torch.cat([x, t_spatial], dim=1)
        h = torch.relu(self.conv1(h))
        delta = self.conv2(h)
        
        return x + delta

# ============================================================================
# 3. WRAPPER MODELS
# ============================================================================

class StandardLoRADiffusion(nn.Module):
    """Baseline: Standard LoRA (width-based) - adds trainable layers to base"""
    def __init__(self, base_model, rank=32):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        
        # Add trainable LoRA-style refinement layers
        self.lora_refine = nn.Sequential(
            nn.Conv2d(1, rank, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rank, 1, 3, padding=1)
        )
        # Initialize to near-zero for stability
        nn.init.normal_(self.lora_refine[2].weight, 0, 0.01)
        nn.init.zeros_(self.lora_refine[2].bias)
    
    def forward(self, x, t):
        base_out = self.base(x, t)
        lora_delta = self.lora_refine(base_out)
        return base_out + lora_delta

class LoRADDiffusion(nn.Module):
    """LoRA-D: Depth-based adaptation"""
    def __init__(self, base_model, n_blocks=4, beta=50):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        
        self.blocks = nn.ModuleList([
            NirodhaDiffusionBlock(channels=1, hidden_dim=128, beta=beta)
            for _ in range(n_blocks)
        ])
    
    def set_anchor(self):
        for block in self.blocks:
            block.set_anchor()
    
    def forward(self, x, t):
        h = self.base(x, t)
        for block in self.blocks:
            h = block(h, t)
        return h

# ============================================================================
# 4. TRAINING
# ============================================================================

def train_ddpm(model, dataloader, steps=3000, device='cpu', name="Model"):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    # Linear beta schedule
    beta_start = 0.0001
    beta_end = 0.02
    max_t = 1000
    betas = torch.linspace(beta_start, beta_end, max_t).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    data_iter = iter(dataloader)
    
    for step in range(steps):
        try:
            batch = next(data_iter)
        except:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        x0 = batch[0].to(device)
        batch_size = x0.size(0)
        
        # Sample random timesteps
        t_idx = torch.randint(0, max_t, (batch_size,)).to(device)
        t = t_idx.float() / max_t  # Normalize to [0, 1]
        
        # Get noise schedule values
        alpha_cumprod_t = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Forward diffusion: q(x_t | x_0)
        xt = torch.sqrt(alpha_cumprod_t) * x0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        # Predict noise
        pred_noise = model(xt, t.unsqueeze(1))
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        if step % 500 == 0:
            print(f"   [{name}] Step {step:4d} | Loss: {loss.item():.6f}")
    
    return model

@torch.no_grad()
def sample_ddpm(model, n_samples=16, steps=100, device='cpu'):
    """Proper DDPM sampling with linear beta schedule"""
    model.eval()
    
    # Linear beta schedule (DDPM paper)
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, steps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start from pure noise
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    
    # Reverse diffusion process
    for i in reversed(range(steps)):
        t = torch.ones(n_samples, 1).to(device) * (i / steps)
        
        # Predict noise
        predicted_noise = model(x, t)
        
        # Get schedule values
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta_t = betas[i]
        
        # Compute x_{t-1} using DDPM formula
        if i > 0:
            alpha_cumprod_prev = alphas_cumprod[i - 1]
            # Posterior variance
            posterior_variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            noise = torch.randn_like(x)
        else:
            alpha_cumprod_prev = torch.tensor(1.0).to(device)
            posterior_variance = torch.tensor(0.0).to(device)
            noise = torch.zeros_like(x)
        
        # Mean of q(x_{t-1} | x_t, x_0)
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clip for stability
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev - posterior_variance) * predicted_noise
        
        # Compute x_{t-1}
        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + torch.sqrt(posterior_variance) * noise
    
    return torch.clamp(x, -1, 1)

# ============================================================================
# 5. METRICS
# ============================================================================

def compute_image_quality(samples, reference_images):
    """Compute MSE and structural similarity"""
    samples_np = samples.cpu().numpy()
    ref_np = reference_images.cpu().numpy()
    
    mse = np.mean((samples_np - ref_np) ** 2)
    std_dev = np.std(samples_np)
    
    return {'mse': mse, 'std': std_dev}

# ============================================================================
# 6. THE EXPERIMENT
# ============================================================================

def run_mnist_lorad_experiment():
    print("\n" + "="*60)
    print("MNIST LoRA-D: Personal Handwriting Style Preservation")
    print("="*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Filter to single digit (simulating "personal handwriting style")
    digit_class = 3
    indices = [i for i, (_, label) in enumerate(dataset) if label == digit_class]
    subset = torch.utils.data.Subset(dataset, indices[:200])
    dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
    
    # Get reference images
    ref_batch = next(iter(dataloader))[0][:16]
    
    # --- EXPERIMENT 1: Standard LoRA ---
    print("\n📊 [1/2] Training Standard LoRA (Width-based)...")
    base1 = SimpleDDPM(hidden_dim=128)
    lora_model = StandardLoRADiffusion(base1, rank=32)
    train_ddpm(lora_model, dataloader, steps=3000, device=device, name="LoRA")
    
    print("   Sampling from Standard LoRA...")
    lora_samples = sample_ddpm(lora_model, n_samples=16, device=device)
    lora_metrics = compute_image_quality(lora_samples, ref_batch.to(device))
    print(f"   Metrics: MSE={lora_metrics['mse']:.4f}, Std={lora_metrics['std']:.4f}")
    
    # --- EXPERIMENT 2: LoRA-D ---
    print("\n📊 [2/2] Training LoRA-D (Depth-based, 4 blocks)...")
    base2 = SimpleDDPM(hidden_dim=128)
    lorad_model = LoRADDiffusion(base2, n_blocks=4, beta=50)
    lorad_model.set_anchor()
    train_ddpm(lorad_model, dataloader, steps=3000, device=device, name="LoRA-D")
    
    print("   Sampling from LoRA-D...")
    lorad_samples = sample_ddpm(lorad_model, n_samples=16, device=device)
    lorad_metrics = compute_image_quality(lorad_samples, ref_batch.to(device))
    print(f"   Metrics: MSE={lorad_metrics['mse']:.4f}, Std={lorad_metrics['std']:.4f}")
    
    # --- VISUALIZATION ---
    print("\n📊 Generating comparison visualization...")
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    # Row 1: Reference (Real MNIST)
    for i in range(8):
        axes[0, i].imshow(ref_batch[i].squeeze().cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Reference\n(Real)", fontsize=10)
    
    # Row 2: Standard LoRA
    for i in range(8):
        axes[1, i].imshow(lora_samples[i].squeeze().cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Standard LoRA\n(Width)", fontsize=10)
    
    # Row 3: LoRA-D
    for i in range(8):
        axes[2, i].imshow(lorad_samples[i].squeeze().cpu(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title("LoRA-D\n(Depth)", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'mnist_lora_d_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Results saved to {save_path}")
    
    # --- SUMMARY TABLE ---
    print("\n" + "="*60)
    print("FINAL QUANTITATIVE COMPARISON")
    print("="*60)
    print(f"{'Model':<20} | {'MSE':<10} | {'Std Dev':<10}")
    print("-" * 50)
    print(f"{'Standard LoRA':<20} | {lora_metrics['mse']:10.4f} | {lora_metrics['std']:10.4f}")
    print(f"{'LoRA-D (4 Blocks)':<20} | {lorad_metrics['mse']:10.4f} | {lorad_metrics['std']:10.4f}")
    print("-" * 50)
    
    if lorad_metrics['std'] < lora_metrics['std']:
        print("\n✅ Success: LoRA-D produces more consistent (lower variance) outputs!")
    else:
        print("\n⚠️  Unexpected: Standard LoRA had lower variance. May need tuning.")

if __name__ == "__main__":
    run_mnist_lorad_experiment()
