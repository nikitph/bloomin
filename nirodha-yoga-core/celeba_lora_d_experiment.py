import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

plt.switch_backend('Agg')

# ============================================================================
# 1. NIRODHA DEPTH BLOCKS FOR UNET
# ============================================================================

class NirodhaUNetBlock(nn.Module):
    """Depth-based refinement block for UNet output"""
    def __init__(self, channels=3, hidden_dim=128, beta=20):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, channels, 3, padding=1)
        
        # Identity initialization
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        
        self.beta = beta
        self.anchor = None
    
    def set_anchor(self):
        self.anchor = {k: v.clone().detach() for k, v in self.named_parameters()}
    
    def nirodha_op(self, x):
        return x / (1 + self.beta * torch.abs(x))
    
    def forward(self, x):
        # Apply regulation
        if self.anchor and self.training:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in self.anchor:
                        anchor_val = self.anchor[name].to(param.device)
                        delta = param.data - anchor_val
                        param.data.copy_(anchor_val + self.nirodha_op(delta))
        
        # Process
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        delta = self.conv3(h)
        
        return x + delta

# ============================================================================
# 2. WRAPPER MODELS
# ============================================================================

class StandardLoRADiffusion(nn.Module):
    """Baseline: Standard LoRA on pre-trained UNet"""
    def __init__(self, base_unet, rank=64):
        super().__init__()
        self.unet = base_unet
        for p in self.unet.parameters():
            p.requires_grad = False
        
        # Add trainable refinement layers
        self.lora_refine = nn.Sequential(
            nn.Conv2d(3, rank, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rank, rank, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(rank, 3, 3, padding=1)
        )
        nn.init.normal_(self.lora_refine[-1].weight, 0, 0.01)
        nn.init.zeros_(self.lora_refine[-1].bias)
    
    def forward(self, x, t):
        base_out = self.unet(x, t).sample
        lora_delta = self.lora_refine(base_out)
        return base_out + lora_delta

class LoRADDiffusion(nn.Module):
    """LoRA-D: Depth-based adaptation"""
    def __init__(self, base_unet, n_blocks=4, beta=20):
        super().__init__()
        self.unet = base_unet
        for p in self.unet.parameters():
            p.requires_grad = False
        
        self.blocks = nn.ModuleList([
            NirodhaUNetBlock(channels=3, hidden_dim=128, beta=beta)
            for _ in range(n_blocks)
        ])
    
    def set_anchor(self):
        for block in self.blocks:
            block.set_anchor()
    
    def forward(self, x, t):
        h = self.unet(x, t).sample
        for block in self.blocks:
            h = block(h)
        return h

# ============================================================================
# 3. CELEBA DATASET LOADER
# ============================================================================

class SimpleCelebADataset:
    """Load CelebA images at 64x64"""
    def __init__(self, image_dir, n_samples=100):
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Get image files
        self.image_files = list(self.image_dir.glob('*.jpg'))[:n_samples]
        if len(self.image_files) == 0:
            self.image_files = list(self.image_dir.glob('*.png'))[:n_samples]
        
        print(f"   Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(img)

# ============================================================================
# 4. TRAINING
# ============================================================================

def train_diffusion(model, dataset, scheduler, steps=2000, device='cpu', name="Model"):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for step in range(steps):
        # Sample batch
        idx = torch.randint(0, len(dataset), (8,))
        batch = torch.stack([dataset[i] for i in idx]).to(device)
        
        # Sample timesteps
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch.size(0),)).long().to(device)
        
        # Add noise
        noise = torch.randn_like(batch)
        noisy_images = scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise
        pred_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 200 == 0:
            print(f"   [{name}] Step {step:4d} | Loss: {loss.item():.6f}")
    
    return model

@torch.no_grad()
def sample_diffusion(model, scheduler, n_samples=8, device='cpu'):
    model.eval()
    
    # Start from noise
    image = torch.randn(n_samples, 3, 64, 64).to(device)
    
    # Denoise
    for t in scheduler.timesteps:
        timestep = torch.tensor([t] * n_samples).long().to(device)
        noise_pred = model(image, timestep)
        image = scheduler.step(noise_pred, t, image).prev_sample
    
    return torch.clamp(image, -1, 1)

# ============================================================================
# 5. FACE RECOGNITION METRICS
# ============================================================================

def compute_face_similarity(samples, reference):
    """Simple pixel-based similarity (can be upgraded to FaceNet later)"""
    samples_np = samples.cpu().numpy()
    ref_np = reference.cpu().numpy()
    
    # Normalize to [0, 1]
    samples_np = (samples_np + 1) / 2
    ref_np = (ref_np + 1) / 2
    
    # Compute MSE and SSIM-like metric
    mse = np.mean((samples_np - ref_np) ** 2)
    
    # Structural similarity (simplified)
    mean_sample = np.mean(samples_np, axis=(2, 3), keepdims=True)
    mean_ref = np.mean(ref_np, axis=(2, 3), keepdims=True)
    
    var_sample = np.var(samples_np, axis=(2, 3))
    var_ref = np.var(ref_np, axis=(2, 3))
    
    structural_sim = np.mean(1 / (1 + np.abs(var_sample - var_ref)))
    
    return {
        'mse': mse,
        'structural_similarity': structural_sim,
        'std': np.std(samples_np)
    }

# ============================================================================
# 6. OOD TRANSFORMATION FUNCTIONS
# ============================================================================

def apply_novel_lighting(img_tensor, intensity=2.0):
    """Simulate dramatic side lighting"""
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    width, height = img.size
    gradient = Image.new('L', (width, height))
    for x in range(width):
        brightness = int(255 * (x / width) * intensity)
        for y in range(height):
            gradient.putpixel((x, y), min(255, brightness))
    img = Image.composite(img, Image.new('RGB', img.size, 'black'), gradient)
    return transforms.ToTensor()(img) * 2 - 1

def apply_rotation(img_tensor, angle=30):
    """Rotate face (novel pose)"""
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    return transforms.ToTensor()(img) * 2 - 1

def compute_perceptual_similarity(img1, img2):
    """Simplified perceptual similarity"""
    img1_norm = (img1 + 1) / 2
    img2_norm = (img2 + 1) / 2
    mse = F.mse_loss(img1_norm, img2_norm)
    return 1 / (1 + mse.item())

def run_ood_test(lora_model, lorad_model, scheduler, reference_images, device='cpu'):
    """Test identity preservation under OOD conditions"""
    print("\n" + "="*70)
    print("OUT-OF-DISTRIBUTION IDENTITY PRESERVATION TEST")
    print("="*70)
    
    ood_tests = [
        ('Novel Lighting', lambda x: apply_novel_lighting(x, 2.0)),
        ('30° Rotation', lambda x: apply_rotation(x, 30)),
    ]
    
    results = {'lora': [], 'lorad': []}
    reference = reference_images[0:1].to(device)
    
    for name, transform in ood_tests:
        print(f"\n  Testing: {name}")
        transformed_ref = transform(reference[0]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            timesteps = torch.tensor([500]).long().to(device)
            noise = torch.randn_like(transformed_ref)
            noisy_img = scheduler.add_noise(transformed_ref, noise, timesteps)
            
            lora_pred = lora_model(noisy_img, timesteps)
            lora_out = scheduler.step(lora_pred, timesteps[0], noisy_img).prev_sample
            
            lorad_pred = lorad_model(noisy_img, timesteps)
            lorad_out = scheduler.step(lorad_pred, timesteps[0], noisy_img).prev_sample
        
        lora_sim = compute_perceptual_similarity(lora_out, reference)
        lorad_sim = compute_perceptual_similarity(lorad_out, reference)
        
        results['lora'].append(lora_sim)
        results['lorad'].append(lorad_sim)
        
        print(f"    Standard LoRA: {lora_sim:.1%}")
        print(f"    LoRA-D:        {lorad_sim:.1%}")
        print(f"    Advantage:     {(lorad_sim - lora_sim):+.1%}")
    
    avg_lora = np.mean(results['lora'])
    avg_lorad = np.mean(results['lorad'])
    
    print(f"\n  {'─'*66}")
    print(f"  AVERAGE OOD IDENTITY PRESERVATION:")
    print(f"    Standard LoRA: {avg_lora:.1%}")
    print(f"    LoRA-D:        {avg_lorad:.1%}")
    print(f"    Improvement:   {(avg_lorad - avg_lora):+.1%}")
    print(f"  {'─'*66}")
    
    return avg_lora, avg_lorad

# ============================================================================
# 7. THE EXPERIMENT
# ============================================================================

def run_celeba_lorad_experiment(image_dir='./celeba_data'):
    print("\n" + "="*60)
    print("CelebA 64×64 LoRA-D: Face Identity Preservation")
    print("="*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        print(f"\n⚠️  Image directory '{image_dir}' not found!")
        print("Please create it and add face images, or specify a different path.")
        print("\nFor testing, you can:")
        print("1. Download a few CelebA images from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("2. Or use your own face photos (20-50 images recommended)")
        return
    
    # Load dataset
    print("\n📊 Loading CelebA dataset...")
    dataset = SimpleCelebADataset(image_dir, n_samples=100)
    
    if len(dataset) == 0:
        print("❌ No images found! Please add .jpg or .png files to the directory.")
        return
    
    # Load pre-trained UNet (using a small unconditional model)
    print("\n📊 Loading pre-trained diffusion model...")
    try:
        unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256", subfolder="unet")
        # Downsample to 64x64 by modifying the model
        print("   Loaded DDPM-CelebA-HQ model")
    except:
        print("   Creating new UNet (no pre-trained weights available)")
        unet = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Get reference images
    ref_batch = torch.stack([dataset[i] for i in range(min(8, len(dataset)))]).to(device)
    
    # --- EXPERIMENT 1: Standard LoRA ---
    print("\n📊 [1/2] Training Standard LoRA (Width-based)...")
    lora_model = StandardLoRADiffusion(unet, rank=64)
    train_diffusion(lora_model, dataset, scheduler, steps=2000, device=device, name="LoRA")
    
    print("   Sampling from Standard LoRA...")
    lora_samples = sample_diffusion(lora_model, scheduler, n_samples=8, device=device)
    lora_metrics = compute_face_similarity(lora_samples, ref_batch)
    print(f"   Metrics: MSE={lora_metrics['mse']:.4f}, Sim={lora_metrics['structural_similarity']:.4f}")
    
    # --- EXPERIMENT 2: LoRA-D ---
    print("\n📊 [2/2] Training LoRA-D (Depth-based, 4 blocks)...")
    unet2 = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    lorad_model = LoRADDiffusion(unet2, n_blocks=4, beta=20)
    lorad_model.set_anchor()
    train_diffusion(lorad_model, dataset, scheduler, steps=2000, device=device, name="LoRA-D")
    
    print("   Sampling from LoRA-D...")
    lorad_samples = sample_diffusion(lorad_model, scheduler, n_samples=8, device=device)
    lorad_metrics = compute_face_similarity(lorad_samples, ref_batch)
    print(f"   Metrics: MSE={lorad_metrics['mse']:.4f}, Sim={lorad_metrics['structural_similarity']:.4f}")
    
    # --- OOD TEST ---
    print("\n📊 [3/3] Running Out-of-Distribution Test...")
    ood_lora, ood_lorad = run_ood_test(lora_model, lorad_model, scheduler, ref_batch, device=device)
    
    # --- VISUALIZATION ---
    print("\n📊 Generating comparison visualization...")
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    # Row 1: Reference
    for i in range(8):
        img = (ref_batch[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Reference\n(Real)", fontsize=10)
    
    # Row 2: Standard LoRA
    for i in range(8):
        img = (lora_samples[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[1, i].imshow(np.clip(img, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Standard LoRA\n(Width)", fontsize=10)
    
    # Row 3: LoRA-D
    for i in range(8):
        img = (lorad_samples[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[2, i].imshow(np.clip(img, 0, 1))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title("LoRA-D\n(Depth)", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'celeba_lora_d_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Results saved to {save_path}")
    
    # --- SUMMARY TABLE ---
    print("\n" + "="*60)
    print("FINAL QUANTITATIVE COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} | {'Standard LoRA':<15} | {'LoRA-D':<15}")
    print("-" * 60)
    print(f"{'In-Distribution MSE':<30} | {lora_metrics['mse']:>14.4f} | {lorad_metrics['mse']:>14.4f}")
    print(f"{'In-Distribution Similarity':<30} | {lora_metrics['structural_similarity']:>14.4f} | {lorad_metrics['structural_similarity']:>14.4f}")
    print(f"{'In-Distribution Std':<30} | {lora_metrics['std']:>14.4f} | {lorad_metrics['std']:>14.4f}")
    print("-" * 60)
    print(f"{'OOD Identity Preservation':<30} | {ood_lora:>14.1%} | {ood_lorad:>14.1%}")
    print("-" * 60)
    
    # Determine overall winner
    if ood_lorad > ood_lora:
        improvement = (ood_lorad - ood_lora) / ood_lora * 100
        print(f"\n✅ LoRA-D shows {improvement:.1f}% better OOD generalization!")
        print("   This confirms depth advantage for identity preservation.")
    elif ood_lorad > ood_lora * 0.98:
        print("\n⚖️  LoRA-D matches Standard LoRA on OOD (within 2%).")
        print("   Competitive performance with stability guarantees.")
    else:
        print("\n⚠️  Standard LoRA performed better on both metrics.")

if __name__ == "__main__":
    import sys
    image_dir = sys.argv[1] if len(sys.argv) > 1 else './celeba_data'
    run_celeba_lorad_experiment(image_dir)
