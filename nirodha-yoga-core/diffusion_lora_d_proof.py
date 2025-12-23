import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import os

plt.switch_backend('Agg')

# ============================================================================
# 1. BASE DIFFUSION MODEL (Tiny MLP)
# ============================================================================

class DiffusionBase(nn.Module):
    """Tiny MLP for 2D Point Diffusion"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim), # [x, y] + timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # Predict noise
        )
        
    def forward(self, x, t):
        # x: [B, 2], t: [B, 1]
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

# ============================================================================
# 2. NIRODHA-D BLOCK (Identity as Procedure)
# ============================================================================

class NirodhaDepthBlock(nn.Module):
    """Adds a processing step to the diffusion output"""
    def __init__(self, dim=2, hidden_dim=64, beta=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        # Identity Initialization: Block starts as zero-mapping
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
        self.beta = beta
        self.anchor = None
        
    def set_anchor(self):
        self.anchor = {k: v.clone().detach() for k, v in self.named_parameters()}
        
    def nirodha_op(self, x):
        return x / (1.0 + self.beta * torch.abs(x))
        
    def apply_regulation(self):
        if self.anchor and self.training:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in self.anchor:
                        anchor_val = self.anchor[name].to(param.device)
                        delta = param.data - anchor_val
                        param.data.copy_(anchor_val + self.nirodha_op(delta))

    def forward(self, x, t):
        self.apply_regulation()
        # x is the noise prediction from base or previous block
        xt = torch.cat([x, t], dim=-1)
        return x + self.net(xt) # Residual addition

# ============================================================================
# 3. LORA WRAPPER (Standard Width-based)
# ============================================================================

class LoRAWrappedBase(nn.Module):
    """Adds standard LoRA deltas to the base model weights"""
    def __init__(self, base_model, rank=8):
        super().__init__()
        self.base = base_model
        # Freeze base
        for p in self.base.parameters(): p.requires_grad = False
        
        # Add LoRA layers to the linear weights
        self.lora_A = nn.Parameter(torch.randn(base_model.net[0].in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, base_model.net[0].out_features))
        
    def forward(self, x, t):
        # This is a simplified LoRA for PoC
        h = torch.cat([x, t], dim=-1)
        # Base pass
        base_out = self.base(x, t)
        # LoRA pass
        lora_delta = (h @ self.lora_A) @ self.lora_B
        # Correct dimensions for last layer output
        # (Simplified: just apply to the final output for PoC)
        return base_out + lora_delta[:, :2]

# ============================================================================
# 4. DATA GENERATOR (Identity & Pose)
# ============================================================================

def generate_star_points(n=1000):
    """Our Target Identity: A 5-pointed star"""
    pts = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append([np.cos(angle), np.sin(angle)])
        angle_inner = 2 * np.pi * (i + 0.5) / 5
        pts.append([0.4 * np.cos(angle_inner), 0.4 * np.sin(angle_inner)])
    pts.append(pts[0])
    pts = np.array(pts)
    
    # Interpolate for more points
    final_pts = []
    for i in range(len(pts)-1):
        step = np.linspace(pts[i], pts[i+1], n//10)
        final_pts.extend(step)
    return torch.tensor(final_pts, dtype=torch.float32)

def rotate_points(pts, angle_deg):
    angle = np.deg2rad(angle_deg)
    rot = torch.tensor([[np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]], dtype=torch.float32)
    return pts @ rot.T

# ============================================================================
# 5. TRAINING & SAMPLING
# ============================================================================

def train_diffusion(model, target_pts, iterations=5000, device='cpu', is_nirodha=False):
    model.to(device); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    for i in range(iterations):
        # 1. Random Pose for the target
        angle = random.uniform(0, 360)
        batch_pts = rotate_points(target_pts, angle).to(device)
        
        # 2. Diffusion Step
        t = torch.rand(batch_pts.size(0), 1).to(device)
        noise = torch.randn_like(batch_pts).to(device)
        xt = (1 - t) * batch_pts + t * noise 
        
        # 3. Predict ground truth directly (PoC simplification)
        pred = model(xt, t)
        loss = F.mse_loss(pred, batch_pts)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print(f"   Step {i:4d} | Loss: {loss.item():.6f}")
    return model

@torch.no_grad()
def sample_diffusion(model, angle=0, device='cpu'):
    model.eval()
    noise = torch.randn(200, 2).to(device)
    # We directly predict the target from noise in our PoC (t=1 approximation)
    t = torch.ones(200, 1).to(device)
    pred = model(noise, t)
    return pred.cpu()

# ============================================================================
# 7. QUANTITATIVE METRICS
# ============================================================================

def chamfer_distance(p1, p2):
    """Simple Chamfer Distance approximation"""
    from scipy.spatial import cKDTree
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    d1, _ = tree1.query(p2)
    d2, _ = tree2.query(p1)
    return (np.mean(d1) + np.mean(d2)) / 2

def fourier_descriptor_match(samples, target):
    """Rotation-invariant shape matching via Fourier Descriptors"""
    def get_fd(pts):
        pts = pts - np.mean(pts, axis=0) # Center
        z = pts[:, 0] + 1j * pts[:, 1]
        fft = np.fft.fft(z)
        coeffs = np.abs(fft[1:11]) # Low frequency
        coeffs /= (np.linalg.norm(coeffs) + 1e-6) # Normalize
        return coeffs
    
    # We need to sample same number of points to compare FFT
    n = min(len(samples), len(target))
    fd1 = get_fd(samples[:n])
    fd2 = get_fd(target[:n])
    similarity = np.dot(fd1, fd2) # Cosine similarity of magnitudes
    return similarity

def compute_metrics(samples, target_shape):
    """Quantify model performance"""
    samples_np = samples.numpy()
    target_np = target_shape.numpy()
    
    chamfer = chamfer_distance(samples_np, target_np)
    fd_similarity = fourier_descriptor_match(samples_np, target_np)
    std_dev = samples.std(dim=0).mean().item()
    
    return {
        'chamfer': chamfer,
        'shape_match': fd_similarity,
        'tightness': std_dev
    }

# ============================================================================
# 8. SHAPES
# ============================================================================

def generate_pentagon_points(n=1000):
    """Test shape for transfer: 5 edges"""
    pts = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append([np.cos(angle), np.sin(angle)])
    pts.append(pts[0])
    pts = np.array(pts)
    final_pts = []
    for i in range(len(pts)-1):
        step = np.linspace(pts[i], pts[i+1], n//5)
        final_pts.extend(step)
    return torch.tensor(final_pts, dtype=torch.float32)

# ============================================================================
# 9. INTEGRATED EXPERIMENT RUNNER
# ============================================================================

def run_lora_d_proof():
    print("\n" + "="*60)
    print("LO-RA D (DEPTH) QUANTITATIVE & ABLATION PROOF")
    print("="*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    star_target = generate_star_points(1000)
    pentagon_target = generate_pentagon_points(1000)
    
    # --- PART 1: ABLATION STUDY ---
    print("\n📊 RUNNING DEPTH ABLATION STUDY...")
    ablation_results = {}
    
    for n_blocks in [1, 2, 8]:
        print(f"\n[Ablation] Testing {n_blocks} Nirodha Blocks...")
        base = DiffusionBase(hidden_dim=256) # More capacity
        blocks = nn.ModuleList([NirodhaDepthBlock(dim=2, hidden_dim=128, beta=2) for _ in range(n_blocks)])
        
        class NirodhaModel(nn.Module):
            def __init__(self, base, blocks):
                super().__init__()
                self.base = base
                for p in self.base.parameters(): p.requires_grad = False
                self.blocks = blocks
            def forward(self, x, t):
                h = self.base(x, t); 
                for b in self.blocks: h = b(h, t)
                return h
            def set_anchor(self):
                for b in self.blocks: b.set_anchor()
        
        model = NirodhaModel(base, blocks)
        model.set_anchor()
        train_diffusion(model, star_target, iterations=10000, device=device) # Full training
        
        # Eval
        samples = sample_diffusion(model, angle=0, device=device)
        metrics = compute_metrics(samples, star_target)
        ablation_results[n_blocks] = metrics
        print(f"   Metrics ({n_blocks}L): Chamfer={metrics['chamfer']:.3f}, Match={metrics['shape_match']:.3f}")

    # --- PART 2: BASELINE COMPARISON (Standard LoRA) ---
    print("\n📊 RUNNING STANDARD LoRA BASELINE...")
    base_bl = DiffusionBase(hidden_dim=256)
    lora_model = LoRAWrappedBase(base_bl, rank=32) # Higher rank for fair baseline
    train_diffusion(lora_model, star_target, iterations=10000, device=device)
    
    samples_lora = sample_diffusion(lora_model, angle=0, device=device)
    metrics_lora = compute_metrics(samples_lora, star_target)
    print(f"   Metrics (LoRA): Chamfer={metrics_lora['chamfer']:.3f}, Match={metrics_lora['shape_match']:.3f}")

    # --- PART 3: CROSS-TASK TRANSFER ---
    print("\n📊 RUNNING CROSS-TASK TRANSFER (Stars -> Pentagons)...")
    samples_transfer = sample_diffusion(model, angle=0, device=device) 
    metrics_transfer = compute_metrics(samples_transfer, pentagon_target)
    print(f"   Metrics (Star Model on Pentagon): Chamfer={metrics_transfer['chamfer']:.3f}, Match={metrics_transfer['shape_match']:.3f}")

    # Final Summary Table
    print("\n" + "="*60)
    print("FINAL QUANTITATIVE COMPARISON")
    print("="*60)
    print(f"{'Model':<20} | {'Chamfer':<10} | {'Shape Match':<10} | {'Tightness':<10}")
    print("-" * 60)
    print(f"{'Standard LoRA':<20} | {metrics_lora['chamfer']:10.3f} | {metrics_lora['shape_match']:10.3f} | {metrics_lora['tightness']:10.3f}")
    for n, m in ablation_results.items():
        print(f"{f'LoRA-D ({n} Blocks)':<20} | {m['chamfer']:10.3f} | {m['shape_match']:10.3f} | {m['tightness']:10.3f}")
    print("-" * 60)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(samples_lora[:,0], samples_lora[:,1], s=5, c='red', alpha=0.5)
    axes[0].set_title("Standard LoRA")
    
    n4_samples = sample_diffusion(model, 0, device)
    axes[1].scatter(n4_samples[:,0], n4_samples[:,1], s=5, c='green', alpha=0.5)
    axes[1].set_title("LoRA-D (4 Blocks)")
    
    axes[2].scatter(samples_transfer[:,0], samples_transfer[:,1], s=5, c='blue', alpha=0.5)
    axes[2].set_title("Transfer (Star -> Pent)")
    
    save_path = os.path.join(os.path.dirname(__file__), 'diffusion_lora_d_metrics.png')
    plt.savefig(save_path)
    print(f"\n✅ Results saved to {save_path}")

if __name__ == "__main__":
    run_lora_d_proof()
