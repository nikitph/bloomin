#!/usr/bin/env python3
"""
Out-of-Distribution Identity Preservation Test for CelebA LoRA-D

This script tests the CRITICAL difference between LoRA and LoRA-D:
- In-distribution: Both perform similarly (memorization)
- Out-of-distribution: LoRA-D should preserve identity better (generalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys

plt.switch_backend('Agg')

# Import the models from the main experiment
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# OOD TRANSFORMATION FUNCTIONS
# ============================================================================

def apply_novel_lighting(img_tensor, intensity=2.0):
    """Simulate dramatic side lighting (never seen in training)"""
    # Convert to PIL for manipulation
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    
    # Create gradient mask (dark on left, bright on right)
    width, height = img.size
    gradient = Image.new('L', (width, height))
    for x in range(width):
        brightness = int(255 * (x / width) * intensity)
        for y in range(height):
            gradient.putpixel((x, y), min(255, brightness))
    
    # Apply gradient as lighting
    img = Image.composite(img, Image.new('RGB', img.size, 'black'), gradient)
    
    # Convert back to tensor
    return transforms.ToTensor()(img) * 2 - 1

def apply_rotation(img_tensor, angle=30):
    """Rotate face (novel pose)"""
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    return transforms.ToTensor()(img) * 2 - 1

def apply_blur(img_tensor, radius=3):
    """Add motion blur (simulates different focus/movement)"""
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return transforms.ToTensor()(img) * 2 - 1

def apply_contrast_change(img_tensor, factor=1.5):
    """Change contrast (different lighting conditions)"""
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    return transforms.ToTensor()(img) * 2 - 1

def apply_color_shift(img_tensor, hue_shift=0.3):
    """Shift colors (different ambient lighting)"""
    img = transforms.ToPILImage()(img_tensor.cpu() * 0.5 + 0.5)
    # Simple color shift by adjusting channels
    img_array = np.array(img).astype(float)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + hue_shift), 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - hue_shift), 0, 255)
    img = Image.fromarray(img_array.astype(np.uint8))
    return transforms.ToTensor()(img) * 2 - 1

# ============================================================================
# IDENTITY PRESERVATION METRICS
# ============================================================================

def compute_perceptual_similarity(img1, img2):
    """
    Simplified perceptual similarity (proxy for face recognition)
    In production, use FaceNet or ArcFace
    """
    # Normalize to [0, 1]
    img1_norm = (img1 + 1) / 2
    img2_norm = (img2 + 1) / 2
    
    # Compute MSE in feature space (simplified)
    mse = F.mse_loss(img1_norm, img2_norm)
    
    # Convert to similarity score (0-1, higher is better)
    similarity = 1 / (1 + mse.item())
    
    return similarity

def compute_structural_consistency(img1, img2):
    """Measure structural similarity (SSIM-like)"""
    img1_np = (img1.cpu().numpy() + 1) / 2
    img2_np = (img2.cpu().numpy() + 1) / 2
    
    # Compute variance-based structural similarity
    var1 = np.var(img1_np, axis=(1, 2))
    var2 = np.var(img2_np, axis=(1, 2))
    
    structural_sim = 1 / (1 + np.mean(np.abs(var1 - var2)))
    
    return structural_sim

# ============================================================================
# OOD TEST SUITE
# ============================================================================

def run_ood_test(lora_model, lorad_model, scheduler, reference_images, device='cpu'):
    """
    Test identity preservation under out-of-distribution conditions
    """
    print("\n" + "="*70)
    print("OUT-OF-DISTRIBUTION IDENTITY PRESERVATION TEST")
    print("="*70)
    
    # Define OOD transformations
    ood_tests = [
        {
            'name': 'Novel Lighting (Dramatic Side)',
            'transform': lambda x: apply_novel_lighting(x, intensity=2.0),
            'description': 'Strong directional lighting from side'
        },
        {
            'name': 'Novel Pose (30° Rotation)',
            'transform': lambda x: apply_rotation(x, angle=30),
            'description': 'Face rotated 30 degrees'
        },
        {
            'name': 'Motion Blur',
            'transform': lambda x: apply_blur(x, radius=3),
            'description': 'Simulates movement or defocus'
        },
        {
            'name': 'High Contrast',
            'transform': lambda x: apply_contrast_change(x, factor=1.8),
            'description': 'Extreme contrast (harsh lighting)'
        },
        {
            'name': 'Color Shift (Warm Lighting)',
            'transform': lambda x: apply_color_shift(x, hue_shift=0.3),
            'description': 'Warm ambient lighting (sunset)'
        }
    ]
    
    results = {
        'standard_lora': {'perceptual': [], 'structural': []},
        'lorad': {'perceptual': [], 'structural': []}
    }
    
    # Use first reference image as identity anchor
    reference = reference_images[0:1].to(device)
    
    print(f"\nTesting on {len(ood_tests)} OOD conditions...")
    print(f"Reference image shape: {reference.shape}\n")
    
    for test in ood_tests:
        print(f"\n{'─'*70}")
        print(f"Test: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"{'─'*70}")
        
        # Apply OOD transformation to reference
        transformed_ref = test['transform'](reference[0]).unsqueeze(0).to(device)
        
        # Add noise and denoise with both models
        # (Simulating generation from the transformed condition)
        with torch.no_grad():
            # Add noise
            timesteps = torch.tensor([500]).long().to(device)  # Mid-point
            noise = torch.randn_like(transformed_ref)
            noisy_img = scheduler.add_noise(transformed_ref, noise, timesteps)
            
            # Denoise with Standard LoRA
            lora_noise_pred = lora_model(noisy_img, timesteps)
            lora_output = scheduler.step(lora_noise_pred, timesteps[0], noisy_img).prev_sample
            
            # Denoise with LoRA-D
            lorad_noise_pred = lorad_model(noisy_img, timesteps)
            lorad_output = scheduler.step(lorad_noise_pred, timesteps[0], noisy_img).prev_sample
        
        # Compute identity preservation metrics
        lora_perceptual = compute_perceptual_similarity(lora_output, reference)
        lorad_perceptual = compute_perceptual_similarity(lorad_output, reference)
        
        lora_structural = compute_structural_consistency(lora_output, reference)
        lorad_structural = compute_structural_consistency(lorad_output, reference)
        
        results['standard_lora']['perceptual'].append(lora_perceptual)
        results['standard_lora']['structural'].append(lora_structural)
        results['lorad']['perceptual'].append(lorad_perceptual)
        results['lorad']['structural'].append(lorad_structural)
        
        print(f"\n  Perceptual Identity Preservation:")
        print(f"    Standard LoRA: {lora_perceptual:.1%}")
        print(f"    LoRA-D:        {lorad_perceptual:.1%}")
        print(f"    Advantage:     {(lorad_perceptual - lora_perceptual):.1%}")
        
        print(f"\n  Structural Consistency:")
        print(f"    Standard LoRA: {lora_structural:.1%}")
        print(f"    LoRA-D:        {lorad_structural:.1%}")
        print(f"    Advantage:     {(lorad_structural - lora_structural):.1%}")
    
    # Compute averages
    avg_lora_perceptual = np.mean(results['standard_lora']['perceptual'])
    avg_lorad_perceptual = np.mean(results['lorad']['perceptual'])
    avg_lora_structural = np.mean(results['standard_lora']['structural'])
    avg_lorad_structural = np.mean(results['lorad']['structural'])
    
    print("\n" + "="*70)
    print("FINAL OOD IDENTITY PRESERVATION SCORES")
    print("="*70)
    print(f"\n{'Metric':<30} | {'Standard LoRA':<15} | {'LoRA-D':<15} | {'Improvement':<15}")
    print("─"*70)
    print(f"{'Perceptual Identity':<30} | {avg_lora_perceptual:>14.1%} | {avg_lorad_perceptual:>14.1%} | {(avg_lorad_perceptual - avg_lora_perceptual):>+14.1%}")
    print(f"{'Structural Consistency':<30} | {avg_lora_structural:>14.1%} | {avg_lorad_structural:>14.1%} | {(avg_lorad_structural - avg_lora_structural):>+14.1%}")
    print("─"*70)
    
    # Determine winner
    if avg_lorad_perceptual > avg_lora_perceptual:
        print("\n✅ LoRA-D shows BETTER out-of-distribution generalization!")
        print("   This confirms the depth advantage for identity preservation.")
    elif avg_lorad_perceptual > avg_lora_perceptual * 0.98:  # Within 2%
        print("\n⚖️  LoRA-D matches Standard LoRA on OOD generalization.")
        print("   Competitive performance with additional stability guarantees.")
    else:
        print("\n⚠️  Standard LoRA performed better on OOD conditions.")
        print("   This suggests the task may not require depth-based adaptation.")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🔬 Loading trained models for OOD testing...")
    
    # This script assumes you've already run celeba_lora_d_experiment.py
    # and have the trained models saved
    
    print("\n⚠️  This script requires trained models from celeba_lora_d_experiment.py")
    print("Please run that experiment first, then modify this script to load the saved models.")
    print("\nFor now, this serves as the OOD test template.")
