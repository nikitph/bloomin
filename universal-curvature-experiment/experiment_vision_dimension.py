import torch
import torchvision
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# --- Data & Model Loading (Adapted from experiment_vision_control.py) ---

def get_cifar_images(n_images=1000): # Increased to 1000 for better PCA
    print(f"Loading CIFAR-100 (Top {n_images})...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    images = []
    count = 0
    for batch, _ in loader:
        images.append(batch)
        count += batch.size(0)
        if count >= n_images:
            break
            
    images = torch.cat(images, dim=0)[:n_images]
    print(f"  ✓ Loaded {images.shape} images")
    return images

def get_resnet_embeddings(images):
    print("\nExtracting ResNet-50 features...")
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    with torch.no_grad():
        emb = feature_extractor(images)
        emb = emb.squeeze()
    print(f"  ✓ ResNet shape: {emb.shape}")
    return emb.numpy()

def get_clip_embeddings(images_tensor):
    print("\nExtracting CLIP (ViT-B/32) features...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=images_tensor)
    print(f"  ✓ CLIP shape: {outputs.shape}")
    return outputs.numpy()

# --- Analysis ---

def analyze_dimension(embeddings, label):
    # Normalize first? Usually PCA is done on raw data to see spread, 
    # but for "Semantic Geometry" we care about the normalized manifold.
    # However, PCA on normalized data (on sphere) is effectively analyzing the tangent space.
    # Let's do PCA on Normalized vectors to be consistent with the "Projective Sphere" view.
    
    # Normalize
    X = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    pca = PCA()
    pca.fit(X)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d_90 = np.searchsorted(cumsum, 0.90) + 1
    d_95 = np.searchsorted(cumsum, 0.95) + 1
    d_99 = np.searchsorted(cumsum, 0.99) + 1
    
    print(f"\nModel: {label}")
    print(f"  Embedding Dim: {embeddings.shape[1]}")
    print(f"  Intrinsic Dim (90%): {d_90}")
    print(f"  Intrinsic Dim (95%): {d_95}")
    print(f"  Intrinsic Dim (99%): {d_99}")
    print(f"  % Coordinates Used (99%): {d_99 / embeddings.shape[1] * 100:.1f}%")
    
    return cumsum, d_95, d_99

def run_vision_dimension_sweep():
    # 1. Get Data
    n_samples = 2000 # Enough to estimate dim ~500
    images = get_cifar_images(n_samples)
    
    # 2. Extract
    emb_resnet = get_resnet_embeddings(images)
    emb_clip = get_clip_embeddings(images)
    
    # 3. Analyze
    results = {}
    results['ResNet-50 (Visual)'] = analyze_dimension(emb_resnet, 'ResNet-50')
    results['CLIP (Semantic)'] = analyze_dimension(emb_clip, 'CLIP ViT-B/32')
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    colors = {'ResNet-50 (Visual)': 'red', 'CLIP (Semantic)': 'blue'}
    
    for name, (curve, d95, d99) in results.items():
        # Plot only up to 512 dimensions for comparison or full?
        # ResNet goes to 2048, CLIP to 512.
        # Let's plot percentage of variance vs number of dimensions
        plt.plot(curve[:1000], label=f"{name} (99%={d99}D)", color=colors[name], linewidth=2)
        plt.plot(d99, 0.99, 'o', color=colors[name])
        
    plt.axhline(0.99, color='k', linestyle='--', alpha=0.3, label='99% Variance')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Cumulative Variance')
    plt.title('Intrinsic Dimension: Visual (ResNet) vs Semantic (CLIP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 600) # Zoom in on the interesting part
    plt.ylim(0, 1.05)
    
    output_path = 'results/vision_dimension_sweep.png'
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_vision_dimension_sweep()
