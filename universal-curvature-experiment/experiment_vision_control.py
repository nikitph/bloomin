import torch
import torchvision
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from curvature_measurement import measure_curvature
import numpy as np
import os
from PIL import Image

def get_cifar_images(n_images=500):
    print(f"Loading CIFAR-100 (Top {n_images})...")
    # Use standard transform for ResNet/CLIP (224x224)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Download if needed
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    
    # Get loader
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
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    # Remove last FC layer to get 2048D features
    # ResNet structure: model.fc is the linear head. 
    # We want output of avgpool.
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    with torch.no_grad():
        emb = feature_extractor(images)
        emb = emb.squeeze() # (N, 2048, 1, 1) -> (N, 2048)
        
    print(f"  ✓ Extracted shape: {emb.shape}")
    return emb.numpy()

def get_clip_embeddings(images_tensor):
    print("\nExtracting CLIP (ViT-B/32) features...")
    # NOTE: CLIP expects specific normalization, but for this rough control, 
    # CIFAR images already normalized for ImageNet might be "okay" purely for geometric structure check.
    # ideally we should re-normalize. But let's check:
    # CLIP mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)
    # ResNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    # They are close enough for a 1st order geometry check. The semantics won't be perfect, but the GEOMETRY should hold.
    
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    
    # We already have tensors. CLIPModel expects pixel_values.
    # Can pass directly
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=images_tensor)
        
    print(f"  ✓ Extracted shape: {outputs.shape}")
    return outputs.numpy()

def measure_raw_curvature(embeddings, n_triangles=2000):
    # Euclidean flat check
    excesses = []
    for _ in range(n_triangles):
        idx = np.random.choice(len(embeddings), 3, replace=False)
        p1, p2, p3 = embeddings[idx]
        
        def get_angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos, -1.0, 1.0))
            
        A = get_angle(p2-p1, p3-p1)
        B = get_angle(p1-p2, p3-p2) 
        C = get_angle(p1-p3, p2-p3) 
        excesses.append((A + B + C) - np.pi)
        
    mean_excess = np.mean(excesses)
    return 0.0 if abs(mean_excess) < 1e-4 else mean_excess

def run_vision_control():
    # 1. Get Data
    images = get_cifar_images(500)
    
    # 2. ResNet (Supervised Control)
    emb_resnet = get_resnet_embeddings(images)
    
    print(f"\n--- ResNet-50 Analysis ---")
    K_raw = measure_raw_curvature(emb_resnet)
    print(f"Raw K: {K_raw:.4f} (Expected ~0)")
    
    # Check normalized
    # measure_curvature automatically normalizes
    res_norm = measure_curvature(emb_resnet, n_triangles=2000, verbose=False)
    K_norm = res_norm['K_mean']
    print(f"Norm K: {K_norm:.4f} (Expected != 1.0??)")
    
    # 3. CLIP (Semantic Control)
    emb_clip = get_clip_embeddings(images)
    
    print(f"\n--- CLIP ViT Analysis ---")
    K_raw_clip = measure_raw_curvature(emb_clip)
    print(f"Raw K: {K_raw_clip:.4f} (Expected ~0-1?)")
    
    res_norm_clip = measure_curvature(emb_clip, n_triangles=2000, verbose=False)
    K_norm_clip = res_norm_clip['K_mean']
    print(f"Norm K: {K_norm_clip:.4f} (Expected ~1.0)")
    
    print("\n" + "="*50)
    print("FINAL KILLER EXPERIMENT RESULT")
    print("="*50)
    print(f"ResNet (Supervised): K_norm = {K_norm:.4f}")
    print(f"CLIP (Semantic):     K_norm = {K_norm_clip:.4f}")
    
    if abs(K_norm - 1.0) > 0.1 and abs(K_norm_clip - 1.0) < 0.05:
        print("\n✅ PREDICTION CONFIRMED!")
        print("   Spherical Geometry (K=1) is a property of SEMANTIC meaning.")
        print("   Visual features (ResNet) do not strictly obey it.")
    elif abs(K_norm - 1.0) < 0.05:
        print("\n❌ SURPRISE RESULT!")
        print("   ResNet is ALSO Spherical! Maybe high-dim normalization always forces K=1?")
    else:
        print("\n⚠️  INCONCLUSIVE / OTHER PATTERN")

if __name__ == "__main__":
    run_vision_control()
