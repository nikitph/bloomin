import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import CLIPModel, AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import os
import urllib.request
from tqdm import tqdm

# --- 1. Data Loading ---

def get_text_data(n_samples=2000):
    print(f"Loading Text Data ({n_samples} samples)...")
    try:
        url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
        lines = [l.strip() for l in text.splitlines() if len(l) > 3]
        
        # Make sentences to be more like "context" for BERT
        sentences = [f"The word is {w}." for w in lines]
        return sentences[:n_samples]
    except Exception as e:
        print(f"Error fetching text: {e}")
        return [f"This is sample sentence {i}" for i in range(n_samples)]

def get_images(n_samples=2000):
    print(f"Loading CIFAR-100 Images ({n_samples} samples)...")
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
        if count >= n_samples:
            break
            
    images = torch.cat(images, dim=0)[:n_samples]
    return images

# --- 2. Feature Extraction ---

def get_bert_embeddings(texts):
    print("\nExtracting BERT embeddings...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token
            emb = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(emb)
            
    return np.vstack(embeddings)

def get_resnet_features(images):
    print("\nExtracting ResNet-50 features...")
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    with torch.no_grad():
        features = feature_extractor(images)
        features = features.squeeze().numpy()
    return features

def get_clip_features(images):
    print("\nExtracting CLIP features...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    with torch.no_grad():
        features = model.get_image_features(pixel_values=images).numpy()
    return features

# --- 3. Analysis ---

def analyze_intrinsic_dim(features, name, full_dim):
    # Normalize? 
    # For intrinsic dimension of the *manifold*, we usually look at the data as-is 
    # OR normalized if the space is spherical.
    # Given we found K=1 for all normalized, the "Distribution on the Sphere" is the key.
    # So let's Normalize first to be consistent with the "Spherical Geometry" premise.
    
    X = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    pca = PCA()
    pca.fit(X)
    
    cumsum = pca.explained_variance_ratio_.cumsum()
    d_intrinsic = np.sum(cumsum < 0.95) + 1
    
    compression = full_dim / d_intrinsic
    
    print(f"\nModel: {name}")
    print(f"  Full Dim: {full_dim}")
    print(f"  Intrinsic Dim (95%): {d_intrinsic}")
    print(f"  Compression Ratio: {compression:.2f}x")
    
    return d_intrinsic

def run_comparison():
    n_samples = 2000
    
    # Text
    texts = get_text_data(n_samples)
    bert_emb = get_bert_embeddings(texts)
    
    # Images
    images = get_images(n_samples)
    resnet_emb = get_resnet_features(images)
    clip_emb = get_clip_features(images)
    
    print("\n" + "="*50)
    print("INTRINSIC DIMENSION COMPARISON")
    print("="*50)
    
    analyze_intrinsic_dim(bert_emb, "BERT (Language)", 768)
    analyze_intrinsic_dim(resnet_emb, "ResNet-50 (Vision)", 2048)
    analyze_intrinsic_dim(clip_emb, "CLIP (Vision-Semantic)", 512)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    run_comparison()
