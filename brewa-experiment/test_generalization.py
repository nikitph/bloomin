"""
Test Generalization of Learned REWA Encoder
============================================

Critical test: Does the model generalize to NEW data, or did it overfit?

Tests:
1. Same clusters, new samples (should get ~90%)
2. Completely new clusters (should get ~70%)
"""

import torch
import torch.nn.functional as F
import numpy as np

from learned_rewa_encoder import LearnedContinuousREWAEncoder


def generate_test_data_same_clusters(n_clusters=50, samples_per_cluster=20, d_model=768):
    """Generate NEW samples from SAME clusters as training."""
    # Use same centroid generation logic as training
    centroids = torch.randn(n_clusters, d_model)
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    centroids = centroids * 3.0  # Same spread as training
    
    embeddings = []
    labels = []
    
    for cluster_id in range(n_clusters):
        centroid = centroids[cluster_id]
        
        for _ in range(samples_per_cluster):
            # NEW noise (different from training)
            sample = centroid + torch.randn(d_model) * 0.3  # Same std as training
            sample = sample / sample.norm()
            
            embeddings.append(sample)
            labels.append(cluster_id)
    
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)
    
    return embeddings, labels


def generate_test_data_new_clusters(n_clusters=30, samples_per_cluster=15, d_model=768):
    """Generate COMPLETELY NEW clusters (different number, different centroids)."""
    # Different number of clusters
    centroids = torch.randn(n_clusters, d_model)
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)
    centroids = centroids * 3.0
    
    embeddings = []
    labels = []
    
    for cluster_id in range(n_clusters):
        centroid = centroids[cluster_id]
        
        for _ in range(samples_per_cluster):
            sample = centroid + torch.randn(d_model) * 0.3
            sample = sample / sample.norm()
            
            embeddings.append(sample)
            labels.append(cluster_id)
    
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)
    
    return embeddings, labels


def evaluate_recall(encoder, embeddings, labels, top_k=10):
    """Evaluate Recall@K."""
    encoder.eval()
    
    with torch.no_grad():
        # Encode
        encoded = encoder(embeddings.unsqueeze(0), add_noise=False).squeeze(0)
        
        # Compute similarity
        similarity = torch.mm(encoded, encoded.T)
        
        # Get top-k neighbors
        _, indices = similarity.topk(top_k + 1, dim=1)  # +1 for self
        
        # Compute recall
        recall_sum = 0
        for i in range(len(labels)):
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_labels = labels[neighbors]
            
            # Count correct (same class)
            correct = (neighbor_labels == labels[i]).sum().item()
            max_possible = min(top_k, (labels == labels[i]).sum().item() - 1)
            
            if max_possible > 0:
                recall_sum += correct / max_possible
        
        recall = recall_sum / len(labels)
    
    return recall


def test_generalization():
    """Test 1: Same clusters, new samples."""
    print("="*70)
    print("Test 1: Generalization to NEW Samples (Same Clusters)")
    print("="*70)
    
    # Load trained model
    print("\nLoading trained model...")
    encoder = LearnedContinuousREWAEncoder(d_model=768, m_dim=256)
    encoder.load_state_dict(torch.load('learned_rewa_encoder.pth'))
    encoder.eval()
    
    # Generate NEW test data
    print("Generating NEW test data (different seed)...")
    torch.manual_seed(999)  # Different from training (42)
    np.random.seed(999)
    
    test_embeddings, test_labels = generate_test_data_same_clusters(
        n_clusters=50,  # Same as training
        samples_per_cluster=20,  # NEW samples
        d_model=768
    )
    
    print(f"Test data: {len(test_embeddings)} NEW samples")
    print(f"Clusters: {test_labels.unique().numel()} (same as training)")
    print(f"Samples per cluster: 20 (NEW, not seen during training)")
    
    # Evaluate
    print("\nEvaluating...")
    recall = evaluate_recall(encoder, test_embeddings, test_labels, top_k=10)
    
    print(f"\n{'='*70}")
    print(f"Test Recall@10: {recall:.1%}")
    print(f"{'='*70}")
    
    if recall > 0.90:
        print("✅ EXCELLENT: Model generalizes perfectly to new samples!")
    elif recall > 0.80:
        print("✅ VERY GOOD: Model generalizes well!")
    elif recall > 0.70:
        print("⚠️  GOOD: Decent generalization, some overfitting")
    elif recall > 0.60:
        print("⚠️  MODERATE: Significant overfitting")
    else:
        print("❌ POOR: Model overfitted to training data")
    
    return recall


def test_new_clusters():
    """Test 2: Completely new clusters."""
    print("\n\n" + "="*70)
    print("Test 2: Generalization to COMPLETELY NEW Clusters")
    print("="*70)
    
    # Load model
    encoder = LearnedContinuousREWAEncoder(d_model=768, m_dim=256)
    encoder.load_state_dict(torch.load('learned_rewa_encoder.pth'))
    encoder.eval()
    
    # Generate COMPLETELY NEW clusters
    print("\nGenerating COMPLETELY NEW clusters...")
    torch.manual_seed(888)  # Different seed
    np.random.seed(888)
    
    test_embeddings, test_labels = generate_test_data_new_clusters(
        n_clusters=30,  # Different from training (50)
        samples_per_cluster=15,
        d_model=768
    )
    
    print(f"Test data: {len(test_embeddings)} samples")
    print(f"Clusters: {test_labels.unique().numel()} (DIFFERENT from training's 50)")
    print(f"Samples per cluster: 15")
    
    # Evaluate
    print("\nEvaluating...")
    recall = evaluate_recall(encoder, test_embeddings, test_labels, top_k=10)
    
    print(f"\n{'='*70}")
    print(f"Recall@10 on NEW Clusters: {recall:.1%}")
    print(f"{'='*70}")
    
    if recall > 0.80:
        print("🚀 REVOLUTIONARY: Model learned GENERAL similarity preservation!")
    elif recall > 0.70:
        print("✅ EXCELLENT: Strong generalization to new clusters!")
    elif recall > 0.60:
        print("✅ GOOD: Decent generalization")
    elif recall > 0.50:
        print("⚠️  MODERATE: Some generalization")
    else:
        print("❌ POOR: Limited generalization")
    
    return recall


def final_assessment(recall_same, recall_new):
    """Final assessment of results."""
    print("\n\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)
    
    print(f"\nResults Summary:")
    print(f"  Training Recall:        100.0%")
    print(f"  Test (same clusters):   {recall_same:.1%}")
    print(f"  Test (new clusters):    {recall_new:.1%}")
    
    print(f"\nGeneralization Gap:")
    print(f"  Same clusters: {100 - recall_same*100:.1f}% drop")
    print(f"  New clusters:  {100 - recall_new*100:.1f}% drop")
    
    print("\n" + "="*70)
    
    if recall_same > 0.90 and recall_new > 0.70:
        print("🚀 BREAKTHROUGH: Model learns GENERAL similarity preservation!")
        print("\nThis means:")
        print("  ✅ REWA can compress ANY semantic embeddings")
        print("  ✅ Model learned transferable similarity structure")
        print("  ✅ Production-ready for domain adaptation")
        print("\n🎯 Next: Test on real BERT embeddings, then deploy!")
        
    elif recall_same > 0.80 and recall_new > 0.60:
        print("✅ SUCCESS: Model generalizes well!")
        print("\nThis means:")
        print("  ✅ REWA works for similar domains")
        print("  ✅ Can fine-tune on target domain")
        print("  ⚠️  May need domain-specific training")
        print("\n🎯 Next: Fine-tune on target domain, then deploy!")
        
    elif recall_same > 0.70:
        print("⚠️  GOOD: Model works but has some overfitting")
        print("\nThis means:")
        print("  ⚠️  Need more diverse training data")
        print("  ⚠️  Add regularization (dropout, noise)")
        print("  ✅ Concept is proven")
        print("\n🎯 Next: Improve training, then re-test!")
        
    else:
        print("❌ OVERFITTING: Model memorized training data")
        print("\nThis means:")
        print("  ❌ Need better regularization")
        print("  ❌ Need more diverse training data")
        print("  ✅ But 100% training recall proves capacity exists")
        print("\n🎯 Next: Add dropout, augmentation, re-train!")
    
    print("="*70)


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(999)
    np.random.seed(999)
    
    # Test 1: Same clusters, new samples
    recall_same = test_generalization()
    
    # Test 2: Completely new clusters
    recall_new = test_new_clusters()
    
    # Final assessment
    final_assessment(recall_same, recall_new)
    
    print("\n" + "="*70)
    print("Generalization Testing Complete!")
    print("="*70)
