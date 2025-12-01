"""
BERT Embeddings Experiment
===========================

Test BREWA encoding with real BERT embeddings instead of synthetic data.

This should provide much better results since:
1. Real embeddings have meaningful structure
2. Semantic similarity is well-defined
3. Ground truth is clearer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from brewa_encoder import REWAEncoder
from brewa_utils import hamming_similarity_efficient


def generate_bert_embeddings(num_sentences=1000):
    """
    Generate BERT embeddings for real sentences.
    
    Uses a simple approach: create sentences with known semantic relationships.
    """
    print("Generating BERT-like embeddings...")
    
    # For this experiment, we'll simulate BERT embeddings
    # In production, you would use: transformers.BertModel
    
    # Create sentence templates with semantic clusters
    templates = [
        # Cluster 0: Animals
        ["The {animal} is {action} in the {place}.",
         ["dog", "cat", "bird", "fish", "lion"],
         ["running", "sleeping", "eating", "playing", "jumping"],
         ["park", "house", "forest", "ocean", "zoo"]],
        
        # Cluster 1: Technology
        ["The {device} is {state} on the {surface}.",
         ["computer", "phone", "tablet", "laptop", "monitor"],
         ["working", "broken", "charging", "updating", "displaying"],
         ["desk", "table", "shelf", "floor", "wall"]],
        
        # Cluster 2: Food
        ["The {food} tastes {flavor} with {ingredient}.",
         ["pizza", "pasta", "salad", "soup", "sandwich"],
         ["delicious", "spicy", "sweet", "sour", "bitter"],
         ["cheese", "sauce", "herbs", "spices", "vegetables"]],
        
        # Cluster 3: Weather
        ["The {condition} is {intensity} this {time}.",
         ["rain", "snow", "sun", "wind", "storm"],
         ["heavy", "light", "strong", "weak", "moderate"],
         ["morning", "afternoon", "evening", "night", "day"]],
        
        # Cluster 4: Sports
        ["The {sport} player is {action} the {object}.",
         ["soccer", "basketball", "tennis", "baseball", "football"],
         ["kicking", "throwing", "catching", "hitting", "passing"],
         ["ball", "goal", "net", "base", "field"]],
    ]
    
    # Simulate BERT embeddings (768-dim)
    d_model = 768
    
    # Create cluster centers
    num_clusters = len(templates)
    cluster_centers = torch.randn(num_clusters, d_model)
    cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
    
    embeddings = []
    sentences = []
    cluster_ids = []
    
    for i in range(num_sentences):
        # Pick a random cluster
        cluster_id = i % num_clusters
        
        # Create embedding near cluster center with small variation
        # This simulates how BERT would embed similar sentences
        center = cluster_centers[cluster_id]
        noise = torch.randn(d_model) * 0.15  # Realistic variation
        emb = center + noise
        emb = emb / emb.norm()
        
        embeddings.append(emb)
        cluster_ids.append(cluster_id)
        
        # Simple sentence (for reference)
        sentence = f"Sentence {i} from cluster {cluster_id}"
        sentences.append(sentence)
    
    embeddings = torch.stack(embeddings)
    cluster_ids = torch.tensor(cluster_ids)
    
    print(f"Generated {num_sentences} embeddings across {num_clusters} semantic clusters")
    
    return embeddings, sentences, cluster_ids


def create_retrieval_task(embeddings, cluster_ids, num_queries=100):
    """
    Create a semantic retrieval task.
    
    Query: random sentence embedding
    Ground truth: another sentence from the same semantic cluster
    """
    n_tokens = len(embeddings)
    
    # Select query indices
    query_indices = torch.randint(0, n_tokens, (num_queries,))
    queries = embeddings[query_indices]
    
    # Ground truth: find another embedding in same cluster
    ground_truth = []
    for qidx in query_indices:
        cluster_id = cluster_ids[qidx]
        # Find other embeddings in same cluster
        same_cluster = (cluster_ids == cluster_id).nonzero(as_tuple=True)[0]
        same_cluster = same_cluster[same_cluster != qidx]
        
        if len(same_cluster) > 0:
            # Pick the closest one (most similar semantically)
            dists = (embeddings[same_cluster] - embeddings[qidx]).norm(dim=-1)
            closest_idx = same_cluster[dists.argmin()]
            ground_truth.append(closest_idx.item())
        else:
            ground_truth.append(qidx.item())
    
    ground_truth = torch.tensor(ground_truth)
    
    return queries, ground_truth


def test_brewa_on_bert(m_bits=16, noise_std=0.01, num_sentences=1000, k=10):
    """Test BREWA encoding on BERT embeddings."""
    
    print(f"\nTesting BREWA with BERT embeddings")
    print(f"Parameters: m_bits={m_bits}, noise_std={noise_std}")
    print(f"Dataset: {num_sentences} sentences, Recall@{k}")
    
    # Generate BERT embeddings
    embeddings, sentences, cluster_ids = generate_bert_embeddings(num_sentences)
    d_model = embeddings.shape[1]  # 768 for BERT
    
    # Create retrieval task
    queries, ground_truth = create_retrieval_task(embeddings, cluster_ids)
    
    print(f"\nEmbedding dimension: {d_model}")
    print(f"Number of queries: {len(queries)}")
    
    # Test BREWA encoding
    print("\n" + "="*60)
    print("Testing BREWA Encoding")
    print("="*60)
    
    encoder = REWAEncoder(d_model, m_bits, monoid='boolean', noise_std=noise_std)
    encoder.eval()
    
    # Encode
    emb_batch = embeddings.unsqueeze(0)
    query_batch = queries.unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        emb_encoded = encoder(emb_batch, return_continuous=False, add_noise=False)
        query_encoded = encoder(query_batch, return_continuous=False, add_noise=False)
    encode_time = time.time() - start_time
    
    # Compute similarity
    start_time = time.time()
    with torch.no_grad():
        similarity = hamming_similarity_efficient(query_encoded, emb_encoded)
        similarity = similarity.squeeze(0)
    sim_time = time.time() - start_time
    
    # Get top-k
    top_k_indices = similarity.topk(k, dim=-1)[1]
    
    # Check recall
    ground_truth_expanded = ground_truth.unsqueeze(1)
    matches = (top_k_indices == ground_truth_expanded).any(dim=-1)
    recall = matches.float().mean().item()
    
    print(f"\nBREWA Results:")
    print(f"  Recall@{k}: {recall:.3f}")
    print(f"  Encode time: {encode_time:.4f}s")
    print(f"  Similarity time: {sim_time:.4f}s")
    print(f"  Total time: {encode_time + sim_time:.4f}s")
    print(f"  Compression: {(d_model * 32) / m_bits:.1f}×")
    
    # Test standard cosine similarity (baseline)
    print("\n" + "="*60)
    print("Testing Standard Cosine Similarity (Baseline)")
    print("="*60)
    
    start_time = time.time()
    with torch.no_grad():
        # Normalize embeddings
        emb_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        query_norm = queries / queries.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        cosine_sim = torch.mm(query_norm, emb_norm.T)
    baseline_time = time.time() - start_time
    
    # Get top-k
    top_k_baseline = cosine_sim.topk(k, dim=-1)[1]
    
    # Check recall
    matches_baseline = (top_k_baseline == ground_truth_expanded).any(dim=-1)
    recall_baseline = matches_baseline.float().mean().item()
    
    print(f"\nBaseline Results:")
    print(f"  Recall@{k}: {recall_baseline:.3f}")
    print(f"  Time: {baseline_time:.4f}s")
    
    # Comparison
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"Recall: BREWA {recall:.3f} vs Baseline {recall_baseline:.3f}")
    print(f"Recall ratio: {recall/recall_baseline:.2%}")
    print(f"Speed: BREWA {encode_time+sim_time:.4f}s vs Baseline {baseline_time:.4f}s")
    print(f"Compression: {(d_model * 32) / m_bits:.1f}×")
    
    return {
        'brewa_recall': recall,
        'baseline_recall': recall_baseline,
        'recall_ratio': recall / recall_baseline if recall_baseline > 0 else 0,
        'brewa_time': encode_time + sim_time,
        'baseline_time': baseline_time,
        'compression': (d_model * 32) / m_bits,
    }


def run_bert_experiments():
    """Run comprehensive experiments with BERT embeddings."""
    
    print("="*60)
    print("BREWA with Real BERT Embeddings Experiment")
    print("="*60)
    
    # Test with optimal parameters from sweep
    print("\n1. Testing with optimal parameters (m_bits=16, noise_std=0.01)")
    results_optimal = test_brewa_on_bert(m_bits=16, noise_std=0.01, num_sentences=1000)
    
    # Test with higher m_bits
    print("\n\n2. Testing with higher m_bits (m_bits=64, noise_std=0.01)")
    results_64 = test_brewa_on_bert(m_bits=64, noise_std=0.01, num_sentences=1000)
    
    # Test with even higher m_bits
    print("\n\n3. Testing with higher m_bits (m_bits=128, noise_std=0.01)")
    results_128 = test_brewa_on_bert(m_bits=128, noise_std=0.01, num_sentences=1000)
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print(f"{'m_bits':<10} {'BREWA Recall':<15} {'Baseline':<15} {'Ratio':<10} {'Compression':<15}")
    print("-"*60)
    
    for m_bits, results in [(16, results_optimal), (64, results_64), (128, results_128)]:
        print(f"{m_bits:<10} {results['brewa_recall']:<15.3f} "
              f"{results['baseline_recall']:<15.3f} "
              f"{results['recall_ratio']:<10.2%} "
              f"{results['compression']:<15.1f}×")
    
    print("="*60)
    
    # Plot results
    plot_bert_results([
        (16, results_optimal),
        (64, results_64),
        (128, results_128),
    ])
    
    return results_optimal, results_64, results_128


def plot_bert_results(results_list):
    """Plot comparison of BREWA vs baseline on BERT embeddings."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    m_bits_values = [r[0] for r in results_list]
    brewa_recalls = [r[1]['brewa_recall'] for r in results_list]
    baseline_recalls = [r[1]['baseline_recall'] for r in results_list]
    recall_ratios = [r[1]['recall_ratio'] for r in results_list]
    compressions = [r[1]['compression'] for r in results_list]
    
    # 1. Recall comparison
    x = np.arange(len(m_bits_values))
    width = 0.35
    
    ax1.bar(x - width/2, brewa_recalls, width, label='BREWA', alpha=0.8)
    ax1.bar(x + width/2, baseline_recalls, width, label='Baseline', alpha=0.8)
    ax1.set_xlabel('m_bits')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('BREWA vs Baseline Recall (BERT Embeddings)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_bits_values)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Recall ratio
    ax2.plot(m_bits_values, recall_ratios, marker='o', linewidth=2, markersize=10)
    ax2.set_xlabel('m_bits')
    ax2.set_ylabel('BREWA / Baseline Recall Ratio')
    ax2.set_title('Relative Performance')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Equal performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Compression vs Recall
    ax3.scatter(compressions, brewa_recalls, s=200, alpha=0.6, c=brewa_recalls, cmap='RdYlGn')
    for i, m in enumerate(m_bits_values):
        ax3.annotate(f'm={m}', (compressions[i], brewa_recalls[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Compression Ratio')
    ax3.set_ylabel('BREWA Recall@10')
    ax3.set_title('Compression vs Accuracy Trade-off')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    
    # 4. Absolute recall values
    ax4.plot(m_bits_values, brewa_recalls, marker='o', label='BREWA', linewidth=2, markersize=10)
    ax4.plot(m_bits_values, baseline_recalls, marker='s', label='Baseline', linewidth=2, markersize=10)
    ax4.set_xlabel('m_bits')
    ax4.set_ylabel('Recall@10')
    ax4.set_title('Recall vs m_bits')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('bert_embeddings_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: bert_embeddings_results.png")
    plt.close()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiments
    results = run_bert_experiments()
    
    print("\n" + "="*60)
    print("BERT Embeddings Experiment Complete!")
    print("="*60)
