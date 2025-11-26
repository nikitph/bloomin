#!/usr/bin/env python3
"""
Comprehensive evaluation of the re-ranker on wikipedia_10k.jsonl dataset.

This script measures:
1. Recall@K (K=1,3,5,10): How many relevant documents are retrieved
2. MRR (Mean Reciprocal Rank): Average position of first relevant result
3. NDCG@K: Normalized Discounted Cumulative Gain
4. Precision@K: Precision at top K
5. Latency: Query execution time
6. Re-ranking impact: How often re-ranking changes the order
"""

import json
import requests
import time
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import statistics

# Server configuration
SBNG_URL = "http://localhost:3001/query"

# Test queries - diverse set covering different topics
TEST_QUERIES = [
    # Science & Technology
    ("quantum mechanics physics", ["quantum", "physics", "mechanics", "particle"]),
    ("artificial intelligence machine learning", ["artificial intelligence", "machine learning", "neural", "algorithm"]),
    ("climate change global warming", ["climate", "warming", "greenhouse", "carbon"]),
    ("DNA genetics heredity", ["DNA", "gene", "genetic", "heredity"]),
    ("computer programming software", ["computer", "programming", "software", "code"]),

    # History & Geography
    ("world war two hitler", ["world war", "hitler", "nazi", "1939"]),
    ("roman empire caesar", ["roman", "empire", "caesar", "rome"]),
    ("egyptian pyramids pharaoh", ["egypt", "pyramid", "pharaoh", "ancient"]),
    ("christopher columbus america", ["columbus", "america", "voyage", "discovery"]),

    # Arts & Culture
    ("leonardo da vinci painting", ["leonardo", "vinci", "painting", "renaissance"]),
    ("beethoven symphony music", ["beethoven", "symphony", "music", "composer"]),
    ("shakespeare literature play", ["shakespeare", "play", "theatre", "literature"]),

    # Nature & Biology
    ("photosynthesis plant chlorophyll", ["photosynthesis", "plant", "chlorophyll", "sunlight"]),
    ("evolution darwin species", ["evolution", "darwin", "species", "natural selection"]),
    ("ocean marine ecosystem", ["ocean", "marine", "sea", "ecosystem"]),

    # Sports & Entertainment
    ("olympic games athletics", ["olympic", "games", "sport", "athlete"]),
    ("cinema film director", ["film", "cinema", "movie", "director"]),

    # Social Sciences
    ("democracy government voting", ["democracy", "government", "voting", "election"]),
    ("economics market capitalism", ["economics", "market", "capitalism", "trade"]),
    ("psychology human behavior", ["psychology", "behavior", "mind", "cognitive"]),
]

def load_corpus(path: str) -> Dict[str, str]:
    """Load the wikipedia corpus into memory."""
    corpus = {}
    with open(path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['id']] = doc['text']
    return corpus

def is_relevant(doc_text: str, keywords: List[str]) -> bool:
    """
    Check if a document is relevant based on keyword presence.
    A document is relevant if it contains at least 2 of the keywords.
    """
    doc_lower = doc_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in doc_lower)
    return matches >= 2

def query_sbng(query: str, k: int, rerank: bool) -> Tuple[List[Dict], float]:
    """Query SBNG and return results with latency."""
    start = time.time()
    try:
        response = requests.post(
            SBNG_URL,
            json={"q": query, "k": k, "rerank": rerank},
            timeout=30
        )
        latency = (time.time() - start) * 1000  # ms

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return [], latency

        results = response.json().get('results', [])
        return results, latency
    except Exception as e:
        latency = (time.time() - start) * 1000
        print(f"Exception during query: {e}")
        return [], latency

def calculate_recall_at_k(results: List[Dict], relevant_ids: set, k: int) -> float:
    """Calculate Recall@K."""
    if not relevant_ids:
        return 0.0
    retrieved_relevant = sum(1 for r in results[:k] if r['doc_id'] in relevant_ids)
    return retrieved_relevant / len(relevant_ids)

def calculate_precision_at_k(results: List[Dict], relevant_ids: set, k: int) -> float:
    """Calculate Precision@K."""
    if not results[:k]:
        return 0.0
    retrieved_relevant = sum(1 for r in results[:k] if r['doc_id'] in relevant_ids)
    return retrieved_relevant / k

def calculate_mrr(results: List[Dict], relevant_ids: set) -> float:
    """Calculate Mean Reciprocal Rank."""
    for i, result in enumerate(results, 1):
        if result['doc_id'] in relevant_ids:
            return 1.0 / i
    return 0.0

def calculate_ndcg_at_k(results: List[Dict], relevant_ids: set, k: int) -> float:
    """Calculate NDCG@K."""
    import math

    # DCG
    dcg = 0.0
    for i, result in enumerate(results[:k], 1):
        rel = 1 if result['doc_id'] in relevant_ids else 0
        dcg += rel / math.log2(i + 1)

    # IDCG (ideal DCG)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_query(query: str, keywords: List[str], corpus: Dict[str, str], k_values: List[int]) -> Dict:
    """Evaluate a single query with and without re-ranking."""

    # Find relevant documents
    relevant_ids = {doc_id for doc_id, text in corpus.items() if is_relevant(text, keywords)}

    if not relevant_ids:
        print(f"  Warning: No relevant documents found for query '{query}'")
        return None

    # Query without re-ranking
    results_no_rerank, latency_no = query_sbng(query, max(k_values), False)

    # Query with re-ranking
    results_rerank, latency_rerank = query_sbng(query, max(k_values), True)

    if not results_no_rerank and not results_rerank:
        print(f"  Warning: No results returned for query '{query}'")
        return None

    # Calculate metrics
    metrics = {
        'query': query,
        'num_relevant': len(relevant_ids),
        'latency_no_rerank': latency_no,
        'latency_rerank': latency_rerank,
        'no_rerank': {},
        'rerank': {},
    }

    for k in k_values:
        # Without re-ranking
        metrics['no_rerank'][f'recall@{k}'] = calculate_recall_at_k(results_no_rerank, relevant_ids, k)
        metrics['no_rerank'][f'precision@{k}'] = calculate_precision_at_k(results_no_rerank, relevant_ids, k)
        metrics['no_rerank'][f'ndcg@{k}'] = calculate_ndcg_at_k(results_no_rerank, relevant_ids, k)

        # With re-ranking
        metrics['rerank'][f'recall@{k}'] = calculate_recall_at_k(results_rerank, relevant_ids, k)
        metrics['rerank'][f'precision@{k}'] = calculate_precision_at_k(results_rerank, relevant_ids, k)
        metrics['rerank'][f'ndcg@{k}'] = calculate_ndcg_at_k(results_rerank, relevant_ids, k)

    # MRR
    metrics['no_rerank']['mrr'] = calculate_mrr(results_no_rerank, relevant_ids)
    metrics['rerank']['mrr'] = calculate_mrr(results_rerank, relevant_ids)

    # Check if order changed
    ids_no = [r['doc_id'] for r in results_no_rerank[:10]]
    ids_rerank = [r['doc_id'] for r in results_rerank[:10]]
    metrics['order_changed'] = ids_no != ids_rerank
    metrics['top1_changed'] = ids_no and ids_rerank and ids_no[0] != ids_rerank[0]

    return metrics

def aggregate_metrics(all_metrics: List[Dict], k_values: List[int]) -> Dict:
    """Aggregate metrics across all queries."""

    aggregated = {
        'no_rerank': defaultdict(list),
        'rerank': defaultdict(list),
        'latency_no_rerank': [],
        'latency_rerank': [],
        'order_changed': 0,
        'top1_changed': 0,
        'total_queries': len(all_metrics),
    }

    for m in all_metrics:
        aggregated['latency_no_rerank'].append(m['latency_no_rerank'])
        aggregated['latency_rerank'].append(m['latency_rerank'])

        if m['order_changed']:
            aggregated['order_changed'] += 1
        if m['top1_changed']:
            aggregated['top1_changed'] += 1

        for k in k_values:
            for metric_type in ['recall', 'precision', 'ndcg']:
                key = f'{metric_type}@{k}'
                aggregated['no_rerank'][key].append(m['no_rerank'][key])
                aggregated['rerank'][key].append(m['rerank'][key])

        aggregated['no_rerank']['mrr'].append(m['no_rerank']['mrr'])
        aggregated['rerank']['mrr'].append(m['rerank']['mrr'])

    # Calculate means
    result = {
        'no_rerank': {},
        'rerank': {},
        'improvements': {},
    }

    for system in ['no_rerank', 'rerank']:
        for metric, values in aggregated[system].items():
            result[system][metric] = statistics.mean(values)

    # Calculate improvements
    for k in k_values:
        for metric_type in ['recall', 'precision', 'ndcg']:
            key = f'{metric_type}@{k}'
            improvement = ((result['rerank'][key] - result['no_rerank'][key]) /
                          result['no_rerank'][key] * 100) if result['no_rerank'][key] > 0 else 0
            result['improvements'][key] = improvement

    mrr_improvement = ((result['rerank']['mrr'] - result['no_rerank']['mrr']) /
                      result['no_rerank']['mrr'] * 100) if result['no_rerank']['mrr'] > 0 else 0
    result['improvements']['mrr'] = mrr_improvement

    # Latency stats
    result['latency'] = {
        'no_rerank_mean': statistics.mean(aggregated['latency_no_rerank']),
        'no_rerank_median': statistics.median(aggregated['latency_no_rerank']),
        'rerank_mean': statistics.mean(aggregated['latency_rerank']),
        'rerank_median': statistics.median(aggregated['latency_rerank']),
        'overhead_ms': statistics.mean(aggregated['latency_rerank']) - statistics.mean(aggregated['latency_no_rerank']),
    }

    result['order_changed_pct'] = (aggregated['order_changed'] / aggregated['total_queries']) * 100
    result['top1_changed_pct'] = (aggregated['top1_changed'] / aggregated['total_queries']) * 100

    return result

def print_results(aggregated: Dict):
    """Print evaluation results in a nice format."""

    print("\n" + "="*80)
    print("RERANKER EVALUATION RESULTS")
    print("="*80)

    print("\n📊 RETRIEVAL METRICS")
    print("-" * 80)
    print(f"{'Metric':<20} {'No Rerank':<15} {'With Rerank':<15} {'Improvement':<15}")
    print("-" * 80)

    # Print metrics
    for metric in ['recall@1', 'recall@3', 'recall@5', 'recall@10']:
        no_rerank = aggregated['no_rerank'][metric]
        rerank = aggregated['rerank'][metric]
        improvement = aggregated['improvements'][metric]
        print(f"{metric:<20} {no_rerank:<15.4f} {rerank:<15.4f} {improvement:>+13.2f}%")

    print()
    for metric in ['precision@1', 'precision@3', 'precision@5', 'precision@10']:
        no_rerank = aggregated['no_rerank'][metric]
        rerank = aggregated['rerank'][metric]
        improvement = aggregated['improvements'][metric]
        print(f"{metric:<20} {no_rerank:<15.4f} {rerank:<15.4f} {improvement:>+13.2f}%")

    print()
    for metric in ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']:
        no_rerank = aggregated['no_rerank'][metric]
        rerank = aggregated['rerank'][metric]
        improvement = aggregated['improvements'][metric]
        print(f"{metric:<20} {no_rerank:<15.4f} {rerank:<15.4f} {improvement:>+13.2f}%")

    print()
    no_rerank_mrr = aggregated['no_rerank']['mrr']
    rerank_mrr = aggregated['rerank']['mrr']
    mrr_improvement = aggregated['improvements']['mrr']
    print(f"{'MRR':<20} {no_rerank_mrr:<15.4f} {rerank_mrr:<15.4f} {mrr_improvement:>+13.2f}%")

    print("\n⚡ LATENCY METRICS")
    print("-" * 80)
    lat = aggregated['latency']
    print(f"No Rerank (mean):   {lat['no_rerank_mean']:>8.2f} ms")
    print(f"No Rerank (median): {lat['no_rerank_median']:>8.2f} ms")
    print(f"With Rerank (mean):   {lat['rerank_mean']:>8.2f} ms")
    print(f"With Rerank (median): {lat['rerank_median']:>8.2f} ms")
    print(f"Rerank Overhead:      {lat['overhead_ms']:>8.2f} ms ({(lat['overhead_ms']/lat['no_rerank_mean']*100):.1f}%)")

    print("\n🔄 RANKING CHANGES")
    print("-" * 80)
    print(f"Order changed (top-10): {aggregated['order_changed_pct']:.1f}%")
    print(f"Top-1 result changed:   {aggregated['top1_changed_pct']:.1f}%")

    print("\n" + "="*80)
    print("\n💡 SUMMARY")
    print("-" * 80)

    # Overall assessment
    avg_recall_improvement = statistics.mean([
        aggregated['improvements']['recall@1'],
        aggregated['improvements']['recall@3'],
        aggregated['improvements']['recall@5'],
        aggregated['improvements']['recall@10'],
    ])

    avg_ndcg_improvement = statistics.mean([
        aggregated['improvements']['ndcg@1'],
        aggregated['improvements']['ndcg@3'],
        aggregated['improvements']['ndcg@5'],
        aggregated['improvements']['ndcg@10'],
    ])

    print(f"Average Recall Improvement:    {avg_recall_improvement:>+8.2f}%")
    print(f"Average NDCG Improvement:      {avg_ndcg_improvement:>+8.2f}%")
    print(f"MRR Improvement:               {mrr_improvement:>+8.2f}%")

    if avg_ndcg_improvement > 10:
        print("\n✅ CONCLUSION: Re-ranker provides SIGNIFICANT quality improvements!")
    elif avg_ndcg_improvement > 5:
        print("\n✅ CONCLUSION: Re-ranker provides MODERATE quality improvements.")
    elif avg_ndcg_improvement > 0:
        print("\n⚠️  CONCLUSION: Re-ranker provides MINOR quality improvements.")
    else:
        print("\n❌ CONCLUSION: Re-ranker does NOT improve quality (may hurt performance).")

    if lat['overhead_ms'] < 100:
        print("✅ Latency overhead is ACCEPTABLE (<100ms).")
    elif lat['overhead_ms'] < 500:
        print("⚠️  Latency overhead is MODERATE (100-500ms).")
    else:
        print("❌ Latency overhead is HIGH (>500ms).")

    print("="*80 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate re-ranker effectiveness')
    parser.add_argument('--corpus', default='data/wikipedia_10k.jsonl',
                       help='Path to corpus JSONL file')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 10],
                       help='K values for evaluation metrics')
    parser.add_argument('--num-queries', type=int, default=None,
                       help='Number of queries to test (default: all)')
    parser.add_argument('--save-results', type=str,
                       help='Save detailed results to JSON file')

    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} documents")

    # Select queries
    queries = TEST_QUERIES[:args.num_queries] if args.num_queries else TEST_QUERIES
    print(f"\nEvaluating on {len(queries)} queries...\n")

    # Evaluate each query
    all_metrics = []
    for i, (query, keywords) in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Evaluating: '{query}'")
        metrics = evaluate_query(query, keywords, corpus, args.k_values)
        if metrics:
            all_metrics.append(metrics)
            print(f"  ✓ Relevant docs: {metrics['num_relevant']}, "
                  f"Order changed: {metrics['order_changed']}, "
                  f"Top-1 changed: {metrics['top1_changed']}")

    if not all_metrics:
        print("\n❌ No successful queries. Check if SBNG server is running.")
        return

    print(f"\n✓ Successfully evaluated {len(all_metrics)} queries")

    # Aggregate and print results
    aggregated = aggregate_metrics(all_metrics, args.k_values)
    print_results(aggregated)

    # Save detailed results if requested
    if args.save_results:
        output = {
            'individual_queries': all_metrics,
            'aggregated': aggregated,
        }
        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Detailed results saved to: {args.save_results}")

if __name__ == '__main__':
    main()
