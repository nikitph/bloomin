#!/usr/bin/env python3
"""
Benchmark re-ranker latency with different candidate counts.
"""
import requests
import time
import statistics

SBNG_URL = "http://localhost:3001/query"

queries = [
    "electric car battery",
    "apple fruit",
    "python programming language",
    "bank river water",
    "jaguar animal speed"
]

def benchmark_latency(k_values=[5, 10, 20, 50]):
    print(f"=== Re-ranker Latency Benchmark ===\n")
    print(f"Testing {len(queries)} queries with different candidate counts (k)\n")
    
    results = {}
    
    for k in k_values:
        print(f"Testing k={k}...")
        latencies = []
        
        for q in queries:
            start = time.time()
            try:
                res = requests.post(SBNG_URL, json={"q": q, "k": k, "rerank": True}, timeout=5)
                if res.status_code == 200:
                    latency_ms = (time.time() - start) * 1000
                    latencies.append(latency_ms)
                else:
                    print(f"  Error {res.status_code} for query: {q}")
            except Exception as e:
                print(f"  Exception for query '{q}': {e}")
        
        if latencies:
            avg = statistics.mean(latencies)
            median = statistics.median(latencies)
            p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
            
            results[k] = {
                'avg': avg,
                'median': median,
                'p95': p95,
                'samples': len(latencies)
            }
            
            print(f"  k={k}: Avg={avg:.1f}ms, Median={median:.1f}ms, P95={p95:.1f}ms ({len(latencies)} samples)\n")
        else:
            print(f"  k={k}: No successful queries\n")
    
    # Summary
    print("\n=== Summary ===")
    print(f"{'k':<10} {'Avg (ms)':<12} {'Median (ms)':<14} {'P95 (ms)':<12}")
    print("-" * 50)
    for k, stats in results.items():
        print(f"{k:<10} {stats['avg']:<12.1f} {stats['median']:<14.1f} {stats['p95']:<12.1f}")
    
    # Find optimal k
    if results:
        optimal_k = min(results.keys(), key=lambda k: results[k]['avg'])
        print(f"\nOptimal k (lowest avg latency): {optimal_k} ({results[optimal_k]['avg']:.1f}ms)")

if __name__ == "__main__":
    benchmark_latency()
