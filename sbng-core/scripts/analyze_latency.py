#!/usr/bin/env python3
"""
Analyze latency breakdown from evaluation results.
"""

import json
import statistics

# Load results
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

queries = data['individual_queries']

print("=" * 80)
print("LATENCY BREAKDOWN ANALYSIS")
print("=" * 80)

# Extract latencies
no_rerank_latencies = [q['latency_no_rerank'] for q in queries]
rerank_latencies = [q['latency_rerank'] for q in queries]

print("\n📊 PER-QUERY LATENCY (ms)")
print("-" * 80)
print(f"{'Query #':<10} {'No Rerank':<15} {'With Rerank':<15} {'Overhead':<15}")
print("-" * 80)

for i, q in enumerate(queries, 1):
    no_rerank = q['latency_no_rerank']
    rerank = q['latency_rerank']
    overhead = rerank - no_rerank
    print(f"{i:<10} {no_rerank:>13.2f} {rerank:>14.2f} {overhead:>14.2f}")

print("-" * 80)

# Statistics
print("\n📈 SUMMARY STATISTICS (ms)")
print("-" * 80)

print("\nNo Rerank:")
print(f"  Min:     {min(no_rerank_latencies):>10.2f}")
print(f"  Max:     {max(no_rerank_latencies):>10.2f}")
print(f"  Mean:    {statistics.mean(no_rerank_latencies):>10.2f}")
print(f"  Median:  {statistics.median(no_rerank_latencies):>10.2f}")
print(f"  StdDev:  {statistics.stdev(no_rerank_latencies):>10.2f}")

print("\nWith Rerank:")
print(f"  Min:     {min(rerank_latencies):>10.2f}")
print(f"  Max:     {max(rerank_latencies):>10.2f}")
print(f"  Mean:    {statistics.mean(rerank_latencies):>10.2f}")
print(f"  Median:  {statistics.median(rerank_latencies):>10.2f}")
print(f"  StdDev:  {statistics.stdev(rerank_latencies):>10.2f}")

# Pure reranking overhead (excluding first query)
if len(rerank_latencies) > 1:
    rerank_without_first = rerank_latencies[1:]
    no_rerank_without_first = no_rerank_latencies[1:]

    print("\n🔥 EXCLUDING FIRST QUERY (removing potential warmup)")
    print("-" * 80)
    print(f"  Queries analyzed: {len(rerank_without_first)}")
    print(f"  Mean no-rerank:   {statistics.mean(no_rerank_without_first):>10.2f} ms")
    print(f"  Mean with-rerank: {statistics.mean(rerank_without_first):>10.2f} ms")
    print(f"  Mean overhead:    {statistics.mean(rerank_without_first) - statistics.mean(no_rerank_without_first):>10.2f} ms")
    print(f"  Median overhead:  {statistics.median([r - n for r, n in zip(rerank_without_first, no_rerank_without_first)]):>10.2f} ms")

# Check if first query is an outlier
first_rerank = rerank_latencies[0]
rest_mean = statistics.mean(rerank_latencies[1:])
print(f"\n⏱️  FIRST QUERY ANALYSIS")
print("-" * 80)
print(f"  First query rerank latency:  {first_rerank:>10.2f} ms")
print(f"  Mean of remaining queries:   {rest_mean:>10.2f} ms")
print(f"  Difference:                  {first_rerank - rest_mean:>10.2f} ms")

if first_rerank > rest_mean * 1.2:
    print(f"  ⚠️  First query is {((first_rerank/rest_mean - 1) * 100):.1f}% slower (likely model warmup)")
else:
    print(f"  ✓ First query latency is consistent with others")

# Pure reranking time estimation
overhead_times = [r - n for r, n in zip(rerank_latencies, no_rerank_latencies)]
print(f"\n⚙️  PURE RERANKING OVERHEAD")
print("-" * 80)
print(f"  Mean overhead:   {statistics.mean(overhead_times):>10.2f} ms")
print(f"  Median overhead: {statistics.median(overhead_times):>10.2f} ms")
print(f"  Min overhead:    {min(overhead_times):>10.2f} ms")
print(f"  Max overhead:    {max(overhead_times):>10.2f} ms")
print(f"  StdDev:          {statistics.stdev(overhead_times):>10.2f} ms")

print("\n" + "=" * 80)
