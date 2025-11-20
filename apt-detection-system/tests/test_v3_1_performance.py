#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v3_1_performance.py
--------------------------------------------------------
Test V3.1 optimizations on realistic log dataset

Compares:
1. V3.0 (non-batched): 153 logs/sec
2. V3.1 (batched + fast path): ???
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'advanced'))

from adalog_production_v3_1 import FastADALogDetectorV31


def test_v31_on_realistic_logs(log_file: str, num_logs: int = 1000, batch_size: int = 100):
    """Test V3.1 on realistic logs with batching"""
    print(f"\n{'='*70}")
    print("V3.1 PERFORMANCE TEST ON REALISTIC LOGS")
    print(f"{'='*70}\n")

    detector = FastADALogDetectorV31("models/adalog_bloom_signatures.json")

    print(f"Processing {num_logs:,} logs in batches of {batch_size}...\n")

    all_detections = []
    fast_path_count = 0
    total_processed = 0

    start = time.time()

    # Read logs in batches
    logs_batch = []
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if total_processed >= num_logs:
                break

            log = json.loads(line)
            text = log['text']
            logs_batch.append((i, text))

            # Process batch when full
            if len(logs_batch) >= batch_size:
                texts = [text for _, text in logs_batch]
                results = detector.detect_batch(texts)

                for (line_num, _), (detected, scores) in zip(logs_batch, results):
                    if detected:
                        all_detections.append({
                            'line': line_num,
                            'families': list(detected)
                        })
                    else:
                        fast_path_count += 1

                total_processed += len(logs_batch)
                logs_batch = []

                # Progress
                if total_processed % 1000 == 0:
                    elapsed = time.time() - start
                    rate = total_processed / elapsed
                    print(f"  ✓ Processed {total_processed:,} logs ({rate:.0f} logs/sec)")

    # Process remaining logs
    if logs_batch:
        texts = [text for _, text in logs_batch]
        results = detector.detect_batch(texts)

        for (line_num, _), (detected, scores) in zip(logs_batch, results):
            if detected:
                all_detections.append({
                    'line': line_num,
                    'families': list(detected)
                })
            else:
                fast_path_count += 1

        total_processed += len(logs_batch)

    elapsed = time.time() - start

    print(f"\n✅ Complete!")
    print(f"   Total logs: {total_processed:,}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {total_processed/elapsed:.0f} logs/sec")
    print(f"   Detections: {len(all_detections)}")
    print(f"   Fast path (skipped): {fast_path_count} ({fast_path_count/total_processed*100:.1f}%)")

    # Break down by family
    family_counts = defaultdict(int)
    for det in all_detections:
        for fam in det['families']:
            family_counts[fam] += 1

    print(f"\n📊 Detections by Family:")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"   {family}: {count}")

    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")

    v30_throughput = 153  # From previous test
    v31_throughput = total_processed / elapsed

    speedup = v31_throughput / v30_throughput

    print(f"   V3.0 (non-batched):     {v30_throughput:.0f} logs/sec")
    print(f"   V3.1 (batched + fast):  {v31_throughput:.0f} logs/sec")
    print(f"   Speedup:                {speedup:.1f}x ✅")

    if speedup >= 5:
        print(f"\n✅ Excellent! V3.1 provides {speedup:.1f}x speedup")
        print(f"   Ready for production deployment")
    elif speedup >= 2:
        print(f"\n✅ Good! V3.1 provides {speedup:.1f}x speedup")
        print(f"   Further optimizations possible with larger batches")
    else:
        print(f"\n⚠️  Modest speedup of {speedup:.1f}x")
        print(f"   Consider:")
        print(f"   - Larger batch sizes (current: {batch_size})")
        print(f"   - More aggressive fast path filtering")
        print(f"   - Embedding caching for repeated logs")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test V3.1 performance')
    parser.add_argument('--input', '-i',
                       default='data/test_logs/realistic_raw_logs.jsonl',
                       help='Input log file')
    parser.add_argument('--num-logs', '-n', type=int, default=1000,
                       help='Number of logs to test')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='Batch size for processing')

    args = parser.parse_args()

    test_v31_on_realistic_logs(args.input, args.num_logs, args.batch_size)


if __name__ == '__main__':
    main()
