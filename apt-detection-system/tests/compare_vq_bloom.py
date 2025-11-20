#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_vq_bloom.py
--------------------------------------------------------
Compare ADALog+VQ+Bloom approach vs direct semantic detection

This answers the key question:
Does VQ quantization + Bloom filters speed up detection while maintaining accuracy?

Comparison:
1. Direct semantic: Full similarity computation for each family (slow but accurate)
2. VQ+Bloom: Quantize to code → Bloom lookup (fast, slightly less accurate)
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'advanced'))

from adalog_semantic_encoder import ADALogSemanticEncoder
from adalog_production import FastADALogDetector


def test_direct_semantic(log_file: str, num_logs: int = 1000):
    """Test direct semantic similarity (no VQ/Bloom)"""
    print(f"\n{'='*70}")
    print("DIRECT SEMANTIC DETECTION (baseline)")
    print(f"{'='*70}\n")

    encoder = ADALogSemanticEncoder()

    detections = []
    start = time.time()

    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_logs:
                break

            log = json.loads(line)
            text = log['text']

            # Direct semantic classification
            emb, families, patterns, probs = encoder.classify(text)

            # Count detections (prob > 0.4 = medium confidence)
            detected = [fam for j, fam in enumerate(families) if probs[j] > 0.4]

            if detected:
                detections.append({
                    'line': i,
                    'text': text,
                    'families': detected
                })

    elapsed = time.time() - start

    print(f"✅ Complete: {num_logs:,} logs in {elapsed:.2f}s ({num_logs/elapsed:.0f} logs/sec)")
    print(f"   Detections: {len(detections)}")

    return detections, elapsed


def test_vq_bloom(log_file: str, num_logs: int = 1000):
    """Test VQ + Bloom filter approach"""
    print(f"\n{'='*70}")
    print("VQ + BLOOM FILTER DETECTION")
    print(f"{'='*70}\n")

    detector = FastADALogDetector("models/adalog_bloom_signatures.json", quick_embed=False)

    detections = []
    start = time.time()

    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_logs:
                break

            log = json.loads(line)
            text = log['text']

            # VQ + Bloom detection
            detected_families, scores = detector.detect(text)

            if detected_families:
                detections.append({
                    'line': i,
                    'text': text,
                    'families': list(detected_families)
                })

    elapsed = time.time() - start

    print(f"✅ Complete: {num_logs:,} logs in {elapsed:.2f}s ({num_logs/elapsed:.0f} logs/sec)")
    print(f"   Detections: {len(detections)}")

    return detections, elapsed


def compare_results(direct_results, vq_results):
    """Compare detection results"""
    direct_dets, direct_time = direct_results
    vq_dets, vq_time = vq_results

    print(f"\n{'='*70}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*70}\n")

    # Performance
    print("⚡ Performance:")
    print(f"   Direct semantic: {direct_time:.2f}s")
    print(f"   VQ + Bloom:      {vq_time:.2f}s ({vq_time/direct_time:.2f}x {'slower' if vq_time > direct_time else 'faster'})")

    speedup = direct_time / vq_time if vq_time < direct_time else -(vq_time / direct_time)
    if speedup > 1:
        print(f"   → VQ+Bloom is {speedup:.1f}x faster! ✅")
    else:
        print(f"   → VQ+Bloom is {-speedup:.1f}x slower (needs optimization)")

    # Detection comparison
    print(f"\n📊 Detection Counts:")
    print(f"   Direct semantic: {len(direct_dets)}")
    print(f"   VQ + Bloom:      {len(vq_dets)}")

    # Overlap analysis
    direct_lines = set(d['line'] for d in direct_dets)
    vq_lines = set(d['line'] for d in vq_dets)

    overlap = direct_lines & vq_lines
    only_direct = direct_lines - vq_lines
    only_vq = vq_lines - direct_lines

    print(f"\n💡 Detection Overlap:")
    print(f"   Both methods:    {len(overlap)}")
    print(f"   Only direct:     {len(only_direct)}")
    print(f"   Only VQ+Bloom:   {len(only_vq)}")

    if len(direct_lines) > 0:
        recall = len(overlap) / len(direct_lines) * 100
        print(f"   Recall:          {recall:.1f}% (VQ catches {recall:.1f}% of direct detections)")

    # Show examples of differences
    if only_direct:
        print(f"\n💡 Examples missed by VQ+Bloom (first 3):")
        for i, det in enumerate([d for d in direct_dets if d['line'] in only_direct][:3], 1):
            print(f"   {i}. {det['text'][:60]}...")
            print(f"      Families: {', '.join(det['families'])}")

    if only_vq:
        print(f"\n💡 Examples only detected by VQ+Bloom (first 3):")
        for i, det in enumerate([d for d in vq_dets if d['line'] in only_vq][:3], 1):
            print(f"   {i}. {det['text'][:60]}...")
            print(f"      Families: {', '.join(det['families'])}")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")

    if speedup > 1.5:
        print(f"✅ VQ+Bloom provides {speedup:.1f}x speedup!")
        print(f"   Recommended for production use")
    elif abs(speedup) < 1.2:
        print(f"⚠️  VQ+Bloom and direct semantic have similar speed")
        print(f"   VQ quantization overhead ≈ Bloom filter savings")
        print(f"   Consider:")
        print(f"   - Larger VQ codebook (more compression)")
        print(f"   - More training examples (better quantization)")
        print(f"   - Batch processing for amortized costs")
    else:
        print(f"❌ VQ+Bloom is slower than direct semantic")
        print(f"   This suggests quantization overhead > Bloom savings")
        print(f"   Root cause: Still computing full semantic embeddings!")
        print(f"\n   💡 The KEY insight:")
        print(f"      We're still using the expensive transformer for embeddings")
        print(f"      VQ+Bloom only optimizes the MATCHING step, not embedding")
        print(f"\n   To get real speedup, we need to:")
        print(f"   1. Cache embeddings for repeated logs")
        print(f"   2. Use distilled/smaller model for embeddings")
        print(f"   3. Batch embed multiple logs at once")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Compare VQ+Bloom vs direct semantic')
    parser.add_argument('--input', '-i',
                       default='data/test_logs/realistic_raw_logs.jsonl',
                       help='Input log file')
    parser.add_argument('--num-logs', '-n', type=int, default=1000,
                       help='Number of logs to test (default: 1000)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("VQ+BLOOM PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    print(f"Logs: {args.num_logs:,}")
    print(f"{'='*70}")

    # Test both approaches
    direct_results = test_direct_semantic(args.input, args.num_logs)
    vq_results = test_vq_bloom(args.input, args.num_logs)

    # Compare
    compare_results(direct_results, vq_results)


if __name__ == '__main__':
    main()
