#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_encoders.py
--------------------------------------------------------
Compare signature vs semantic vs hybrid encoders

Shows the difference between:
1. Signature mode: Fast regex matching
2. Semantic mode: True NLP-based understanding
3. Hybrid mode: Best of both worlds
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from advanced.adalog_bloom_temporal_v3 import CompositeEngine


def test_encoder(mode: str, log_file: str, num_logs: int = 1000):
    """
    Test a specific encoder mode

    Args:
        mode: 'signature', 'semantic', or 'hybrid'
        log_file: Path to log file
        num_logs: Number of logs to process

    Returns:
        (detections, processing_time, detections_by_family)
    """
    families = [
        "CVE-2024-6387", "CVE-2021-44228", "CVE-2017-0144",
        "reconnaissance", "exploitation", "privilege_escalation",
        "persistence", "lateral_movement", "exfiltration"
    ]

    print(f"\n{'='*70}")
    print(f"TESTING: {mode.upper()} MODE")
    print(f"{'='*70}\n")

    engine = CompositeEngine(families, fast_mode=True, encoder_mode=mode)

    detections = []
    detections_by_family = defaultdict(list)

    start = time.time()
    processed = 0

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if processed >= num_logs:
                break

            log = json.loads(line)

            # Normalize
            log['id'] = f"log_{line_num}"
            log['entity'] = log.get('host', 'unknown')

            if 'timestamp' in log:
                ts_obj = datetime.fromisoformat(log['timestamp'])
                log['ts'] = int(ts_obj.timestamp())
            else:
                log['ts'] = int(time.time())

            log['text'] = log.get('text', str(log))
            log['cred_hash'] = log.get('user', '')
            log['asn'] = ''
            log['src_ip'] = log.get('source_ip', '')

            # Classify
            emb, fam_conf, patterns, probs = engine.adalog.classify(log['text'])

            detected_families = set()
            for tier in ['high', 'medium']:
                for fam, prob in fam_conf[tier]:
                    detected_families.add(fam)

            if detected_families:
                detection = {
                    'line': line_num,
                    'text': log['text'],
                    'families': list(detected_families),
                    'max_prob': max(probs)
                }
                detections.append(detection)

                for fam in detected_families:
                    detections_by_family[fam].append(detection)

            # Ingest
            engine.ingest(log)
            processed += 1

    elapsed = time.time() - start

    print(f"✅ Complete: {processed:,} logs in {elapsed:.2f}s ({processed/elapsed:.0f} logs/sec)")
    print(f"   Detections: {len(detections)} ({len(detections)/processed*100:.2f}%)")

    return detections, elapsed, detections_by_family


def compare_detections(signature_dets, semantic_dets, hybrid_dets):
    """Compare detection results across encoders"""
    print(f"\n{'='*70}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*70}\n")

    # Detection rates
    print("📊 Detection Rates:")
    print(f"   Signature: {len(signature_dets[0])} detections")
    print(f"   Semantic:  {len(semantic_dets[0])} detections")
    print(f"   Hybrid:    {len(hybrid_dets[0])} detections")

    # Performance
    print(f"\n⚡ Performance:")
    print(f"   Signature: {signature_dets[1]:.2f}s")
    print(f"   Semantic:  {semantic_dets[1]:.2f}s  ({semantic_dets[1]/signature_dets[1]:.1f}x slower)")
    print(f"   Hybrid:    {hybrid_dets[1]:.2f}s  ({hybrid_dets[1]/signature_dets[1]:.1f}x slower)")

    # Family breakdown
    print(f"\n📊 Detections by Family:")

    all_families = set()
    all_families.update(signature_dets[2].keys())
    all_families.update(semantic_dets[2].keys())
    all_families.update(hybrid_dets[2].keys())

    for family in sorted(all_families):
        sig_count = len(signature_dets[2].get(family, []))
        sem_count = len(semantic_dets[2].get(family, []))
        hyb_count = len(hybrid_dets[2].get(family, []))

        print(f"\n   {family}:")
        print(f"      Signature: {sig_count:4d}")
        print(f"      Semantic:  {sem_count:4d}")
        print(f"      Hybrid:    {hyb_count:4d}")

    # Unique detections
    print(f"\n💡 Unique Detections:")

    sig_lines = set(d['line'] for d in signature_dets[0])
    sem_lines = set(d['line'] for d in semantic_dets[0])
    hyb_lines = set(d['line'] for d in hybrid_dets[0])

    only_sig = sig_lines - sem_lines - hyb_lines
    only_sem = sem_lines - sig_lines - hyb_lines
    only_hyb = hyb_lines - sig_lines - sem_lines

    print(f"   Only Signature: {len(only_sig)}")
    print(f"   Only Semantic:  {len(only_sem)}")
    print(f"   Only Hybrid:    {len(only_hyb)}")
    print(f"   All three:      {len(sig_lines & sem_lines & hyb_lines)}")

    # Show examples of semantic-only detections
    if only_sem:
        print(f"\n💡 Example semantic detections (missed by signature):")
        sem_only_dets = [d for d in semantic_dets[0] if d['line'] in only_sem]
        for i, det in enumerate(sem_only_dets[:3], 1):
            print(f"   {i}. {det['text'][:70]}...")
            print(f"      Detected as: {', '.join(det['families'])}")

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}\n")

    print("🎯 Use signature mode when:")
    print("   • Speed is critical (3,000+ logs/sec)")
    print("   • Known CVEs with clear indicators")
    print("   • Limited compute resources")

    print("\n🧠 Use semantic mode when:")
    print("   • Detecting attack variants")
    print("   • Zero-day detection needed")
    print("   • Log formats vary widely")

    print("\n⚡ Use hybrid mode when:")
    print("   • Balance speed and accuracy")
    print("   • Catch both known patterns and variants")
    print("   • Production deployment (recommended)")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Compare encoder modes')
    parser.add_argument('--input', '-i',
                       default='data/test_logs/realistic_raw_logs.jsonl',
                       help='Input log file')
    parser.add_argument('--num-logs', '-n', type=int, default=1000,
                       help='Number of logs to test (default: 1000)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("ENCODER COMPARISON TEST")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    print(f"Logs: {args.num_logs:,}")
    print(f"{'='*70}")

    # Test each mode
    signature_results = test_encoder('signature', args.input, args.num_logs)
    semantic_results = test_encoder('semantic', args.input, args.num_logs)
    hybrid_results = test_encoder('hybrid', args.input, args.num_logs)

    # Compare results
    compare_detections(signature_results, semantic_results, hybrid_results)


if __name__ == '__main__':
    main()
