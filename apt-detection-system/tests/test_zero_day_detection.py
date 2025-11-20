#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_zero_day_detection.py
--------------------------------------------------------
Test V3 zero-day detection WITHOUT signature patterns

This tests if the system can detect attacks as anomalies through:
1. Tier 1b: Dynamic Family Discovery (unsupervised clustering)
2. Tier 2b: Temporal correlation (unusual patterns over time)
3. Tier 3: Graph-based campaign detection

Uses ORIGINAL MockADALogEncoder (random probabilities, no signatures)
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Temporarily replace MockADALogEncoder with original version
import advanced.adalog_bloom_temporal_v3 as v3_module
import numpy as np
import hashlib

# Save original
OriginalEncoder = v3_module.MockADALogEncoder

# Create version without signatures
class RandomMockEncoder:
    """Original mock encoder with random probabilities (no signatures)"""

    def __init__(self, dim=64, families=None):
        self.dim = dim
        self.families = families or [
            "process_crash", "privilege_escalation", "persistence",
            "outbound_c2", "recon", "file_mod"
        ]

    def embed(self, text):
        h = int(hashlib.sha256(text.encode('utf8')).hexdigest(), 16)
        rng = np.random.RandomState(h & 0xffffffff)
        return rng.normal(size=(self.dim,)).astype(np.float32)

    def classify(self, text):
        emb = self.embed(text)
        base = abs(np.tanh(np.sin(np.sum(emb[:8]))))
        probs = np.clip(np.random.normal(base, 0.15, len(self.families)), 0, 1)
        patterns = {
            f: f"log_sem_{f}_{round(probs[i], 2)}"
            for i, f in enumerate(self.families)
            if probs[i] > 0.2
        }
        return emb, self.families, patterns, probs


# Monkey patch
v3_module.MockADALogEncoder = RandomMockEncoder

from advanced.adalog_bloom_temporal_v3 import CompositeEngine


class ZeroDayDetectionTest:
    """Test zero-day detection with unsupervised learning"""

    def __init__(self, log_file: str, fast_mode: bool = False):
        self.log_file = log_file
        self.fast_mode = fast_mode

        # Use generic families (no CVE names)
        self.families = [
            "process_crash", "privilege_escalation", "persistence",
            "outbound_c2", "recon", "file_mod"
        ]

        print(f"\n{'='*70}")
        print("ZERO-DAY DETECTION TEST")
        print(f"{'='*70}")
        print(f"Encoder: Random probabilities (NO signature patterns)")
        print(f"Fast mode: {fast_mode}")
        if fast_mode:
            print("  ⚠️  Tier 1b (Dynamic Discovery): DISABLED")
            print("  ⚠️  Tier 3 (Graph + Scoring): DISABLED")
            print("  Detection relies on: Random semantic classifier only")
        else:
            print("  ✅ Tier 1b (Dynamic Discovery): ENABLED")
            print("  ✅ Tier 3 (Graph + Scoring): ENABLED")
            print("  Detection relies on: Unsupervised clustering + graph correlation")
        print(f"{'='*70}\n")

        self.engine = CompositeEngine(self.families, fast_mode=fast_mode)
        self.detected_events = []

    def ingest_logs(self):
        """Ingest logs and track detections"""
        start = time.time()
        processed = 0
        detected = 0

        detections_by_family = defaultdict(list)

        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
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

                # Check detections
                emb, fam_conf, patterns, probs = self.engine.adalog.classify(log['text'])

                detected_families = set()
                for tier in ['high', 'medium']:
                    for fam, prob in fam_conf[tier]:
                        detected_families.add(fam)

                if detected_families:
                    detected += 1
                    detection = {
                        'line': line_num,
                        'text': log['text'],
                        'families': list(detected_families),
                        'max_prob': max(probs)
                    }
                    self.detected_events.append(detection)

                    for fam in detected_families:
                        detections_by_family[fam].append(detection)

                # Ingest
                self.engine.ingest(log)

                processed += 1

                if processed % 10000 == 0:
                    elapsed = time.time() - start
                    rate = processed / elapsed
                    print(f"  ✓ Processed {processed:,} logs ({rate:.0f} logs/sec) | Detected: {detected}")

        elapsed = time.time() - start
        print(f"\n✅ Complete: {processed:,} logs in {elapsed:.2f}s ({processed/elapsed:.0f} logs/sec)")
        print(f"   Detections (random): {detected} ({detected/processed*100:.2f}%)")

        return detected, detections_by_family

    def analyze(self, detected, detections_by_family):
        """Analyze results"""
        print(f"\n{'='*70}")
        print("ANALYSIS")
        print(f"{'='*70}")

        if detected > 0:
            print(f"\n⚠️  Random encoder detected {detected} events")
            print(f"   (These are FALSE POSITIVES from random probabilities)")

            print(f"\n   Top families (random):")
            for fam in sorted(detections_by_family.keys(),
                            key=lambda x: len(detections_by_family[x]),
                            reverse=True)[:5]:
                count = len(detections_by_family[fam])
                print(f"      • {fam}: {count} events")

        # Check if Dynamic Discovery found anything
        if self.engine.disc is not None:
            discovered = len(self.engine.disc.exemplars)
            print(f"\n   Tier 1b (Dynamic Discovery):")
            if discovered > 0:
                print(f"      ✅ Discovered {discovered} threat families")
                print(f"      (These are unsupervised clusters of anomalous patterns)")
            else:
                print(f"      ⚠️  No families discovered (need more data or tuning)")
        else:
            print(f"\n   Tier 1b: DISABLED (fast_mode=True)")

        # Check graph
        if self.engine.tempo is not None:
            campaigns = self.engine.tempo.campaigns()
            print(f"\n   Tier 3 (Graph-based campaigns):")
            if len(campaigns) > 0:
                print(f"      ✅ Detected {len(campaigns)} campaign components")
            else:
                print(f"      ⚠️  No campaigns detected")
        else:
            print(f"\n   Tier 3: DISABLED (fast_mode=True)")

        print(f"\n{'='*70}")
        print("CONCLUSION")
        print(f"{'='*70}")

        if self.fast_mode:
            print(f"\n❌ With fast_mode=True:")
            print(f"   • Tier 1b (Discovery): DISABLED")
            print(f"   • Tier 3 (Graph): DISABLED")
            print(f"   • Only random semantic classifier active")
            print(f"   • Cannot detect zero-day attacks reliably")
            print(f"\n💡 Zero-day detection requires:")
            print(f"   1. fast_mode=False (enable ML/Graph)")
            print(f"   2. Sufficient training data for clustering")
            print(f"   3. Temporal/actor correlation across events")
        else:
            print(f"\n⚠️  With fast_mode=False:")
            print(f"   • Tier 1b + Tier 3 enabled")
            print(f"   • Can detect anomalies through unsupervised learning")
            print(f"   • But much slower ({self.engine.ingest.__doc__})")
            print(f"\n🎯 For production:")
            print(f"   • Known CVEs: Use signature patterns (fast + accurate)")
            print(f"   • Zero-day: Use full V3 with ML/Graph (slow but detects unknowns)")

        print(f"{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test zero-day detection')
    parser.add_argument('--input', '-i',
                       default='data/test_logs/realistic_raw_logs.jsonl',
                       help='Input log file')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast mode (disables ML/Graph)')
    parser.add_argument('--slow', action='store_true',
                       help='Use full mode (enables ML/Graph for zero-day)')

    args = parser.parse_args()

    fast_mode = args.fast or not args.slow

    test = ZeroDayDetectionTest(args.input, fast_mode=fast_mode)
    detected, by_family = test.ingest_logs()
    test.analyze(detected, by_family)


if __name__ == '__main__':
    main()
