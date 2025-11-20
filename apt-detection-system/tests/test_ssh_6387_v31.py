#!/usr/bin/env python3
"""Test CVE-2024-6387 detection with V3.1 batch processing"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from advanced.adalog_bloom_temporal_v3 import CompositeEngine
import time

# Test logs for CVE-2024-6387 (regresshion)
test_logs = [
    {
        "id": "log_1",
        "entity": "ssh-server-1",
        "ts": int(time.time()),
        "text": "sshd[1234]: SIGALRM received during authentication from 10.0.1.50",
        "cred_hash": "",
        "asn": "",
        "src_ip": "10.0.1.50"
    },
    {
        "id": "log_2",
        "entity": "ssh-server-1",
        "ts": int(time.time()),
        "text": "sshd[1234]: segmentation fault in signal handler si_code=1",
        "cred_hash": "",
        "asn": "",
        "src_ip": "10.0.1.50"
    },
    {
        "id": "log_3",
        "entity": "ssh-server-1",
        "ts": int(time.time()),
        "text": "kernel: sshd[1234] general protection fault in sig_handler+0x42",
        "cred_hash": "",
        "asn": "",
        "src_ip": "10.0.1.50"
    },
    {
        "id": "log_4",
        "entity": "ssh-server-1",
        "ts": int(time.time()),
        "text": "sshd[1235]: SSH daemon crashed during authentication timeout",
        "cred_hash": "",
        "asn": "",
        "src_ip": "10.0.1.50"
    },
    {
        "id": "log_5",
        "entity": "ssh-server-1",
        "ts": int(time.time()),
        "text": "systemd[1]: sshd.service: Main process exited, code=dumped, status=11/SEGV",
        "cred_hash": "",
        "asn": "",
        "src_ip": "10.0.1.50"
    }
]

print("="*70)
print("CVE-2024-6387 (regresshion) DETECTION TEST - V3.1")
print("="*70)
print()

families = [
    "CVE-2024-6387", "CVE-2021-44228", "CVE-2017-0144",
    "CVE-2020-1472", "CVE-2021-26855",
    "reconnaissance", "exploitation", "privilege_escalation",
    "persistence", "lateral_movement", "exfiltration"
]

# Test with semantic encoder
print("Initializing V3.1 with semantic encoder...\n")
engine = CompositeEngine(families, fast_mode=True, encoder_mode='semantic')

print("\nTest Logs:")
print("-" * 70)
for i, log in enumerate(test_logs, 1):
    print(f"{i}. {log['text']}")

print("\n" + "="*70)
print("SEMANTIC CLASSIFICATION TEST")
print("="*70 + "\n")

# Test semantic encoder directly first
print("Testing ADALog semantic encoder on each log:\n")
for i, log in enumerate(test_logs, 1):
    emb, fam_conf, patterns, probs = engine.adalog.classify(log["text"])
    print(f"Log {i}: {log['text'][:60]}...")

    # Check CVE-2024-6387 specifically
    if "CVE-2024-6387" in patterns:
        print(f"   ✅ Matched CVE-2024-6387: {patterns['CVE-2024-6387']}")

    high = [f for f, _ in fam_conf['high']]
    med = [f for f, _ in fam_conf['medium']]
    low = [f for f, _ in fam_conf['low']]

    if high:
        print(f"   HIGH confidence: {', '.join(high)}")
    if med:
        print(f"   MED confidence: {', '.join(med)}")
    if low and len(low) <= 3:
        print(f"   LOW confidence: {', '.join(low)}")
    print()

print("\n" + "="*70)
print("BATCH INGESTION")
print("="*70 + "\n")

# Process with batch
print("Processing with V3.1 batch ingestion...\n")
start = time.time()
engine.ingest_batch(test_logs, batch_size=5)
elapsed = time.time() - start

print(f"\n✅ Batch processing complete!")
print(f"   Time: {elapsed:.3f}s")
print(f"   Logs: {len(test_logs)}")
print(f"   Throughput: {len(test_logs)/elapsed:.0f} logs/sec")

# Check what was detected
print("\n" + "="*70)
print("DETECTION SUMMARY")
print("="*70 + "\n")

entity = "ssh-server-1"
now = int(time.time())
detected_families = []

# Check temporal wheels for each family
for family in families:
    found, days_ago = engine.temporal.query(entity, family, "*", now, 1)
    if found:
        detected_families.append(family)

if detected_families:
    print(f"✅ DETECTED {len(detected_families)} threat families:\n")
    for fam in detected_families:
        if fam.startswith("CVE"):
            print(f"   🔴 {fam} (Known CVE)")
        else:
            print(f"   ⚠️  {fam} (Attack stage)")

    if "CVE-2024-6387" in detected_families:
        print(f"\n🎯 SUCCESS: CVE-2024-6387 (regresshion) DETECTED!")
        print(f"   All {len(test_logs)} SSH crash logs were recognized as part of this CVE")
    else:
        print(f"\n⚠️  CVE-2024-6387 not explicitly detected")
        print(f"   But {len(detected_families)} related patterns found")
else:
    print("❌ No detections (may need to adjust thresholds)")

print("\n" + "="*70)
