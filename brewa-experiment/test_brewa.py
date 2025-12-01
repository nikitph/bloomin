# -*- coding: utf-8 -*-
"""
Main Test Runner
================

Run all BREWA experiments and generate comprehensive report.
"""

import torch
import sys

# Test imports
print("Testing imports...")
try:
    from brewa_utils import (
        hadamard_transform_batched,
        hamming_similarity_efficient,
        print_capacity_table,
        print_scaling_law_table,
    )
    print("[OK] brewa_utils")
except Exception as e:
    print("[FAIL] brewa_utils:", str(e))
    sys.exit(1)

try:
    from brewa_encoder import REWAEncoder, MultiMonoidREWAEncoder
    print("[OK] brewa_encoder")
except Exception as e:
    print("[FAIL] brewa_encoder:", str(e))
    sys.exit(1)

try:
    from multi_monoid_attention import (
        BooleanREWAHead,
        TropicalREWAHead,
        RealREWAHead,
        ProductMonoidHead,
        MultiMonoidAttention,
    )
    print("[OK] multi_monoid_attention")
except Exception as e:
    print("[FAIL] multi_monoid_attention:", str(e))
    sys.exit(1)

try:
    from brewa_attention import (
        BREWAAttention,
        BREWATransformerBlock,
        BREWATransformer,
    )
    print("[OK] brewa_attention")
except Exception as e:
    print("[FAIL] brewa_attention:", str(e))
    sys.exit(1)

print("\n" + "="*60)
print("All imports successful!")
print("="*60)

# Run quick smoke tests
print("\n" + "="*60)
print("Running Smoke Tests")
print("="*60)

print("\n1. Testing REWA Encoder...")
encoder = REWAEncoder(d_model=128, m_bits=32, monoid='boolean')
x = torch.randn(2, 10, 128)
encoded = encoder(x)
print("   Input:", x.shape, "-> Encoded:", encoded.shape)
print("   Compression: {:.1f}x".format(encoder.get_compression_ratio()))

print("\n2. Testing Multi-Monoid Attention...")
mma = MultiMonoidAttention(d_model=128, num_heads=4, m_bits=32)
Q = K = V = torch.randn(2, 10, 128)
out = mma(Q, K, V)
print("   Input:", Q.shape, "-> Output:", out.shape)
print("   Head stats:", mma.get_head_statistics())

print("\n3. Testing BREWA Attention...")
brewa = BREWAAttention(d_model=128, num_heads=4, m_bits=32)
x = torch.randn(2, 10, 128)
out = brewa(x)
print("   Input:", x.shape, "-> Output:", out.shape)
print("   Compression stats:", brewa.get_compression_stats())

print("\n4. Testing BREWA Transformer...")
model = BREWATransformer(
    vocab_size=1000,
    d_model=128,
    num_layers=2,
    num_heads=4,
    m_bits=32,
    max_seq_len=512,
)
input_ids = torch.randint(0, 1000, (2, 32))
logits = model(input_ids)
print("   Input:", input_ids.shape, "-> Logits:", logits.shape)
print("   Model stats:", model.get_model_stats())

print("\n" + "="*60)
print("All smoke tests passed! [OK]")
print("="*60)

# Print theoretical tables
print("\n")
print_capacity_table()
print_scaling_law_table()

print("\n" + "="*60)
print("BREWA Implementation Complete!")
print("="*60)
print("\nNext steps:")
print("1. Run: python experiment_compression.py")
print("2. Run: python experiment_monoid_specialization.py")
print("3. Run: python experiment_capacity.py (takes longer)")
print("\nAll experiments will generate plots and detailed results.")
print("="*60)
