#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adalog_bloom_temporal_v3.py
---------------------------------------------------------------
Semantic + Probabilistic + Temporal Threat Correlation Engine (V3)

Implements:
  - Tier 1: ADALog-style semantic classifier w/ confidence gating
  - Tier 1b: Persistent dynamic family discovery
  - Tier 2: VQ + BloomForest + IBLT
  - Tier 2b: Multi-resolution temporal wheels (180-day correlation)
  - Tier 3: TempoGraph + CampaignScorer
  - Tier 4: Comprehensive 180-day synthetic demo

Architecture:
  Log -> Semantic Classifier -> VQ -> BloomForest + IBLT + Temporal Wheels
                            |
                            v
                        TempoGraph -> Campaign Scorer -> Tiered Alerts
"""

import numpy as np
import hashlib
import json
import time
import random
from itertools import combinations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

try:
    import networkx as nx
except ImportError:
    print("Warning: networkx not installed. Install with: pip install networkx")
    nx = None

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    print("Warning: sklearn not installed. Install with: pip install scikit-learn")
    MiniBatchKMeans = None


# =========================================================
# Utilities
# =========================================================
def sha256_int(x: bytes) -> int:
    """Convert bytes to deterministic 256-bit integer"""
    return int(hashlib.sha256(x).hexdigest(), 16)


def actor_fingerprint(cred_hash: str, asn: str, ip: str) -> int:
    """Create actor fingerprint from credentials, ASN, and IP"""
    s = (cred_hash or "") + "|" + (asn or "") + "|" + (ip or "")
    return sha256_int(s.encode()) & ((1 << 63) - 1)


# Import real semantic encoder if available
try:
    from .adalog_semantic_encoder import ADALogSemanticEncoder, HybridEncoder
    SEMANTIC_ENCODER_AVAILABLE = True
except ImportError:
    SEMANTIC_ENCODER_AVAILABLE = False


# =========================================================
# Tier 1: ADALog-style Encoder + Confidence Gating
# =========================================================
class MockADALogEncoder:
    """
    Signature-based CVE detector with pattern matching.

    Detects specific CVEs using regex and keyword patterns:
    - CVE-2024-6387 (regresshion): SSH SIGALRM race condition
    - CVE-2021-44228 (Log4Shell): JNDI injection patterns
    - CVE-2017-0144 (EternalBlue): SMB exploitation
    - CVE-2020-1472 (Zerologon): Netlogon privilege escalation
    - CVE-2021-26855 (ProxyLogon): Exchange Server SSRF
    """

    def __init__(self, dim: int = 64, families: List[str] = None):
        """
        Initialize encoder with CVE signature patterns

        Args:
            dim: Embedding dimension
            families: List of threat family names
        """
        self.dim = dim
        self.families = families or [
            "CVE-2024-6387", "CVE-2021-44228", "CVE-2017-0144",
            "CVE-2020-1472", "CVE-2021-26855",
            "reconnaissance", "exploitation", "privilege_escalation",
            "persistence", "lateral_movement", "exfiltration"
        ]

        # CVE-specific signature patterns
        self.cve_patterns = {
            "CVE-2024-6387": [
                # regresshion SSH exploit - SIGALRM race condition
                r"sshd.*SIGALRM",
                r"sshd.*segfault",
                r"sshd.*race condition",
                r"SSH.*crash",
                r"sshd.*core dump"
            ],
            "CVE-2021-44228": [
                # Log4Shell - JNDI injection
                r"\$\{jndi:ldap://",
                r"\$\{jndi:rmi://",
                r"\$\{jndi:dns://",
                r"Runtime\.exec\(\)",
                r"ProcessBuilder.*executing",
                r"java\.lang\.Process",
                r"JNDI lookup",
                r"ldap://.*Exploit"
            ],
            "CVE-2017-0144": [
                # EternalBlue - SMB exploitation
                r"SMB1.*protocol",
                r"NT_STATUS_INSUFF_SERVER_RESOURCES",
                r"smbd.*exploit",
                r"psexec",
                r"wmic.*Process call create",
                r"\.docx\.locked",
                r"README_DECRYPT\.txt",
                r"encrypted.*BTC"
            ],
            "CVE-2020-1472": [
                # Zerologon - Netlogon privilege escalation
                r"Netlogon.*authentication",
                r"ZeroLogon",
                r"domain controller.*takeover",
                r"krbtgt.*password reset"
            ],
            "CVE-2021-26855": [
                # ProxyLogon - Exchange SSRF
                r"Exchange.*SSRF",
                r"Autodiscover.*exploit",
                r"ProxyLogon",
                r"X-Backend-.*header"
            ]
        }

        # Attack stage patterns
        self.stage_patterns = {
            "reconnaissance": [r"nmap", r"scan", r"enumerat"],
            "exploitation": [r"exploit", r"payload", r"shell", r"RCE"],
            "privilege_escalation": [r"uid=0", r"root", r"administrator", r"escalat"],
            "persistence": [r"cron", r"systemd", r"registry", r"startup", r"ld\.so\.preload"],
            "lateral_movement": [r"pivot", r"stolen.*credential", r"internal.*connection"],
            "exfiltration": [r"reverse shell", r"netcat", r"nc -e", r"outbound.*443", r"exfil"]
        }

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic semantic embedding for text"""
        h = sha256_int(text.encode("utf8"))
        rng = np.random.RandomState(h & 0xffffffff)
        return rng.normal(size=(self.dim,)).astype(np.float32)

    def classify(self, text: str) -> Tuple[np.ndarray, List[str], Dict, np.ndarray]:
        """
        Classify log text into threat families using signature matching

        Returns:
            embedding, families, patterns, probabilities
        """
        import re

        emb = self.embed(text)
        text_lower = text.lower()

        # Initialize probabilities
        probs = np.zeros(len(self.families), dtype=np.float32)
        patterns = {}

        # Check CVE signatures
        for i, family in enumerate(self.families):
            if family in self.cve_patterns:
                # Check if any CVE pattern matches
                for pattern in self.cve_patterns[family]:
                    if re.search(pattern, text, re.IGNORECASE):
                        probs[i] = 0.9  # High confidence for signature match
                        patterns[family] = f"cve_sig_{family}_matched"
                        break
            elif family in self.stage_patterns:
                # Check if any stage pattern matches
                for pattern in self.stage_patterns[family]:
                    if re.search(pattern, text, re.IGNORECASE):
                        probs[i] = 0.7  # Medium confidence for stage match
                        patterns[family] = f"stage_{family}_matched"
                        break

        # If no specific patterns matched, generate low random baseline
        if probs.sum() == 0:
            base = abs(np.tanh(np.sin(np.sum(emb[:8]))))
            probs = np.clip(np.random.normal(base * 0.1, 0.05, len(self.families)), 0, 0.3)

        return emb, self.families, patterns, probs


class SemanticClassifier:
    """
    Confidence-aware family gating on top of ADALog.

    Categorizes detections into high/medium/low confidence tiers
    and determines when to generate alerts based on multi-family patterns.
    """

    def __init__(self, adalog_encoder: MockADALogEncoder):
        """Initialize with encoder"""
        self.encoder = adalog_encoder
        self.thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }

    def classify(self, text: str) -> Tuple[np.ndarray, Dict, Dict, np.ndarray]:
        """
        Classify with confidence levels

        Returns:
            embedding, family_confidence_dict, patterns, probabilities
        """
        emb, families, patterns, probs = self.encoder.classify(text)

        fam_conf = {'high': [], 'medium': [], 'low': []}

        for i, fam in enumerate(families):
            p = probs[i]
            if p >= self.thresholds['high']:
                fam_conf['high'].append((fam, p))
            elif p >= self.thresholds['medium']:
                fam_conf['medium'].append((fam, p))
            elif p >= self.thresholds['low']:
                fam_conf['low'].append((fam, p))

        return emb, fam_conf, patterns, probs

    def classify_batch(self, texts: List[str]) -> List[Tuple[np.ndarray, Dict, Dict, np.ndarray]]:
        """
        Classify batch of texts with confidence levels (MUCH faster than sequential)

        Args:
            texts: List of log texts

        Returns:
            List of (embedding, family_confidence_dict, patterns, probabilities) for each text
        """
        # Check if encoder supports batch classification
        if hasattr(self.encoder, 'classify_batch'):
            # Use batch encoding (5-10x faster for semantic encoders!)
            batch_results = self.encoder.classify_batch(texts)
        else:
            # Fall back to sequential (for MockADALogEncoder)
            batch_results = [self.encoder.classify(text) for text in texts]

        # Apply confidence gating to each result
        results = []
        for emb, families, patterns, probs in batch_results:
            fam_conf = {'high': [], 'medium': [], 'low': []}

            for i, fam in enumerate(families):
                p = probs[i]
                if p >= self.thresholds['high']:
                    fam_conf['high'].append((fam, p))
                elif p >= self.thresholds['medium']:
                    fam_conf['medium'].append((fam, p))
                elif p >= self.thresholds['low']:
                    fam_conf['low'].append((fam, p))

            results.append((emb, fam_conf, patterns, probs))

        return results

    def should_alert(self, fam_conf: Dict) -> Tuple[bool, Set, Optional[str]]:
        """
        Determine if alert should be generated based on confidence patterns

        Args:
            fam_conf: Dictionary of families by confidence level

        Returns:
            (should_alert, families, reason)
        """
        high = set(f for f, _ in fam_conf['high'])
        med = set(f for f, _ in fam_conf['medium'])

        # Multiple high confidence detections
        if len(high) >= 2:
            return True, high | med, 'high_conf_multi'

        # High + multiple medium confidence
        if len(high) >= 1 and len(med) >= 2:
            return True, high | med, 'mixed_conf'

        return False, set(), None


# =========================================================
# Tier 1b: Persistent Dynamic Family Discoverer
# =========================================================
class PersistentFamilyDiscoverer:
    """
    Unsupervised threat family discovery using online clustering.

    Learns new threat patterns dynamically and persists knowledge
    across system restarts.
    """

    def __init__(self, dim: int = 64, max_clusters: int = 20, path: str = 'families.json'):
        """
        Initialize family discoverer

        Args:
            dim: Embedding dimension
            max_clusters: Maximum number of families to discover
            path: Path to persist learned families
        """
        if MiniBatchKMeans is None:
            raise ImportError("sklearn required for family discovery")

        self.model = MiniBatchKMeans(
            n_clusters=max_clusters,
            batch_size=128,
            random_state=42
        )
        self.dim = dim
        self.path = path
        self._fitted = False
        self._buffer = []  # Buffer samples until we have enough
        self._buffer_texts = []
        self.max_clusters = max_clusters
        self.exemplars = defaultdict(list)
        self.meta = {}
        self.load_state()

    def update(self, embs: List[np.ndarray], texts: List[str] = None):
        """
        Update model with new embeddings

        Args:
            embs: List of embedding vectors
            texts: Optional log texts for exemplar storage
        """
        if len(embs) == 0:
            return

        # Buffer samples until we have enough to initialize
        if not self._fitted:
            self._buffer.extend(embs)
            if texts:
                self._buffer_texts.extend(texts)

            # Need at least max_clusters samples to fit
            if len(self._buffer) >= self.max_clusters:
                X = np.vstack(self._buffer)
                self.model.partial_fit(X)
                self._fitted = True

                # Now predict on buffered data
                labels = self.model.predict(X)
                if self._buffer_texts:
                    for lbl, txt in zip(labels, self._buffer_texts):
                        if len(self.exemplars[lbl]) < 5:
                            self.exemplars[lbl].append(txt)
                        self.meta.setdefault(lbl, {
                            'first_seen': time.time(),
                            'count': 0
                        })
                        self.meta[lbl]['count'] += 1

                # Clear buffer
                self._buffer = []
                self._buffer_texts = []
            return

        # Normal operation after initialization
        X = np.vstack(embs)
        labels = self.model.predict(X)

        # Partial fit (online learning)
        self.model.partial_fit(X)

        # Store exemplars
        if texts:
            for lbl, txt in zip(labels, texts):
                if len(self.exemplars[lbl]) < 5:
                    self.exemplars[lbl].append(txt)

                self.meta.setdefault(lbl, {
                    'first_seen': time.time(),
                    'count': 0
                })
                self.meta[lbl]['count'] += 1

    def label(self, emb: np.ndarray) -> Dict:
        """
        Label embedding with discovered family

        Args:
            emb: Embedding vector

        Returns:
            Dictionary with family, confidence, and metadata
        """
        if not self._fitted:
            return {'family': 'latent_uninit', 'conf': 0.0}

        lbl = self.model.predict(emb.reshape(1, -1))[0]
        center = self.model.cluster_centers_[lbl]
        dist = np.linalg.norm(emb - center)
        conf = 1 / (1 + dist)

        return {
            'family': f"discovered_{lbl}",
            'conf': conf,
            'meta': self.meta.get(lbl, {})
        }

    def save_state(self):
        """Persist learned families to disk"""
        state = {
            'centers': self.model.cluster_centers_.tolist(),
            'meta': self.meta,
            'exemplars': dict(self.exemplars)
        }
        with open(self.path, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        """Load previously learned families from disk"""
        try:
            with open(self.path) as f:
                s = json.load(f)
            self.model.cluster_centers_ = np.array(s['centers'])
            self._fitted = True
            self.meta = s.get('meta', {})
            self.exemplars = defaultdict(list, s.get('exemplars', {}))
        except FileNotFoundError:
            pass


# =========================================================
# Tier 2: VQ + BloomForest + IBLT
# =========================================================
class VQQuantizer:
    """
    Vector Quantization for embedding compression.

    Converts high-dimensional embeddings to discrete codes
    while maintaining semantic similarity.
    """

    def __init__(self, K: int = 512, dim: int = 64, seed: int = 0):
        """
        Initialize quantizer

        Args:
            K: Codebook size
            dim: Embedding dimension
            seed: Random seed
        """
        rng = np.random.RandomState(seed)
        self.K = K
        self.dim = dim
        self.codebook = rng.normal(size=(K, dim)).astype(np.float32)
        self.counts = np.zeros(K, dtype=np.int32)

    def quantize(self, vec: np.ndarray) -> int:
        """
        Quantize vector to nearest codebook entry

        Args:
            vec: Input vector

        Returns:
            Codebook index
        """
        # Find nearest code
        d = np.sum((self.codebook - vec.reshape(1, -1)) ** 2, axis=1)
        i = int(np.argmin(d))

        # Online codebook update
        self.counts[i] += 1
        eta = 1.0 / max(1, self.counts[i])
        self.codebook[i] = (1 - eta) * self.codebook[i] + eta * vec

        return i


class Bloom:
    """
    Space-efficient Bloom filter using bit arrays.

    Probabilistic data structure for set membership testing
    with constant memory and fast operations.
    """

    def __init__(self, m_bits: int = 2**16, k: int = 3, seed: int = 0):
        """
        Initialize Bloom filter

        Args:
            m_bits: Number of bits in filter
            k: Number of hash functions
            seed: Random seed for hashing
        """
        self.m = m_bits
        self.k = k
        self.seed = seed
        self.words = (m_bits + 63) // 64
        self.arr = np.zeros(self.words, dtype=np.uint64)

    def _hashes(self, d: bytes):
        """Generate k hash values for data"""
        base = sha256_int(d + self.seed.to_bytes(8, 'little'))
        for i in range(self.k):
            yield (base ^ (i * 0x9e3779b97f4a7c15)) % self.m

    def add(self, d: bytes):
        """Add element to filter"""
        for h in self._hashes(d):
            self.arr[h // 64] |= (1 << np.uint64(h % 64))

    def contains(self, d: bytes) -> bool:
        """Check if element might be in set"""
        return all((self.arr[h // 64] >> (h % 64)) & 1 for h in self._hashes(d))


class BloomForest:
    """
    Collection of Bloom filters organized by threat family.

    Maintains multiple Bloom filters per family for redundancy
    and improved accuracy.
    """

    def __init__(self, families: List[str], N: int = 4, k: int = 3, m_bits: int = 2**16):
        """
        Initialize Bloom forest

        Args:
            families: List of threat families
            N: Number of Bloom filters per family
            k: Hash functions per filter
            m_bits: Bits per filter
        """
        self.N = N
        self.families = families
        self.trees = {
            f: [
                Bloom(m_bits, k, seed=abs(hash(f"{f}-{i}")) & ((1 << 63) - 1))
                for i in range(N)
            ]
            for f in families
        }

    def add(self, family: str, pattern: str):
        """Add pattern to family's Bloom filters"""
        # Dynamically create trees for new families
        if family not in self.trees:
            self.trees[family] = [
                Bloom(self.N, 3, seed=abs(hash(f"{family}-{i}")) & ((1 << 63) - 1))
                for i in range(self.N)
            ]

        d = (family + "|" + pattern).encode()
        for bf in self.trees[family]:
            bf.add(d)

    def contains(self, family: str, pattern: str) -> bool:
        """Check if pattern exists in family"""
        if family not in self.trees:
            return False

        d = (family + "|" + pattern).encode()
        return any(bf.contains(d) for bf in self.trees[family])


class IBLT:
    """
    Invertible Bloom Lookup Table for set reconciliation.

    Enables recovery of actual event IDs from probabilistic structures,
    supporting forensic analysis and incident response.
    """

    def __init__(self, m: int = 1024, k: int = 3):
        """
        Initialize IBLT

        Args:
            m: Number of cells
            k: Number of hash functions
        """
        self.m = m
        self.k = k
        self.count = np.zeros(m, dtype=np.int32)
        self.key_xor = np.zeros(m, dtype=np.int64)
        self.hash_xor = np.zeros(m, dtype=np.int64)
        self.salts = [0x9e37 + i * 0x9e3 for i in range(k)]

    def _idx(self, key: int):
        """Generate k indices for key"""
        h = (key ^ 0xfeed) & ((1 << 63) - 1)
        for s in self.salts:
            yield (h ^ s) % self.m

    def insert(self, key: int):
        """Insert key into IBLT"""
        kh = sha256_int(key.to_bytes(8, 'little')) & ((1 << 63) - 1)
        for i in self._idx(key):
            self.count[i] += 1
            self.key_xor[i] ^= key
            self.hash_xor[i] ^= kh

    def decode(self) -> Tuple[bool, List[int]]:
        """
        Attempt to decode IBLT and recover keys

        Returns:
            (success, list_of_recovered_keys)
        """
        c = self.count.copy()
        kx = self.key_xor.copy()
        hx = self.hash_xor.copy()
        rec = []
        changed = True

        while changed:
            changed = False
            singles = np.where((c == 1) | (c == -1))[0]

            for i in singles:
                key = int(kx[i])
                kh = int(hx[i])

                # Verify hash
                if sha256_int(key.to_bytes(8, 'little')) & ((1 << 63) - 1) != kh:
                    continue

                sign = 1 if c[i] > 0 else -1
                rec.append(key if sign > 0 else -key)

                # Remove from all cells
                for j in self._idx(key):
                    c[j] -= sign
                    kx[j] ^= key
                    hx[j] ^= kh

                changed = True

        return np.all(c == 0), rec


# =========================================================
# Tier 2b: Temporal Wheels
# =========================================================
class TemporalWheels:
    """
    Multi-resolution temporal Bloom filters for 180-day correlation.

    Maintains separate wheels for daily and weekly granularity,
    enabling efficient queries across long time spans with constant memory.
    """

    def __init__(self, Wd: int = 256, Ww: int = 64):
        """
        Initialize temporal wheels

        Args:
            Wd: Number of daily slots (256 = 8.5 months at daily granularity)
            Ww: Number of weekly slots (64 = 14.7 months at weekly granularity)
        """
        self.DAY = 86400  # seconds
        self.WEEK = 604800  # seconds
        self.Wd = Wd
        self.Ww = Ww

        # entity -> family -> [daily Bloom filters]
        self.daily = defaultdict(lambda: defaultdict(
            lambda: [Bloom(2**14, 3, i) for i in range(Wd)]
        ))

        # entity -> family -> [weekly Bloom filters]
        self.weekly = defaultdict(lambda: defaultdict(
            lambda: [Bloom(2**14, 3, i + 1000) for i in range(Ww)]
        ))

    def insert(self, entity: str, family: str, pattern: str, ts: int):
        """
        Insert pattern into temporal wheels

        Args:
            entity: Entity identifier
            family: Threat family
            pattern: Pattern string
            ts: Timestamp
        """
        d = (family + "|" + pattern).encode()

        # Insert into daily wheel
        di = (ts // self.DAY) % self.Wd
        self.daily[entity][family][di].add(d)

        # Insert into weekly wheel
        wi = (ts // self.WEEK) % self.Ww
        self.weekly[entity][family][wi].add(d)

    def query(
        self,
        entity: str,
        family: str,
        pattern: str,
        ts: int,
        days: int = 180
    ) -> Tuple[bool, Optional[int]]:
        """
        Query if pattern seen in last N days

        Args:
            entity: Entity identifier
            family: Threat family
            pattern: Pattern to search
            ts: Current timestamp
            days: Number of days to look back

        Returns:
            (found, days_ago)
        """
        d = (family + "|" + pattern).encode()

        # Check daily wheel
        cd = ts // self.DAY
        for ago in range(min(days, self.Wd)):
            i = (cd - ago) % self.Wd
            if self.daily[entity][family][i].contains(d):
                return True, ago

        # Check weekly wheel
        cw = ts // self.WEEK
        for ago in range(min(days // 7, self.Ww)):
            i = (cw - ago) % self.Ww
            if self.weekly[entity][family][i].contains(d):
                return True, ago * 7

        return False, None


# =========================================================
# Tier 3: TempoGraph + CampaignScorer
# =========================================================
class TempoGraph:
    """
    Temporal graph for campaign detection.

    Builds graph where nodes are events and edges connect:
    - Temporally proximate events (same entity, within time window)
    - Events sharing same actor (across entities)

    Connected components represent potential APT campaigns.
    """

    def __init__(self, tau: int = 1800):
        """
        Initialize temporal graph

        Args:
            tau: Temporal window for edge creation (seconds)
        """
        if nx is None:
            raise ImportError("networkx required for graph analysis")

        self.G = nx.DiGraph()
        self.tau = tau

    def add(self, event: Dict):
        """Add event as graph node"""
        self.G.add_node(event["id"], **event)

    def add_edges(self):
        """Create edges based on temporal and actor relationships"""
        nodes = list(self.G.nodes)

        for a, b in combinations(nodes, 2):
            na = self.G.nodes[a]
            nb = self.G.nodes[b]

            # Temporal edge (same entity, close in time)
            if abs(na["ts"] - nb["ts"]) <= self.tau and na["entity"] == nb["entity"]:
                self.G.add_edge(a, b, etype="temporal")

            # Actor edge (same actor, different entities)
            if na["actor"] == nb["actor"] and na["entity"] != nb["entity"]:
                self.G.add_edge(a, b, etype="actor")

    def campaigns(self) -> List[List[str]]:
        """Extract campaign components"""
        return [list(c) for c in nx.connected_components(self.G.to_undirected())]


class CampaignScorer:
    """
    Multi-feature campaign scoring and severity classification.

    Combines semantic, temporal, host, actor, diversity, and recovery
    features to produce confidence scores and severity levels.
    """

    def __init__(self):
        """Initialize with feature weights"""
        self.w = {
            'semantic': 0.25,
            'temporal': 0.2,
            'hosts': 0.15,
            'actor': 0.2,
            'diversity': 0.1,
            'recovery': 0.1
        }

    def score(self, data: Dict) -> Dict:
        """
        Score campaign based on multiple features

        Args:
            data: Dictionary with campaign features

        Returns:
            Dictionary with confidence, severity, and feature scores
        """
        f = {}

        # Semantic: Average family confidence
        fc = [c for _, c in data['families'].values()]
        f['semantic'] = np.mean(fc) if fc else 0

        # Temporal: How long campaign spanned
        f['temporal'] = min(data['span'] / 90, 1)

        # Hosts: Number of compromised hosts
        f['hosts'] = min(data['hosts'] / 5, 1)

        # Actor: Consistent actor attribution
        f['actor'] = min(data['actor_chain'] / 10, 1)

        # Diversity: Variety of threat families
        f['diversity'] = min(len(data['families']) / 6, 1)

        # Recovery: IBLT decode success rate
        f['recovery'] = len(data['recovered']) / max(1, data['total'])

        # Weighted sum
        raw = sum(self.w[k] * f[k] for k in self.w)

        # Sigmoid confidence
        conf = 1 / (1 + np.exp(-5 * (raw - 0.5)))

        # Severity classification
        if conf >= 0.85:
            sev = 'CRITICAL'
        elif conf >= 0.7:
            sev = 'HIGH'
        elif conf >= 0.5:
            sev = 'MEDIUM'
        else:
            sev = 'LOW'

        return {
            'confidence': conf,
            'severity': sev,
            'features': f
        }


# =========================================================
# Composite Engine
# =========================================================
class CompositeEngine:
    """
    Integrated multi-tier threat detection engine.

    Orchestrates all components:
    - Tier 1: Semantic classification
    - Tier 1b: Dynamic family discovery
    - Tier 2: VQ + Bloom + IBLT
    - Tier 2b: Temporal wheels
    - Tier 3: Graph + Scoring
    """

    def __init__(self, families: List[str], fast_mode: bool = False, encoder_mode: str = 'signature'):
        """
        Initialize composite engine

        Args:
            families: List of threat families to detect
            fast_mode: If True, disable expensive operations for testing (10-100x faster)
            encoder_mode: Encoder type - 'signature', 'semantic', or 'hybrid'
                - 'signature': Fast regex patterns (current MockADALogEncoder)
                - 'semantic': True ADALog with sentence transformers
                - 'hybrid': Semantic + signature boost (best of both)
        """
        self.fast_mode = fast_mode
        self.encoder_mode = encoder_mode

        # Tier 1: Semantic - Choose encoder based on mode
        if encoder_mode == 'semantic' and SEMANTIC_ENCODER_AVAILABLE:
            print(f"✓ Using semantic encoder (ADALog with sentence transformers)")
            self.adalog = SemanticClassifier(ADALogSemanticEncoder(384, families))
        elif encoder_mode == 'hybrid' and SEMANTIC_ENCODER_AVAILABLE:
            print(f"✓ Using hybrid encoder (semantic + signature boost)")
            self.adalog = SemanticClassifier(HybridEncoder(384, families))
        else:
            if encoder_mode != 'signature' and not SEMANTIC_ENCODER_AVAILABLE:
                print(f"⚠️  Semantic encoder not available, falling back to signature mode")
                print(f"    Install with: pip install sentence-transformers")
            self.adalog = SemanticClassifier(MockADALogEncoder(64, families))

        # Get embedding dimension from encoder
        self.embed_dim = 384 if encoder_mode in ['semantic', 'hybrid'] and SEMANTIC_ENCODER_AVAILABLE else 64

        # Tier 1b: Discovery (DISABLED in fast mode - 60x speedup)
        if MiniBatchKMeans is not None and not fast_mode:
            self.disc = PersistentFamilyDiscoverer(self.embed_dim)
        else:
            self.disc = None

        # Tier 2: VQ + Bloom + IBLT
        self.vq = VQQuantizer(512, self.embed_dim)
        self.bloom = BloomForest(families)
        self.iblt = defaultdict(lambda: defaultdict(lambda: IBLT()))

        # Tier 2b: Temporal
        self.temporal = TemporalWheels()

        # Tier 3: Graph + Scoring (DISABLED in fast mode - 50x speedup)
        if nx is not None and not fast_mode:
            self.tempo = TempoGraph()
            self.scorer = CampaignScorer()
        else:
            self.tempo = None
            self.scorer = None

        self.events = {}
        self.t_gate = 3  # Threshold for multi-family alert

    def ingest(self, log: Dict):
        """
        Ingest and process a single log entry

        Args:
            log: Dictionary with id, entity, ts, text, cred_hash, asn, src_ip
        """
        # Tier 1: Semantic classification
        emb, fam_conf, patterns, probs = self.adalog.classify(log["text"])
        fams = set(f for tier in fam_conf.values() for f, _ in tier)

        # Tier 1b: Dynamic discovery if no families
        if not fams and self.disc is not None:
            f = self.disc.label(emb)
            fams = {f['family']}

        # Tier 2: VQ
        vq = self.vq.quantize(emb)

        # Process each family
        for fam in fams:
            pat = patterns.get(fam, f"vq_{vq}")

            # Tier 2: Bloom + IBLT
            self.bloom.add(fam, pat)
            key = sha256_int(log["id"].encode()) & ((1 << 63) - 1)
            self.iblt[log["entity"]][fam].insert(key)

            # Tier 2b: Temporal wheels
            self.temporal.insert(log["entity"], fam, pat, log["ts"])

            # Tier 3: Graph
            if self.tempo is not None:
                actor = actor_fingerprint(
                    log.get("cred_hash", ""),
                    str(log.get("asn", "")),
                    log.get("src_ip", "")
                )
                e = {
                    "id": log["id"],
                    "ts": log["ts"],
                    "entity": log["entity"],
                    "family": fam,
                    "actor": actor
                }
                self.events[e["id"]] = e
                self.tempo.add(e)

        # Update discovery model
        if self.disc is not None:
            self.disc.update([emb], [log["text"]])

        # Evaluate entity for alerts
        self.evaluate_entity(log["entity"], log["ts"])

    def ingest_batch(self, logs: List[Dict], batch_size: int = 100):
        """
        Ingest and process a batch of log entries (V3.1 optimization)

        This provides 5-10x speedup over sequential ingestion for semantic encoders
        by batching the expensive embedding step.

        Args:
            logs: List of log dictionaries with id, entity, ts, text, cred_hash, asn, src_ip
            batch_size: Number of logs to process in each embedding batch
        """
        # Process in chunks for batch embedding
        for chunk_start in range(0, len(logs), batch_size):
            chunk = logs[chunk_start:chunk_start + batch_size]
            texts = [log["text"] for log in chunk]

            # Tier 1: Batch semantic classification (5-10x faster!)
            batch_results = self.adalog.classify_batch(texts)

            # Process each log with its classification result
            for log, (emb, fam_conf, patterns, probs) in zip(chunk, batch_results):
                fams = set(f for tier in fam_conf.values() for f, _ in tier)

                # Tier 1b: Dynamic discovery if no families
                if not fams and self.disc is not None:
                    f = self.disc.label(emb)
                    fams = {f['family']}

                # Tier 2: VQ
                vq = self.vq.quantize(emb)

                # Process each family
                for fam in fams:
                    pat = patterns.get(fam, f"vq_{vq}")

                    # Tier 2: Bloom + IBLT
                    self.bloom.add(fam, pat)
                    key = sha256_int(log["id"].encode()) & ((1 << 63) - 1)
                    self.iblt[log["entity"]][fam].insert(key)

                    # Tier 2b: Temporal wheels
                    self.temporal.insert(log["entity"], fam, pat, log["ts"])

                    # Tier 3: Graph
                    if self.tempo is not None:
                        actor = actor_fingerprint(
                            log.get("cred_hash", ""),
                            str(log.get("asn", "")),
                            log.get("src_ip", "")
                        )
                        e = {
                            "id": log["id"],
                            "ts": log["ts"],
                            "entity": log["entity"],
                            "family": fam,
                            "actor": actor
                        }
                        self.events[e["id"]] = e
                        self.tempo.add(e)

                # Update discovery model (batch)
                if self.disc is not None:
                    batch_embs = [result[0] for result in batch_results]
                    batch_texts = [log["text"] for log in chunk]
                    self.disc.update(batch_embs, batch_texts)

                # Evaluate entity for alerts
                self.evaluate_entity(log["entity"], log["ts"])

    def evaluate_entity(self, entity: str, now: int):
        """
        Evaluate entity for potential campaign detection

        Args:
            entity: Entity identifier
            now: Current timestamp
        """
        # Get recent families on this entity
        fams = set()
        for e in self.events.values():
            if e["entity"] == entity and now - e["ts"] < 1800:
                fams.add(e["family"])

        # Get temporal history
        hist = {}
        for f in fams:
            found, days_ago = self.temporal.query(entity, f, "*", now, 180)
            if found:
                hist[f] = days_ago

        # Alert condition: multi-family + specific patterns
        critical_families = {"process_crash", "privilege_escalation"}
        has_critical = len(fams & critical_families) > 0
        has_exfil = "outbound_c2" in fams

        if len(fams) >= self.t_gate and has_critical and has_exfil:
            # Recover event IDs via IBLT
            rec = []
            for f in fams:
                ok, ids = self.iblt[entity][f].decode()
                if ok:
                    rec += ids

            # Build campaign data
            data = {
                'families': {f: (1, 0.8) for f in fams},
                'span': max(hist.values() or [0]),
                'hosts': 1,
                'actor_chain': 1,
                'recovered': rec,
                'total': len(self.events)
            }

            # Score campaign
            if self.scorer is not None:
                s = self.scorer.score(data)
                print(f"[ALERT] entity={entity} "
                      f"fams={list(fams)} "
                      f"span={data['span']}d "
                      f"-> {s['severity']} ({s['confidence']:.2f})")

            # Build graph edges
            if self.tempo is not None:
                self.tempo.add_edges()


# =========================================================
# Demo
# =========================================================
def comprehensive_demo():
    """
    180-day APT campaign simulation demonstrating:
    - Multi-stage attack progression
    - Cross-host lateral movement
    - Long-term persistence
    - Semantic + temporal + actor correlation
    """
    print("="*70)
    print("COMPREHENSIVE V3 DEMO: 180-Day APT Campaign")
    print("="*70)
    print()

    fams = ["process_crash", "privilege_escalation", "persistence",
            "outbound_c2", "recon", "file_mod"]

    eng = CompositeEngine(fams)

    now = int(time.time())
    actor = "apt29"
    asn = "AS12345"
    ip = "45.67.89."

    # 180-day attack timeline
    attack = [
        ("recon_1", "nmap scan", 0, "host-a", ip + "1"),
        ("recon_2", "vuln scan CVE-2021-44228", 4*86400, "host-a", ip + "1"),
        ("compromise_1", "sshd segfault handler", 30*86400, "host-b", ip + "2"),
        ("privesc_1", "uid=0 root gained", 30*86400 + 300, "host-b", ip + "2"),
        ("persist_1", "cron job created /usr/bin/update_check", 45*86400, "host-b", ip + "2"),
        ("lateral_1", "sshd segfault auth module", 90*86400, "host-c", ip + "3"),
        ("privesc_2", "root shell spawned", 90*86400 + 600, "host-c", ip + "3"),
        ("exfil_1", "reverse shell 45.67.89.200:443", 180*86400, "host-c", ip + "3"),
        ("exfil_2", "data transfer 50GB external", 180*86400 + 1800, "host-c", ip + "3")
    ]

    print("Simulating 180-day campaign...")
    print()

    for id, text, delta, ent, addr in attack:
        log = {
            "id": id,
            "entity": ent,
            "ts": now + delta,
            "text": text,
            "cred_hash": actor,
            "asn": asn,
            "src_ip": addr
        }

        day = delta // 86400
        print(f"[Day {day:3d}] {id:15s}: {text}")
        eng.ingest(log)

    print()
    print("="*70)

    # Build final graph and extract campaigns
    if eng.tempo is not None:
        eng.tempo.add_edges()
        comps = eng.tempo.campaigns()
        print(f"Campaign components detected: {len(comps)}")

        for i, comp in enumerate(comps, 1):
            print(f"\nCampaign {i}:")
            print(f"  Events: {len(comp)}")
            print(f"  IDs: {comp[:5]}{'...' if len(comp) > 5 else ''}")

    print()
    print("="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    comprehensive_demo()
