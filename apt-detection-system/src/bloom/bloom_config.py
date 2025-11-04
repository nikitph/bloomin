"""
Bloom Filter Configuration Module

Defines optimal parameters for dual-layer Bloom filters:
- Pattern Bloom: Fast CVE signature matching
- Event Bloom: Temporal correlation tracking
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class BloomConfig:
    """Configuration manager for Bloom filter parameters"""

    def __init__(self, config_path: str = None):
        """
        Initialize Bloom filter configuration

        Args:
            config_path: Path to YAML config file. If None, uses default values.
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = self._default_config()

        self._validate_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'bloom_filters': {
                'pattern_capacity': 100000,
                'pattern_error_rate': 0.01,
                'event_capacity': 1000000,
                'event_error_rate': 0.023,
                'hash_functions_pattern': 7,
                'hash_functions_event': 6
            },
            'detection': {
                'pattern_match_threshold': 3,
                'correlation_window_days': 'unlimited',
                'tier1_threshold': 1,
                'tier2_threshold': 2,
                'tier3_threshold': 3
            },
            'performance': {
                'batch_size': 512,
                'num_threads': 8,
                'enable_parallel': False
            },
            'tokenization': {
                'ngram_min': 4,
                'ngram_max': 6,
                'use_sliding_window': True
            },
            'memory': {
                'max_memory_mb': 720,
                'enable_monitoring': True,
                'alert_threshold_mb': 650
            }
        }

    def _validate_config(self):
        """Validate configuration parameters"""
        bf_config = self.config.get('bloom_filters', {})

        # Validate error rates
        if not (0 < bf_config.get('pattern_error_rate', 0.01) < 1):
            raise ValueError("pattern_error_rate must be between 0 and 1")
        if not (0 < bf_config.get('event_error_rate', 0.023) < 1):
            raise ValueError("event_error_rate must be between 0 and 1")

        # Validate capacities
        if bf_config.get('pattern_capacity', 0) <= 0:
            raise ValueError("pattern_capacity must be positive")
        if bf_config.get('event_capacity', 0) <= 0:
            raise ValueError("event_capacity must be positive")

    @property
    def pattern_capacity(self) -> int:
        """Get pattern Bloom filter capacity"""
        return self.config['bloom_filters']['pattern_capacity']

    @property
    def pattern_error_rate(self) -> float:
        """Get pattern Bloom filter error rate"""
        return self.config['bloom_filters']['pattern_error_rate']

    @property
    def event_capacity(self) -> int:
        """Get event Bloom filter capacity"""
        return self.config['bloom_filters']['event_capacity']

    @property
    def event_error_rate(self) -> float:
        """Get event Bloom filter error rate"""
        return self.config['bloom_filters']['event_error_rate']

    @property
    def pattern_match_threshold(self) -> int:
        """Minimum pattern matches to trigger alert"""
        return self.config['detection']['pattern_match_threshold']

    @property
    def tier_thresholds(self) -> Dict[int, int]:
        """Get alert tier thresholds"""
        return {
            1: self.config['detection']['tier1_threshold'],
            2: self.config['detection']['tier2_threshold'],
            3: self.config['detection']['tier3_threshold']
        }

    @property
    def ngram_range(self) -> tuple:
        """Get n-gram range for tokenization"""
        return (
            self.config['tokenization']['ngram_min'],
            self.config['tokenization']['ngram_max']
        )

    @property
    def batch_size(self) -> int:
        """Get batch processing size"""
        return self.config['performance']['batch_size']

    @property
    def max_memory_mb(self) -> int:
        """Get maximum memory budget in MB"""
        return self.config['memory']['max_memory_mb']

    def get_bloom_memory_estimate(self) -> Dict[str, float]:
        """
        Estimate memory usage for Bloom filters

        Returns:
            Dictionary with memory estimates in MB
        """
        # Formula: m = -n * ln(p) / (ln(2)^2)
        # where m = bits, n = capacity, p = error rate
        import math

        def bits_to_mb(bits):
            return bits / 8 / 1024 / 1024

        pattern_bits = (-self.pattern_capacity * math.log(self.pattern_error_rate) /
                       (math.log(2) ** 2))
        event_bits = (-self.event_capacity * math.log(self.event_error_rate) /
                     (math.log(2) ** 2))

        return {
            'pattern_bloom_mb': bits_to_mb(pattern_bits),
            'event_bloom_mb': bits_to_mb(event_bits),
            'total_mb': bits_to_mb(pattern_bits + event_bits)
        }

    def __str__(self) -> str:
        """String representation of configuration"""
        memory = self.get_bloom_memory_estimate()
        return f"""BloomConfig:
  Pattern Bloom: {self.pattern_capacity:,} capacity, {self.pattern_error_rate:.1%} error ({memory['pattern_bloom_mb']:.2f} MB)
  Event Bloom: {self.event_capacity:,} capacity, {self.event_error_rate:.2%} error ({memory['event_bloom_mb']:.2f} MB)
  Total Memory: {memory['total_mb']:.2f} MB (budget: {self.max_memory_mb} MB)
  N-gram range: {self.ngram_range[0]}-{self.ngram_range[1]}
  Pattern threshold: {self.pattern_match_threshold} matches
"""


if __name__ == "__main__":
    # Test configuration
    config = BloomConfig()
    print(config)
    print("\nMemory estimates:", config.get_bloom_memory_estimate())
