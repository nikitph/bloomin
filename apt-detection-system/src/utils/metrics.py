"""
Performance Metrics Module

Tracks and reports performance metrics for the APT detection system.
"""

import time
import psutil
import os
from typing import Dict, Any
from datetime import datetime


class PerformanceMetrics:
    """Tracks system performance metrics"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

        self.metrics = {
            'logs_processed': 0,
            'pattern_matches': 0,
            'correlations_found': 0,
            'alerts_generated': 0,
            'processing_times_us': [],
            'memory_samples_mb': []
        }

    def record_log_processed(self, processing_time_us: float):
        """
        Record a log processing event

        Args:
            processing_time_us: Processing time in microseconds
        """
        self.metrics['logs_processed'] += 1
        self.metrics['processing_times_us'].append(processing_time_us)

    def record_pattern_match(self):
        """Record a pattern match event"""
        self.metrics['pattern_matches'] += 1

    def record_correlation(self):
        """Record a correlation found event"""
        self.metrics['correlations_found'] += 1

    def record_alert(self):
        """Record an alert generated event"""
        self.metrics['alerts_generated'] += 1

    def sample_memory(self):
        """Sample current memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics['memory_samples_mb'].append(memory_mb)
        return memory_mb

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

    def get_throughput(self) -> float:
        """
        Get processing throughput in logs/second

        Returns:
            Logs processed per second
        """
        elapsed = self.get_elapsed_time()
        if elapsed > 0:
            return self.metrics['logs_processed'] / elapsed
        return 0.0

    def get_avg_processing_time_us(self) -> float:
        """
        Get average processing time per log in microseconds

        Returns:
            Average processing time
        """
        times = self.metrics['processing_times_us']
        if times:
            return sum(times) / len(times)
        return 0.0

    def get_max_memory_mb(self) -> float:
        """Get maximum memory usage in MB"""
        samples = self.metrics['memory_samples_mb']
        return max(samples) if samples else 0.0

    def get_avg_memory_mb(self) -> float:
        """Get average memory usage in MB"""
        samples = self.metrics['memory_samples_mb']
        return sum(samples) / len(samples) if samples else 0.0

    def get_pattern_match_rate(self) -> float:
        """Get pattern match rate (matches per log)"""
        if self.metrics['logs_processed'] > 0:
            return self.metrics['pattern_matches'] / self.metrics['logs_processed']
        return 0.0

    def get_correlation_rate(self) -> float:
        """Get correlation rate (correlations per log)"""
        if self.metrics['logs_processed'] > 0:
            return self.metrics['correlations_found'] / self.metrics['logs_processed']
        return 0.0

    def get_alert_rate(self) -> float:
        """Get alert rate (alerts per log)"""
        if self.metrics['logs_processed'] > 0:
            return self.metrics['alerts_generated'] / self.metrics['logs_processed']
        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary

        Returns:
            Dictionary with all metrics
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'elapsed_time_sec': self.get_elapsed_time(),
            'logs_processed': self.metrics['logs_processed'],
            'pattern_matches': self.metrics['pattern_matches'],
            'correlations_found': self.metrics['correlations_found'],
            'alerts_generated': self.metrics['alerts_generated'],
            'throughput_logs_per_sec': self.get_throughput(),
            'avg_processing_time_us': self.get_avg_processing_time_us(),
            'pattern_match_rate': self.get_pattern_match_rate(),
            'correlation_rate': self.get_correlation_rate(),
            'alert_rate': self.get_alert_rate(),
            'current_memory_mb': self.get_current_memory_mb(),
            'max_memory_mb': self.get_max_memory_mb(),
            'avg_memory_mb': self.get_avg_memory_mb()
        }

    def print_summary(self):
        """Print metrics summary to console"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)

        print(f"\nProcessing:")
        print(f"  Logs processed:      {summary['logs_processed']:,}")
        print(f"  Elapsed time:        {summary['elapsed_time_sec']:.2f} sec")
        print(f"  Throughput:          {summary['throughput_logs_per_sec']:.2f} logs/sec")
        print(f"  Avg processing time: {summary['avg_processing_time_us']:.2f} μs")

        print(f"\nDetection:")
        print(f"  Pattern matches:     {summary['pattern_matches']:,} ({summary['pattern_match_rate']:.2%} of logs)")
        print(f"  Correlations found:  {summary['correlations_found']:,} ({summary['correlation_rate']:.2%} of logs)")
        print(f"  Alerts generated:    {summary['alerts_generated']:,} ({summary['alert_rate']:.2%} of logs)")

        print(f"\nMemory:")
        print(f"  Current:             {summary['current_memory_mb']:.2f} MB")
        print(f"  Average:             {summary['avg_memory_mb']:.2f} MB")
        print(f"  Maximum:             {summary['max_memory_mb']:.2f} MB")

        print("="*60 + "\n")

    def reset(self):
        """Reset all metrics"""
        self.start_time = time.time()
        self.metrics = {
            'logs_processed': 0,
            'pattern_matches': 0,
            'correlations_found': 0,
            'alerts_generated': 0,
            'processing_times_us': [],
            'memory_samples_mb': []
        }


class BenchmarkTimer:
    """Context manager for timing code blocks"""

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer

        Args:
            name: Name of operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed_us = None

    def __enter__(self):
        """Start timer"""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timer"""
        self.elapsed_us = (time.perf_counter() - self.start_time) * 1_000_000

    def __repr__(self):
        if self.elapsed_us is not None:
            return f"{self.name}: {self.elapsed_us:.2f} μs"
        return f"{self.name}: not completed"


if __name__ == "__main__":
    # Test metrics tracking
    metrics = PerformanceMetrics()

    # Simulate processing
    print("Simulating log processing...")
    for i in range(1000):
        with BenchmarkTimer("Log processing") as timer:
            time.sleep(0.0001)  # Simulate work

        metrics.record_log_processed(timer.elapsed_us)

        if i % 100 == 0:
            metrics.record_pattern_match()

        if i % 200 == 0:
            metrics.record_correlation()

        if i % 500 == 0:
            metrics.record_alert()
            metrics.sample_memory()

    # Print summary
    metrics.print_summary()

    # Test benchmark timer
    print("\nBenchmark Timer Test:")
    with BenchmarkTimer("Test operation") as timer:
        sum(range(10000))
    print(timer)
