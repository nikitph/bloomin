#!/usr/bin/env python3
"""
Streaming Test Runner

Consumes streaming logs and processes them in real-time through the APT detection system.

Usage:
    # Generate and process stream
    python tests/stream_log_generator.py --rate 1000 --duration 60 | python tests/run_streaming_test.py

    # Process with custom config
    python tests/stream_log_generator.py | python tests/run_streaming_test.py --config config.yaml
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bloom.bloom_config import BloomConfig
from src.signatures.cve_loader import CVELoader
from src.processors.log_processor import LogProcessor


class StreamingProcessor:
    """Processes streaming logs in real-time"""

    def __init__(self, processor: LogProcessor, progress_interval: int = 1000):
        """
        Initialize streaming processor

        Args:
            processor: LogProcessor instance
            progress_interval: Report progress every N logs
        """
        self.processor = processor
        self.progress_interval = progress_interval

        # Tracking
        self.start_time = time.time()
        self.logs_processed = 0
        self.alerts_generated = 0
        self.apt_logs_detected = 0

        self.alerts_by_tier = {1: 0, 2: 0, 3: 0}

    def process_stream(self, input_stream=sys.stdin):
        """
        Process logs from stream

        Args:
            input_stream: Input stream (default: stdin)
        """
        print("Starting real-time log processing...", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print("", file=sys.stderr)

        try:
            for line in input_stream:
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON log
                    log_entry = json.loads(line)

                    # Track ground truth
                    if log_entry.get('category') == 'malicious':
                        self.apt_logs_detected += 1

                    # Process through APT detection
                    alert = self.processor.process_log(log_entry)

                    if alert:
                        self.alerts_generated += 1
                        self.alerts_by_tier[alert.tier] += 1

                        # Print critical alerts immediately
                        if alert.tier == 3:
                            print(f"\n🚨 CRITICAL ALERT: {alert.get_summary()}", file=sys.stderr)

                    self.logs_processed += 1

                    # Progress reporting
                    if self.logs_processed % self.progress_interval == 0:
                        self._report_progress()

                except json.JSONDecodeError:
                    continue  # Skip malformed JSON

        except KeyboardInterrupt:
            print(f"\n\nStream processing interrupted", file=sys.stderr)

        # Final report
        self._report_final()

    def _report_progress(self):
        """Report current progress"""
        elapsed = time.time() - self.start_time
        rate = self.logs_processed / elapsed if elapsed > 0 else 0

        memory_mb = self.processor.metrics.get_current_memory_mb()

        print(f"[{self.logs_processed:,} logs] "
              f"Rate: {rate:.0f} logs/sec | "
              f"Memory: {memory_mb:.1f} MB | "
              f"Alerts: {self.alerts_generated} "
              f"(T1:{self.alerts_by_tier[1]} T2:{self.alerts_by_tier[2]} T3:{self.alerts_by_tier[3]})",
              file=sys.stderr)

    def _report_final(self):
        """Print final statistics"""
        elapsed = time.time() - self.start_time
        rate = self.logs_processed / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}", file=sys.stderr)
        print("STREAMING TEST COMPLETE", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        print(f"\nProcessing Summary:", file=sys.stderr)
        print(f"  Total logs processed: {self.logs_processed:,}", file=sys.stderr)
        print(f"  Ground truth APT logs: {self.apt_logs_detected}", file=sys.stderr)
        print(f"  Total time: {elapsed:.2f} seconds", file=sys.stderr)
        print(f"  Throughput: {rate:.2f} logs/sec", file=sys.stderr)
        print(f"  Avg latency: {self.processor.metrics.get_avg_processing_time_us():.2f} μs", file=sys.stderr)

        print(f"\nDetection Results:", file=sys.stderr)
        print(f"  Total alerts: {self.alerts_generated}", file=sys.stderr)
        print(f"    Tier 1 (LOW): {self.alerts_by_tier[1]}", file=sys.stderr)
        print(f"    Tier 2 (HIGH): {self.alerts_by_tier[2]}", file=sys.stderr)
        print(f"    Tier 3 (CRITICAL): {self.alerts_by_tier[3]}", file=sys.stderr)

        print(f"\nMemory Usage:", file=sys.stderr)
        print(f"  Current: {self.processor.metrics.get_current_memory_mb():.2f} MB", file=sys.stderr)
        print(f"  Maximum: {self.processor.metrics.get_max_memory_mb():.2f} MB", file=sys.stderr)

        # Campaign info
        campaigns = self.processor.event_bloom.get_all_campaigns()
        print(f"\nActive Campaigns: {len(campaigns)}", file=sys.stderr)

        # Show critical campaigns
        critical_campaigns = [c for c in campaigns if len(c['stages']) >= 3]
        if critical_campaigns:
            print(f"\nCritical APT Campaigns Detected: {len(critical_campaigns)}", file=sys.stderr)
            for i, campaign in enumerate(critical_campaigns[:5], 1):
                print(f"  {i}. {campaign['correlation_key']}", file=sys.stderr)
                print(f"     Stages: {campaign['stages']}, Span: {campaign['time_span_days']:.1f} days", file=sys.stderr)

        print(f"\n{'='*60}", file=sys.stderr)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Process streaming logs through APT detection system'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--signatures',
        type=str,
        default=None,
        help='Path to CVE signatures file'
    )

    parser.add_argument(
        '--progress',
        type=int,
        default=1000,
        help='Progress update interval (default: 1000 logs)'
    )

    args = parser.parse_args()

    # Initialize system
    base_path = Path(__file__).parent.parent

    if args.config:
        config = BloomConfig(args.config)
    else:
        config = BloomConfig(str(base_path / "config.yaml"))

    if args.signatures is None:
        signatures_path = base_path / "data/signatures/cve_signatures.json"
    else:
        signatures_path = Path(args.signatures)

    print("Initializing APT Detection System...", file=sys.stderr)
    cve_loader = CVELoader(str(signatures_path))
    processor = LogProcessor(cve_loader, config)
    print("", file=sys.stderr)

    # Create streaming processor
    stream_processor = StreamingProcessor(processor, progress_interval=args.progress)

    # Process stdin
    stream_processor.process_stream(sys.stdin)


if __name__ == "__main__":
    main()
