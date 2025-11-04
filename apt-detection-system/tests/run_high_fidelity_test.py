#!/usr/bin/env python3
"""
High-Fidelity Performance Test Runner

Tests the APT detection system with large-scale realistic data (5-10GB).

Features:
- Streaming JSON processing for memory efficiency
- Real-time performance metrics
- Campaign detection validation
- Memory profiling
- Comprehensive results report
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.bloom.bloom_config import BloomConfig
from src.signatures.cve_loader import CVELoader
from src.processors.log_processor import LogProcessor


class StreamingJSONReader:
    """Memory-efficient streaming JSON array reader"""

    def __init__(self, file_path: str):
        """Initialize streaming reader"""
        self.file_path = file_path
        self.file_size = Path(file_path).stat().st_size

    def read_logs(self):
        """
        Stream logs from JSON file one at a time

        Yields:
            Log entry dictionary
        """
        with open(self.file_path, 'r') as f:
            # Skip opening bracket
            line = f.readline()
            if line.strip() != '[':
                raise ValueError("Expected JSON array")

            buffer = ""
            depth = 0
            in_string = False
            escape = False

            while True:
                char = f.read(1)
                if not char:
                    break

                # Track string boundaries
                if char == '"' and not escape:
                    in_string = not in_string

                # Track escape sequences
                escape = (char == '\\' and not escape)

                if not in_string:
                    if char == '{':
                        depth += 1
                        buffer += char
                    elif char == '}':
                        buffer += char
                        depth -= 1

                        if depth == 0:
                            # Complete object found
                            try:
                                log = json.loads(buffer)
                                yield log
                            except json.JSONDecodeError:
                                pass  # Skip malformed entries
                            buffer = ""
                    elif depth > 0:
                        buffer += char
                else:
                    if depth > 0:
                        buffer += char


def run_high_fidelity_test(
    dataset_path: str,
    config_path: str = None,
    signatures_path: str = None,
    sample_size: int = None,
    progress_interval: int = 10000
):
    """
    Run high-fidelity performance test

    Args:
        dataset_path: Path to large test dataset
        config_path: Path to configuration file
        signatures_path: Path to CVE signatures
        sample_size: Number of logs to process (None = all)
        progress_interval: Report progress every N logs
    """
    print(f"\n{'='*70}")
    print(f"HIGH-FIDELITY PERFORMANCE TEST")
    print(f"{'='*70}")

    # Check if dataset exists
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print(f"\nGenerate it first:")
        print(f"  python tests/generate_large_dataset.py --size 5.0 --campaigns 50")
        sys.exit(1)

    # Get file size
    file_size_gb = dataset_file.stat().st_size / 1024 / 1024 / 1024
    print(f"\nDataset: {dataset_path}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"{'='*70}\n")

    # Initialize system
    base_path = Path(__file__).parent.parent

    if config_path:
        config = BloomConfig(config_path)
    else:
        config = BloomConfig(str(base_path / "config.yaml"))

    if signatures_path is None:
        signatures_path = base_path / "data/signatures/cve_signatures.json"

    print("Initializing APT Detection System...")
    cve_loader = CVELoader(str(signatures_path))
    processor = LogProcessor(cve_loader, config)
    print()

    # Initialize streaming reader
    print("Starting log processing...")
    print(f"Progress updates every {progress_interval:,} logs")
    print(f"{'='*70}\n")

    reader = StreamingJSONReader(dataset_path)

    # Performance tracking
    start_time = time.time()
    logs_processed = 0
    last_progress_time = start_time
    last_progress_count = 0

    alerts_by_tier = {1: 0, 2: 0, 3: 0}
    apt_campaigns_detected = set()
    ground_truth_malicious = 0
    detected_malicious = 0

    # Process logs
    try:
        for log in reader.read_logs():
            # Track ground truth
            if log.get('category') == 'malicious':
                ground_truth_malicious += 1

            # Process log
            alert = processor.process_log(log)

            if alert:
                detected_malicious += 1
                alerts_by_tier[alert.tier] += 1

                if alert.tier == 3:
                    apt_campaigns_detected.add(alert.correlation_key)

            logs_processed += 1

            # Progress reporting
            if logs_processed % progress_interval == 0:
                current_time = time.time()
                elapsed = current_time - last_progress_time
                logs_since_last = logs_processed - last_progress_count

                throughput = logs_since_last / elapsed if elapsed > 0 else 0
                total_elapsed = current_time - start_time
                overall_throughput = logs_processed / total_elapsed if total_elapsed > 0 else 0

                memory_mb = processor.metrics.get_current_memory_mb()

                print(f"[{logs_processed:,} logs] "
                      f"Throughput: {throughput:.0f} logs/sec "
                      f"(avg: {overall_throughput:.0f}) | "
                      f"Memory: {memory_mb:.1f} MB | "
                      f"Alerts: {sum(alerts_by_tier.values())} "
                      f"(T1:{alerts_by_tier[1]} T2:{alerts_by_tier[2]} T3:{alerts_by_tier[3]})")

                last_progress_time = current_time
                last_progress_count = logs_processed

                # Sample memory
                processor.metrics.sample_memory()

            # Check sample size limit
            if sample_size and logs_processed >= sample_size:
                print(f"\nReached sample size limit: {sample_size:,} logs")
                break

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    # Calculate final statistics
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")

    # Print comprehensive results
    print(f"\nProcessing Summary:")
    print(f"  Total logs processed: {logs_processed:,}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average throughput: {logs_processed / total_time:.2f} logs/sec")
    print(f"  Average latency: {processor.metrics.get_avg_processing_time_us():.2f} μs per log")

    print(f"\nMemory Usage:")
    print(f"  Current: {processor.metrics.get_current_memory_mb():.2f} MB")
    print(f"  Maximum: {processor.metrics.get_max_memory_mb():.2f} MB")
    print(f"  Average: {processor.metrics.get_avg_memory_mb():.2f} MB")
    print(f"  Budget: {config.max_memory_mb} MB")
    print(f"  Within budget: {'✓' if processor.metrics.get_max_memory_mb() < config.max_memory_mb else '✗'}")

    print(f"\nDetection Results:")
    print(f"  Ground truth malicious logs: {ground_truth_malicious:,}")
    print(f"  Detected malicious events: {detected_malicious:,}")
    print(f"  Total alerts generated: {sum(alerts_by_tier.values()):,}")
    print(f"    Tier 1 (LOW): {alerts_by_tier[1]:,}")
    print(f"    Tier 2 (HIGH): {alerts_by_tier[2]:,}")
    print(f"    Tier 3 (CRITICAL): {alerts_by_tier[3]:,}")
    print(f"  APT campaigns detected: {len(apt_campaigns_detected)}")

    # Detection accuracy
    if ground_truth_malicious > 0:
        detection_rate = (detected_malicious / ground_truth_malicious) * 100
        print(f"  Detection rate: {detection_rate:.1f}%")

    print(f"\nBloom Filter Statistics:")
    pattern_stats = processor.pattern_bloom.get_performance_stats()
    event_stats = processor.event_bloom.get_performance_stats()

    print(f"  Pattern Bloom (Layer 1):")
    print(f"    Checks: {pattern_stats['total_checks']:,}")
    print(f"    Matches: {pattern_stats['total_matches']:,}")
    print(f"    Match rate: {pattern_stats['match_rate']:.2%}")
    print(f"    Avg time: {pattern_stats['avg_time_us']:.2f} μs")

    print(f"  Event Bloom (Layer 2):")
    print(f"    Events: {event_stats['events_processed']:,}")
    print(f"    New campaigns: {event_stats['new_campaigns']:,}")
    print(f"    Multi-stage: {event_stats['multi_stage_detected']:,}")
    print(f"    APT campaigns: {event_stats['apt_campaigns_detected']:,}")
    print(f"    Avg time: {event_stats['avg_time_us']:.2f} μs")

    print(f"\nPerformance vs Targets:")
    avg_time = processor.metrics.get_avg_processing_time_us()
    throughput = logs_processed / total_time
    max_memory = processor.metrics.get_max_memory_mb()

    print(f"  Throughput: {throughput:.0f} logs/sec {'✓' if throughput > 200000 else '✗'} (target: >200K)")
    print(f"  Latency: {avg_time:.2f} μs {'✓' if avg_time < 5 else '✗'} (target: <5 μs)")
    print(f"  Memory: {max_memory:.2f} MB {'✓' if max_memory < 720 else '✗'} (target: <720 MB)")

    print(f"\n{'='*70}")

    # Save detailed results
    results_path = Path(__file__).parent.parent / "data/results/high_fidelity_results.json"
    results = {
        'test_info': {
            'dataset': dataset_path,
            'dataset_size_gb': file_size_gb,
            'logs_processed': logs_processed,
            'duration_seconds': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'performance': {
            'throughput_logs_per_sec': logs_processed / total_time,
            'avg_latency_us': processor.metrics.get_avg_processing_time_us(),
            'max_memory_mb': processor.metrics.get_max_memory_mb(),
            'avg_memory_mb': processor.metrics.get_avg_memory_mb()
        },
        'detection': {
            'ground_truth_malicious': ground_truth_malicious,
            'detected_malicious': detected_malicious,
            'total_alerts': sum(alerts_by_tier.values()),
            'tier1_alerts': alerts_by_tier[1],
            'tier2_alerts': alerts_by_tier[2],
            'tier3_alerts': alerts_by_tier[3],
            'apt_campaigns_detected': len(apt_campaigns_detected)
        },
        'bloom_filters': {
            'pattern_bloom': pattern_stats,
            'event_bloom': event_stats
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {results_path}")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run high-fidelity performance test on large dataset'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/test_logs/large_dataset.json',
        help='Path to test dataset (default: data/test_logs/large_dataset.json)'
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
        '--sample',
        type=int,
        default=None,
        help='Process only first N logs (for quick testing)'
    )

    parser.add_argument(
        '--progress',
        type=int,
        default=10000,
        help='Progress update interval (default: 10000 logs)'
    )

    args = parser.parse_args()

    # Resolve paths
    base_path = Path(__file__).parent.parent
    dataset_path = base_path / args.dataset

    run_high_fidelity_test(
        dataset_path=str(dataset_path),
        config_path=args.config,
        signatures_path=args.signatures,
        sample_size=args.sample,
        progress_interval=args.progress
    )


if __name__ == "__main__":
    main()
