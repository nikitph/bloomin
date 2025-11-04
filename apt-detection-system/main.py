#!/usr/bin/env python3
"""
APT Detection System - Main Entry Point

Bloom Filter-Based Advanced Persistent Threat Detection System

Usage:
    python main.py [options]

Options:
    --config PATH          Path to config file (default: config.yaml)
    --logs PATH           Path to log file (default: data/test_logs/sample_logs.json)
    --output PATH         Output directory (default: data/results)
    --signatures PATH     Path to CVE signatures (default: data/signatures/cve_signatures.json)
    --verbose            Enable verbose output
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.bloom.bloom_config import BloomConfig
from src.signatures.cve_loader import CVELoader
from src.processors.log_processor import LogProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="APT Detection System - Bloom Filter-Based Multi-Stage Attack Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process sample logs with default configuration
    python main.py

    # Process custom log file
    python main.py --logs /path/to/logs.json

    # Use custom configuration
    python main.py --config custom_config.yaml

    # Verbose output
    python main.py --verbose
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--logs',
        type=str,
        default='data/test_logs/sample_logs.json',
        help='Path to log file (default: data/test_logs/sample_logs.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/results',
        help='Output directory for results (default: data/results)'
    )

    parser.add_argument(
        '--signatures',
        type=str,
        default='data/signatures/cve_signatures.json',
        help='Path to CVE signatures file (default: data/signatures/cve_signatures.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           APT DETECTION SYSTEM - PROOF OF CONCEPT                    ║
║                                                                      ║
║     Dual-Layer Bloom Filter for Multi-Stage Attack Detection        ║
║                                                                      ║
║  Innovation: Unlimited temporal correlation with constant memory    ║
║  Target: 222K logs/sec | Memory: 720 MB | Correlation: Days-Months  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    args = parse_args()

    print_banner()

    # Resolve paths
    base_path = Path(__file__).parent
    config_path = base_path / args.config
    logs_path = base_path / args.logs
    signatures_path = base_path / args.signatures
    output_path = base_path / args.output

    # Verify paths
    print("\nConfiguration:")
    print("-" * 70)
    print(f"  Config:      {config_path}")
    print(f"  Signatures:  {signatures_path}")
    print(f"  Logs:        {logs_path}")
    print(f"  Output:      {output_path}")
    print("-" * 70)

    if not signatures_path.exists():
        print(f"\n❌ Error: Signature file not found: {signatures_path}")
        sys.exit(1)

    if not logs_path.exists():
        print(f"\n❌ Error: Log file not found: {logs_path}")
        sys.exit(1)

    # Load configuration
    print("\nLoading configuration...")
    if config_path.exists():
        config = BloomConfig(str(config_path))
    else:
        print(f"⚠️  Config file not found, using defaults")
        config = BloomConfig()

    if args.verbose:
        print(config)

    # Load CVE signatures
    print(f"\nLoading CVE signatures from {signatures_path}...")
    cve_loader = CVELoader(str(signatures_path))

    if args.verbose:
        print(f"\n{cve_loader}")
        stats = cve_loader.get_stats()
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"  Avg patterns per CVE: {stats['avg_patterns_per_cve']:.1f}")

    # Initialize processor
    print("\n" + "="*70)
    processor = LogProcessor(cve_loader, config)

    # Process logs
    print("\n" + "="*70)
    print("PROCESSING LOGS")
    print("="*70)

    try:
        alerts = processor.process_logs_from_file(str(logs_path))

        # Print detailed alerts
        if alerts:
            print("\n" + "="*70)
            print("GENERATED ALERTS")
            print("="*70)

            for alert in alerts:
                print(f"\n{alert.get_summary()}")
                if args.verbose:
                    print("\nAttack Chain:")
                    for stage in alert.attack_chain:
                        print(f"  Stage {stage['stage']}: {stage['technique_name']} ({stage['technique']})")
                        print(f"    Time: {stage['timestamp']}")
                        print(f"    Source: {stage['source_ip']} → {stage['target']}")
                    print(f"\nRecommendation: {alert.recommendation}")
                    print("-" * 70)

        # Print statistics
        processor.print_stats()

        # Export results
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)
        processor.export_results(str(output_path))

        # Success summary
        print("\n" + "="*70)
        print("SUCCESS SUMMARY")
        print("="*70)
        print(f"✓ Processed {processor.metrics.metrics['logs_processed']} logs")
        print(f"✓ Generated {len(alerts)} alerts")
        print(f"✓ Detected {processor.event_bloom.stats['multi_stage_detected']} multi-stage attacks")
        print(f"✓ Identified {processor.event_bloom.stats['apt_campaigns_detected']} APT campaigns")
        print(f"✓ Average processing time: {processor.metrics.get_avg_processing_time_us():.2f} μs per log")
        print(f"✓ Throughput: {processor.metrics.get_throughput():.2f} logs/sec")
        print(f"✓ Memory usage: {processor.metrics.get_current_memory_mb():.2f} MB")
        print("="*70 + "\n")

        # Check if we met performance targets
        avg_time = processor.metrics.get_avg_processing_time_us()
        memory = processor.metrics.get_max_memory_mb()

        print("Performance vs Targets:")
        print(f"  Processing time: {avg_time:.2f} μs {'✓' if avg_time < 5 else '✗'} (target: <5 μs)")
        print(f"  Throughput: {processor.metrics.get_throughput():.0f} logs/sec {'✓' if processor.metrics.get_throughput() > 200000 else '✗'} (target: >200K)")
        print(f"  Memory: {memory:.2f} MB {'✓' if memory < 720 else '✗'} (target: <720 MB)")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
