#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v3_detection.py
--------------------------------------------------------
Run V3 APT detection system on large test datasets
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from advanced.adalog_bloom_temporal_v3 import CompositeEngine


class V3TestRunner:
    """Run V3 system on test dataset and analyze results"""

    def __init__(self, input_file: str):
        self.input_file = input_file

        # Define threat families to detect
        families = [
            "CVE-2024-6387",  # regresshion SSH exploit
            "reconnaissance",
            "exploitation",
            "privilege_escalation",
            "persistence",
            "lateral_movement",
            "exfiltration"
        ]

        self.engine = CompositeEngine(families)
        self.stats = {
            'total_logs': 0,
            'alerts_generated': 0,
            'campaigns_detected': set(),
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
        self.alerts = []

    def process_dataset(self):
        """Process entire dataset through V3 system"""
        print(f"Processing dataset: {self.input_file}")
        print("=" * 70)

        start = time.time()

        with open(self.input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    log = json.loads(line)
                    self.stats['total_logs'] += 1

                    # Track time range
                    if 'timestamp' in log:
                        ts = datetime.fromisoformat(log['timestamp'])
                        if self.stats['start_time'] is None or ts < self.stats['start_time']:
                            self.stats['start_time'] = ts
                        if self.stats['end_time'] is None or ts > self.stats['end_time']:
                            self.stats['end_time'] = ts

                    # Ensure required fields for V3 engine
                    if 'id' not in log:
                        log['id'] = f"log_{line_num}"
                    if 'entity' not in log:
                        log['entity'] = log.get('host', 'unknown')
                    if 'ts' not in log:
                        log['ts'] = int(ts.timestamp()) if 'timestamp' in log else int(time.time())
                    if 'text' not in log:
                        log['text'] = str(log)  # Convert entire log to text
                    if 'cred_hash' not in log:
                        log['cred_hash'] = log.get('user', '')
                    if 'asn' not in log:
                        log['asn'] = ''
                    if 'src_ip' not in log:
                        log['src_ip'] = log.get('source_ip', '')

                    # Process through V3 engine
                    result = self.engine.ingest(log)

                    # Check for alerts/campaigns
                    if result and 'campaigns' in result:
                        self.stats['alerts_generated'] += 1
                        self.alerts.append({
                            'log_num': line_num,
                            'timestamp': log.get('timestamp'),
                            'result': result
                        })

                    # Progress reporting
                    if line_num % 10000 == 0:
                        elapsed = time.time() - start
                        rate = line_num / elapsed if elapsed > 0 else 0
                        print(f"  Processed {line_num:,} logs ({rate:.0f} logs/sec)")

                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num}")
                    continue

        self.stats['processing_time'] = time.time() - start

        # Analyze campaigns
        self._analyze_campaigns()

    def _analyze_campaigns(self):
        """Analyze detected campaigns using graph analysis"""
        print("\n" + "=" * 70)
        print("Campaign Analysis")
        print("=" * 70)

        # Get final campaign components from graph
        if self.engine.tempo is None:
            print("   ⚠️  Graph analysis not available (networkx required)")
            return

        campaigns = self.engine.tempo.campaigns()

        print(f"\n📊 **Campaign Components Detected**: {len(campaigns)}")

        for i, component in enumerate(campaigns, 1):
            events = [self.engine.tempo.G.nodes[node_id] for node_id in component]

            # Extract details
            event_ids = [e['id'] for e in events]
            actors = set(e.get('actor', 'unknown') for e in events)
            entities = set(e.get('entity', 'unknown') for e in events)
            timestamps = sorted(e['ts'] for e in events)

            # Time span
            if len(timestamps) > 1:
                time_span = (timestamps[-1] - timestamps[0]) / 86400  # days
            else:
                time_span = 0

            print(f"\n🎯 **Campaign {i}**:")
            print(f"   Events: {len(events)}")
            print(f"   Event IDs: {event_ids[:10]}{'...' if len(event_ids) > 10 else ''}")
            print(f"   Actors: {actors}")
            print(f"   Entities: {entities}")
            print(f"   Time span: {time_span:.1f} days")

            # Try to score the campaign
            if len(events) >= 3:
                # Build campaign data for scorer
                campaign_data = {
                    'event_ids': event_ids,
                    'semantic': {'high': len([e for e in events if e.get('semantic_conf', 0) > 0.7])},
                    'temporal': {'span_days': time_span, 'num_edges': len(component) - 1},
                    'hosts': {'unique': len(entities)},
                    'actor': {'shared': len(actors) == 1},
                    'diversity': {'stages': len(set(e.get('stage', 'unknown') for e in events))},
                    'recovery': {'ids': event_ids}
                }

                score_result = self.engine.scorer.score(campaign_data)
                severity = score_result['severity']
                confidence = score_result['confidence']

                print(f"   🚨 Severity: {severity} (confidence: {confidence:.2f})")

    def print_summary(self):
        """Print detection summary"""
        print("\n" + "=" * 70)
        print("🛡️ **CVE-2024-6387 Detection Summary**")
        print("=" * 70)

        print(f"\n📊 **Processing Statistics**:")
        print(f"   Total logs processed: {self.stats['total_logs']:,}")
        print(f"   Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"   Throughput: {self.stats['total_logs'] / self.stats['processing_time']:.0f} logs/sec")

        if self.stats['start_time'] and self.stats['end_time']:
            time_span = (self.stats['end_time'] - self.stats['start_time']).total_seconds() / 86400
            print(f"   Log time range: {self.stats['start_time']} to {self.stats['end_time']}")
            print(f"   Campaign duration: {time_span:.1f} days")

        print(f"\n🔍 **Detection Results**:")
        print(f"   Alerts generated: {self.stats['alerts_generated']}")

        # Campaign analysis from graph
        if self.engine.tempo:
            campaigns = self.engine.tempo.campaigns()
            print(f"   Campaign components: {len(campaigns)}")
        else:
            print(f"   Campaign components: N/A (networkx required)")

        # Analyze attack indicators
        self._print_attack_indicators()

    def _print_attack_indicators(self):
        """Analyze and print specific attack indicators found"""
        print(f"\n🧠 **Attack Indicators Detected**:")

        # Analyze all events in graph
        indicators = {
            'sshd_crashes': [],
            'file_modifications': [],
            'privilege_escalation': [],
            'persistence': [],
            'lateral_movement': [],
            'exfiltration': []
        }

        if not self.engine.tempo:
            print(f"   ⚠️  Attack indicator analysis requires networkx")
            return

        for node_id, data in self.engine.tempo.G.nodes(data=True):
            text = data.get('text', '').lower()

            if 'sshd' in text and ('crash' in text or 'sigalrm' in text or 'segfault' in text):
                indicators['sshd_crashes'].append(node_id)
            elif 'passwd' in text or 'libc' in text or 'shadow' in text:
                indicators['file_modifications'].append(node_id)
            elif 'uid=0' in text or 'root' in text and 'shell' in text:
                indicators['privilege_escalation'].append(node_id)
            elif 'cron' in text or 'ld_preload' in text:
                indicators['persistence'].append(node_id)
            elif 'nmap' in text and 'internal' in text:
                indicators['lateral_movement'].append(node_id)
            elif 'reverse shell' in text or 'outbound' in text:
                indicators['exfiltration'].append(node_id)

        if indicators['sshd_crashes']:
            print(f"   🔸 SSHD crashes/SIGALRM: {len(indicators['sshd_crashes'])} events")

        if indicators['file_modifications']:
            print(f"   🔸 File modifications (/etc/passwd, libc): {len(indicators['file_modifications'])} events")

        if indicators['privilege_escalation']:
            print(f"   🔸 Privilege escalation (uid=0): {len(indicators['privilege_escalation'])} events")

        if indicators['persistence']:
            print(f"   🔸 Persistence mechanisms: {len(indicators['persistence'])} events")

        if indicators['lateral_movement']:
            print(f"   🔸 Lateral movement: {len(indicators['lateral_movement'])} events")

        if indicators['exfiltration']:
            print(f"   🔸 Data exfiltration: {len(indicators['exfiltration'])} events")

        total_indicators = sum(len(v) for v in indicators.values())
        if total_indicators == 0:
            print(f"   ⚠️  No specific attack indicators detected")

    def generate_report(self, output_file: str = None):
        """Generate detailed detection report"""
        if output_file is None:
            output_file = self.input_file.replace('.json', '_detection_report.txt')

        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CVE-2024-6387 DETECTION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Total logs: {self.stats['total_logs']:,}\n")
            f.write(f"Processing time: {self.stats['processing_time']:.2f}s\n\n")

            # Campaign details
            if not self.engine.tempo:
                f.write("Campaign components: N/A (networkx required)\n")
                return

            campaigns = self.engine.tempo.campaigns()
            f.write(f"Campaign components detected: {len(campaigns)}\n\n")

            for i, component in enumerate(campaigns, 1):
                events = [self.engine.tempo.G.nodes[node_id] for node_id in component]
                f.write(f"\nCampaign {i}:\n")
                f.write(f"  Events: {len(events)}\n")
                for event in events[:20]:  # Limit to first 20
                    f.write(f"    - {event.get('id', 'unknown')}: {event.get('text', '')[:100]}\n")
                if len(events) > 20:
                    f.write(f"    ... and {len(events) - 20} more events\n")

        print(f"\n📄 Report saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run V3 detection on test dataset')
    parser.add_argument('--input', '-i', required=True,
                        help='Input log file (JSON lines format)')
    parser.add_argument('--report', '-r', default=None,
                        help='Output report file (default: auto-generated)')

    args = parser.parse_args()

    # Run detection
    runner = V3TestRunner(args.input)
    runner.process_dataset()
    runner.print_summary()
    runner.generate_report(args.report)


if __name__ == '__main__':
    main()
