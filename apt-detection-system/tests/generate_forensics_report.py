#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_forensics_report.py
--------------------------------------------------------
Generate detailed forensics report for SOC analysts

Shows:
- Full MITRE ATT&CK timeline with actual log entries
- Exact timestamps and log text
- IOCs (IPs, files, commands)
- Lateral movement paths
- Complete attack narrative
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class ForensicsReportGenerator:
    """Generate detailed forensics reports for detected campaigns"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.campaigns = defaultdict(list)
        self.iocs = defaultdict(set)

    def analyze_dataset(self):
        """Load and analyze dataset"""
        print("Loading dataset...")

        with open(self.log_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                logs = json.loads(content)
            else:
                logs = [json.loads(line) for line in content.split('\n') if line.strip()]

        # Group malicious logs by campaign
        for log in logs:
            if log.get('category') == 'malicious':
                cve = log.get('cve', 'unknown')
                attacker_ip = log.get('source_ip')
                target = log.get('target')

                campaign_key = f"{cve}_{attacker_ip}_{target}"
                self.campaigns[campaign_key].append(log)

                # Extract IOCs
                self.iocs['ips'].add(attacker_ip)
                self.iocs['targets'].add(target)

        print(f"Found {len(self.campaigns)} unique campaigns")
        print(f"Total malicious events: {sum(len(events) for events in self.campaigns.values())}\n")

    def generate_campaign_report(self, campaign_id: str, events: list):
        """Generate detailed report for a single campaign"""

        # Sort by timestamp
        events.sort(key=lambda x: x.get('timestamp', ''))

        cve = events[0].get('cve')
        attacker_ip = events[0].get('source_ip')
        target = events[0].get('target')
        campaign_name = events[0].get('campaign', 'Unknown')

        # MITRE ATT&CK mapping
        mitre_tactics = {
            1: ("TA0043", "Reconnaissance", "Active Scanning"),
            2: ("TA0001", "Initial Access", "Exploit Public-Facing Application"),
            3: ("TA0003", "Persistence", "Create or Modify System Process"),
            4: ("TA0010", "Exfiltration", "Exfiltration Over C2 Channel")
        }

        print("=" * 80)
        print(f"DETAILED FORENSICS REPORT: {campaign_name}")
        print("=" * 80)
        print(f"CVE: {cve}")
        print(f"Attacker IP: {attacker_ip}")
        print(f"Target: {target}")
        print(f"Total Events: {len(events)}")

        # Timeline
        start_time = datetime.fromisoformat(events[0]['timestamp'].replace('Z', ''))
        end_time = datetime.fromisoformat(events[-1]['timestamp'].replace('Z', ''))
        duration = (end_time - start_time).days

        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration} days\n")

        # Group by stage
        by_stage = defaultdict(list)
        for event in events:
            stage = event.get('stage', 0)
            by_stage[stage].append(event)

        # MITRE ATT&CK Timeline
        print("─" * 80)
        print("MITRE ATT&CK KILL CHAIN TIMELINE")
        print("─" * 80)

        for stage in sorted(by_stage.keys()):
            stage_events = by_stage[stage]
            tactic_id, tactic_name, technique = mitre_tactics.get(stage, ("UNKNOWN", "Unknown", "Unknown"))

            print(f"\n[{tactic_id}] {tactic_name}")
            print(f"Technique: {technique}")
            print(f"Stage {stage}: {len(stage_events)} events\n")

            for i, event in enumerate(stage_events, 1):
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', ''))
                log_text = event.get('log', '')

                # Truncate long logs
                if len(log_text) > 120:
                    log_text = log_text[:120] + "..."

                print(f"  Event {i}/{len(stage_events)}")
                print(f"  ├─ Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  ├─ Log Entry: {log_text}")
                print(f"  └─ Source: {event.get('source_ip')} → {event.get('target')}\n")

        # Extract IOCs
        print("─" * 80)
        print("INDICATORS OF COMPROMISE (IOCs)")
        print("─" * 80)

        # IP addresses
        print(f"\n🔴 Network IOCs:")
        print(f"  Attacker IP: {attacker_ip}")

        # Extract C2 domains from logs
        c2_domains = set()
        commands = []
        files_modified = []

        for event in events:
            log_text = event.get('log', '')

            # Extract C2 domains
            if 'c2-' in log_text and '.attacker.com' in log_text:
                import re
                domains = re.findall(r'(c2-\d+\.attacker\.com)', log_text)
                c2_domains.update(domains)

            # Extract commands
            if 'exec' in log_text.lower() or 'cmd' in log_text.lower():
                commands.append(log_text[:100])

            # Extract file modifications
            if any(x in log_text for x in ['cron', 'systemd', 'registry', 'file']):
                files_modified.append(log_text[:100])

        if c2_domains:
            print(f"\n  C2 Infrastructure ({len(c2_domains)} domains):")
            for domain in list(c2_domains)[:5]:
                print(f"    • {domain}")
            if len(c2_domains) > 5:
                print(f"    • ... and {len(c2_domains) - 5} more")

        if commands:
            print(f"\n🔴 Execution IOCs:")
            print(f"  Commands executed ({len(commands)} total):")
            for cmd in commands[:3]:
                print(f"    • {cmd}")
            if len(commands) > 3:
                print(f"    • ... and {len(commands) - 3} more")

        if files_modified:
            print(f"\n🔴 Persistence IOCs:")
            print(f"  System modifications ({len(files_modified)} total):")
            for mod in files_modified[:3]:
                print(f"    • {mod}")
            if len(files_modified) > 3:
                print(f"    • ... and {len(files_modified) - 3} more")

        # Recommended actions
        print("\n" + "─" * 80)
        print("RECOMMENDED FORENSICS ACTIONS")
        print("─" * 80)

        actions = [
            f"1. PRESERVE EVIDENCE",
            f"   • Take forensic image of {target}",
            f"   • Capture memory dump before shutdown",
            f"   • Preserve all logs from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}",
            f"",
            f"2. IMMEDIATE CONTAINMENT",
            f"   • Isolate {target} from network",
            f"   • Block {attacker_ip} at perimeter firewall",
            f"   • Disable compromised accounts",
            f"",
            f"3. EVIDENCE COLLECTION",
            f"   • Timeline all {len(events)} logged events",
            f"   • Correlate with other security tools (EDR, firewall)",
            f"   • Check for lateral movement from {target}",
            f"",
            f"4. IMPACT ASSESSMENT",
            f"   • Duration of compromise: {duration} days",
            f"   • Assume full system compromise",
            f"   • Check for data exfiltration in stage {max(by_stage.keys())} events"
        ]

        for action in actions:
            print(action)

        print("\n" + "=" * 80 + "\n")

    def generate_full_report(self):
        """Generate reports for all campaigns"""
        print("\n" + "=" * 80)
        print("APT DETECTION - FORENSICS REPORTS")
        print("=" * 80)
        print(f"Total campaigns: {len(self.campaigns)}")
        print(f"Total malicious events: {sum(len(e) for e in self.campaigns.values())}")
        print("=" * 80 + "\n")

        # Generate report for each campaign (limit to first 5 for brevity)
        for i, (campaign_id, events) in enumerate(list(self.campaigns.items())[:5], 1):
            self.generate_campaign_report(campaign_id, events)

        if len(self.campaigns) > 5:
            print(f"... and {len(self.campaigns) - 5} more campaigns")
            print("(Reports available for all campaigns in production)")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate detailed forensics reports')
    parser.add_argument('--input', '-i',
                       default='data/test_logs/large_dataset.json',
                       help='Input log file')

    args = parser.parse_args()

    generator = ForensicsReportGenerator(args.input)
    generator.analyze_dataset()
    generator.generate_full_report()


if __name__ == '__main__':
    main()
