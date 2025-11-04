"""
Alert Manager Module

Generates tiered alerts based on attack stage progression.

Alert Tiers:
- Tier 1 (LOW): Isolated suspicious event, no prior stages detected
- Tier 2 (HIGH): 2-stage attack detected
- Tier 3 (CRITICAL): 3+ stage attack detected, confirmed APT campaign
"""

import json
import uuid
from typing import List, Dict, Any
from datetime import datetime

from ..bloom.event_bloom import CorrelationResult
from ..utils.fingerprint import EventFingerprint
from ..signatures.mitre_mapper import MITREMapper
from ..bloom.bloom_config import BloomConfig


class Alert:
    """Represents a security alert"""

    def __init__(
        self,
        alert_id: str,
        tier: int,
        severity: str,
        cve_id: str,
        cve_name: str,
        attack_chain: List[Dict[str, Any]],
        time_span_days: float,
        correlation_key: str,
        recommendation: str,
        timestamp: str = None
    ):
        """
        Initialize alert

        Args:
            alert_id: Unique alert identifier
            tier: Alert tier (1, 2, or 3)
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            cve_id: CVE identifier
            cve_name: CVE name
            attack_chain: List of attack stage dictionaries
            time_span_days: Time span of attack in days
            correlation_key: Correlation key for campaign
            recommendation: Recommended response action
            timestamp: Alert timestamp
        """
        self.alert_id = alert_id
        self.tier = tier
        self.severity = severity
        self.cve_id = cve_id
        self.cve_name = cve_name
        self.attack_chain = attack_chain
        self.time_span_days = time_span_days
        self.correlation_key = correlation_key
        self.recommendation = recommendation
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'tier': self.tier,
            'severity': self.severity,
            'cve': self.cve_id,
            'cve_name': self.cve_name,
            'attack_chain': self.attack_chain,
            'time_span_days': round(self.time_span_days, 2),
            'correlation_key': self.correlation_key,
            'recommendation': self.recommendation
        }

    def to_json(self) -> str:
        """Convert alert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def get_summary(self) -> str:
        """Get human-readable alert summary"""
        return (
            f"[TIER {self.tier} - {self.severity}] {self.cve_id} ({self.cve_name}): "
            f"{len(self.attack_chain)}-stage attack detected over {self.time_span_days:.1f} days"
        )

    def __repr__(self) -> str:
        return f"Alert(tier={self.tier}, {self.cve_id}, {len(self.attack_chain)} stages)"


class AlertManager:
    """Manages alert generation and prioritization"""

    def __init__(self, config: BloomConfig = None):
        """
        Initialize alert manager

        Args:
            config: Configuration
        """
        self.config = config or BloomConfig()
        self.mitre_mapper = MITREMapper()

        # Track generated alerts
        self.alerts: List[Alert] = []

        # Campaign alert tracking (prevent duplicate alerts)
        self.campaign_alerts: Dict[str, List[str]] = {}  # correlation_key -> list of alert_ids

        # Statistics
        self.stats = {
            'tier1_alerts': 0,
            'tier2_alerts': 0,
            'tier3_alerts': 0,
            'total_alerts': 0
        }

    def generate_alert(
        self,
        correlation_result: CorrelationResult,
        campaign_events: List[EventFingerprint],
        cve_name: str = ""
    ) -> Alert:
        """
        Generate alert based on correlation result

        Args:
            correlation_result: Correlation result from Event Bloom
            campaign_events: List of all events in campaign
            cve_name: CVE name (optional)

        Returns:
            Generated Alert object
        """
        # Determine tier based on number of stages
        num_stages = correlation_result.num_stages
        tier = self._calculate_tier(num_stages)

        # Determine severity
        technique_ids = [fp.technique_id for fp in campaign_events]
        severity = self.mitre_mapper.classify_severity(technique_ids)

        # Build attack chain
        attack_chain = self._build_attack_chain(campaign_events)

        # Get recommendation
        recommendation = self.mitre_mapper.get_recommended_response(technique_ids)

        # Generate unique alert ID
        alert_id = f"APT-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        # Create alert
        alert = Alert(
            alert_id=alert_id,
            tier=tier,
            severity=severity,
            cve_id=correlation_result.fingerprint.cve_id,
            cve_name=cve_name,
            attack_chain=attack_chain,
            time_span_days=correlation_result.time_span_days,
            correlation_key=correlation_result.fingerprint.get_correlation_key(),
            recommendation=recommendation
        )

        # Track alert
        self.alerts.append(alert)
        self._update_campaign_alerts(alert)
        self._update_stats(tier)

        return alert

    def _calculate_tier(self, num_stages: int) -> int:
        """
        Calculate alert tier based on number of stages

        Args:
            num_stages: Number of attack stages detected

        Returns:
            Tier number (1, 2, or 3)
        """
        thresholds = self.config.tier_thresholds

        if num_stages >= thresholds[3]:
            return 3
        elif num_stages >= thresholds[2]:
            return 2
        else:
            return 1

    def _build_attack_chain(self, campaign_events: List[EventFingerprint]) -> List[Dict[str, Any]]:
        """
        Build attack chain from campaign events

        Args:
            campaign_events: List of event fingerprints

        Returns:
            List of attack stage dictionaries
        """
        # Sort by stage then timestamp
        sorted_events = sorted(campaign_events, key=lambda e: (e.stage, e.timestamp))

        attack_chain = []
        for event in sorted_events:
            technique = self.mitre_mapper.get_technique(event.technique_id)

            attack_chain.append({
                'stage': event.stage,
                'technique': event.technique_id,
                'technique_name': technique.name if technique else 'Unknown',
                'tactic': technique.tactic if technique else 'Unknown',
                'timestamp': event.timestamp,
                'source_ip': event.source_ip,
                'target': event.target_asset
            })

        return attack_chain

    def _update_campaign_alerts(self, alert: Alert):
        """Update campaign alert tracking"""
        if alert.correlation_key not in self.campaign_alerts:
            self.campaign_alerts[alert.correlation_key] = []

        self.campaign_alerts[alert.correlation_key].append(alert.alert_id)

    def _update_stats(self, tier: int):
        """Update statistics"""
        self.stats['total_alerts'] += 1

        if tier == 1:
            self.stats['tier1_alerts'] += 1
        elif tier == 2:
            self.stats['tier2_alerts'] += 1
        elif tier == 3:
            self.stats['tier3_alerts'] += 1

    def should_generate_alert(self, correlation_result: CorrelationResult) -> bool:
        """
        Determine if alert should be generated

        Args:
            correlation_result: Correlation result

        Returns:
            True if alert should be generated
        """
        # Always alert on new campaigns
        if correlation_result.is_new_campaign:
            return True

        # Alert on stage progression (new stage detected)
        correlation_key = correlation_result.fingerprint.get_correlation_key()
        if correlation_key in self.campaign_alerts:
            # Check if we've already alerted for this tier
            existing_alerts = [
                alert for alert in self.alerts
                if alert.correlation_key == correlation_key
            ]
            if existing_alerts:
                highest_tier = max(alert.tier for alert in existing_alerts)
                current_tier = self._calculate_tier(correlation_result.num_stages)

                # Alert if tier increased
                return current_tier > highest_tier

        return True

    def get_alerts_by_tier(self, tier: int) -> List[Alert]:
        """
        Get alerts by tier

        Args:
            tier: Tier number (1, 2, or 3)

        Returns:
            List of alerts
        """
        return [alert for alert in self.alerts if alert.tier == tier]

    def get_campaign_alerts(self, correlation_key: str) -> List[Alert]:
        """
        Get all alerts for a campaign

        Args:
            correlation_key: Correlation key

        Returns:
            List of alerts
        """
        return [
            alert for alert in self.alerts
            if alert.correlation_key == correlation_key
        ]

    def export_alerts(self, output_path: str):
        """
        Export alerts to JSON file

        Args:
            output_path: Output file path
        """
        alerts_data = [alert.to_dict() for alert in self.alerts]

        with open(output_path, 'w') as f:
            json.dump(alerts_data, f, indent=2)

        print(f"Exported {len(alerts_data)} alerts to {output_path}")

    def get_stats(self) -> dict:
        """Get alert statistics"""
        return self.stats.copy()

    def print_stats(self):
        """Print alert statistics"""
        print("\n" + "="*60)
        print("ALERT STATISTICS")
        print("="*60)
        print(f"Total alerts:       {self.stats['total_alerts']}")
        print(f"  Tier 1 (LOW):     {self.stats['tier1_alerts']}")
        print(f"  Tier 2 (HIGH):    {self.stats['tier2_alerts']}")
        print(f"  Tier 3 (CRITICAL): {self.stats['tier3_alerts']}")
        print(f"Active campaigns:   {len(self.campaign_alerts)}")
        print("="*60 + "\n")

    def __repr__(self) -> str:
        return f"AlertManager({self.stats['total_alerts']} alerts generated)"


if __name__ == "__main__":
    # Test alert manager
    from ..utils.fingerprint import EventFingerprint
    from ..bloom.event_bloom import CorrelationResult

    alert_manager = AlertManager()

    # Create test events
    events = [
        EventFingerprint(
            source_ip='10.0.1.5',
            target_asset='nginx:1.19',
            cve_id='CVE-2021-44228',
            stage=1,
            technique_id='T1190',
            timestamp='2024-10-01T10:00:00Z'
        ),
        EventFingerprint(
            source_ip='10.0.1.5',
            target_asset='nginx:1.19',
            cve_id='CVE-2021-44228',
            stage=2,
            technique_id='T1059',
            timestamp='2024-10-12T14:30:00Z'
        ),
        EventFingerprint(
            source_ip='10.0.1.5',
            target_asset='nginx:1.19',
            cve_id='CVE-2021-44228',
            stage=3,
            technique_id='T1053',
            timestamp='2024-11-04T04:15:00Z'
        )
    ]

    # Simulate correlation results
    correlation_result = CorrelationResult(
        fingerprint=events[2],
        is_new_campaign=False,
        detected_stages={1, 2, 3},
        previous_stages=[1, 2],
        time_span_days=33.0
    )

    # Generate alert
    alert = alert_manager.generate_alert(
        correlation_result=correlation_result,
        campaign_events=events,
        cve_name="Log4Shell"
    )

    print("Generated Alert:")
    print(alert.get_summary())
    print("\nAlert JSON:")
    print(alert.to_json())

    # Print stats
    alert_manager.print_stats()
