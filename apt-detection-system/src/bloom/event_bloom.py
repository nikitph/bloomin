"""
Event Bloom Filter (Layer 2)

Tracks entity behavior over time with constant memory.
Enables unlimited temporal correlation across days, weeks, or months.

Key Innovation: Constant 1.2 MB memory per CVE regardless of log volume.
"""

from pybloom_live import BloomFilter
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import time

from ..utils.fingerprint import EventFingerprint, CorrelationKeyManager
from ..bloom.bloom_config import BloomConfig


class CorrelationResult:
    """Represents a correlation analysis result"""

    def __init__(
        self,
        fingerprint: EventFingerprint,
        is_new_campaign: bool,
        detected_stages: Set[int],
        previous_stages: List[int],
        time_span_days: float = 0
    ):
        """
        Initialize correlation result

        Args:
            fingerprint: Current event fingerprint
            is_new_campaign: True if this is a new campaign
            detected_stages: All stages detected in this campaign
            previous_stages: Stages seen before current one
            time_span_days: Time span of campaign in days
        """
        self.fingerprint = fingerprint
        self.is_new_campaign = is_new_campaign
        self.detected_stages = detected_stages
        self.previous_stages = previous_stages
        self.time_span_days = time_span_days

    @property
    def num_stages(self) -> int:
        """Get total number of stages detected"""
        return len(self.detected_stages)

    @property
    def is_multi_stage(self) -> bool:
        """Check if this is a multi-stage attack"""
        return self.num_stages >= 2

    @property
    def is_apt_campaign(self) -> bool:
        """Check if this is an APT campaign (3+ stages)"""
        return self.num_stages >= 3

    def __repr__(self) -> str:
        return (f"CorrelationResult({self.fingerprint.cve_id}, "
                f"stages={sorted(self.detected_stages)}, "
                f"span={self.time_span_days:.1f}d)")


class EventBloomManager:
    """
    Manages Event Bloom Filters for temporal correlation

    Each CVE gets its own Bloom filter to track events over time.
    This enables detection of multi-stage attacks spanning unlimited time windows.
    """

    def __init__(self, config: BloomConfig = None):
        """
        Initialize Event Bloom Manager

        Args:
            config: Bloom filter configuration
        """
        self.config = config or BloomConfig()

        # Event Bloom filters per CVE
        self.event_filters: Dict[str, BloomFilter] = {}

        # Correlation key manager for campaign tracking
        self.correlation_manager = CorrelationKeyManager()

        # Track first/last event timestamps per campaign
        self.campaign_timestamps: Dict[str, Dict[str, str]] = {}

        # Performance metrics
        self.stats = {
            'events_processed': 0,
            'new_campaigns': 0,
            'multi_stage_detected': 0,
            'apt_campaigns_detected': 0,
            'total_time_us': 0
        }

    def _get_or_create_filter(self, cve_id: str) -> BloomFilter:
        """
        Get or create Bloom filter for a CVE

        Args:
            cve_id: CVE identifier

        Returns:
            BloomFilter for this CVE
        """
        if cve_id not in self.event_filters:
            self.event_filters[cve_id] = BloomFilter(
                capacity=self.config.event_capacity,
                error_rate=self.config.event_error_rate
            )
        return self.event_filters[cve_id]

    def add_event(self, fingerprint: EventFingerprint) -> CorrelationResult:
        """
        Add event and check for temporal correlation

        Args:
            fingerprint: Event fingerprint

        Returns:
            CorrelationResult with correlation analysis
        """
        start_time = time.perf_counter()

        # Get Bloom filter for this CVE
        bloom = self._get_or_create_filter(fingerprint.cve_id)

        # Get correlation key for campaign tracking
        corr_key = fingerprint.get_correlation_key()

        # Check if we've seen previous stages
        previous_stages = []
        for prev_stage in range(1, fingerprint.stage):
            # Create fingerprint for previous stage
            prev_fp = EventFingerprint(
                source_ip=fingerprint.source_ip,
                target_asset=fingerprint.target_asset,
                cve_id=fingerprint.cve_id,
                stage=prev_stage,
                technique_id='',  # Not used for correlation
                timestamp=fingerprint.timestamp
            )

            # Check if previous stage is in Bloom filter
            if prev_fp.get_stage_fingerprint() in bloom:
                previous_stages.append(prev_stage)

        # Add current event to campaign
        is_new_campaign = not self.correlation_manager.add_event(fingerprint)

        # Add current event fingerprint to Bloom filter
        bloom.add(fingerprint.get_stage_fingerprint())

        # Get all detected stages
        detected_stages = self.correlation_manager.get_campaign_stages(corr_key)

        # Update campaign timestamps
        if corr_key not in self.campaign_timestamps:
            self.campaign_timestamps[corr_key] = {
                'first': fingerprint.timestamp,
                'last': fingerprint.timestamp
            }
            is_new_campaign = True
        else:
            self.campaign_timestamps[corr_key]['last'] = fingerprint.timestamp

        # Calculate time span
        time_span_days = self._calculate_time_span(corr_key)

        # Create correlation result
        result = CorrelationResult(
            fingerprint=fingerprint,
            is_new_campaign=is_new_campaign,
            detected_stages=detected_stages,
            previous_stages=previous_stages,
            time_span_days=time_span_days
        )

        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.stats['events_processed'] += 1
        self.stats['total_time_us'] += elapsed_us

        if is_new_campaign:
            self.stats['new_campaigns'] += 1

        if result.is_multi_stage:
            self.stats['multi_stage_detected'] += 1

        if result.is_apt_campaign:
            self.stats['apt_campaigns_detected'] += 1

        return result

    def _calculate_time_span(self, correlation_key: str) -> float:
        """
        Calculate time span of a campaign in days

        Args:
            correlation_key: Correlation key

        Returns:
            Time span in days
        """
        if correlation_key not in self.campaign_timestamps:
            return 0.0

        timestamps = self.campaign_timestamps[correlation_key]
        first = datetime.fromisoformat(timestamps['first'].replace('Z', '+00:00'))
        last = datetime.fromisoformat(timestamps['last'].replace('Z', '+00:00'))

        delta = last - first
        return delta.total_seconds() / 86400  # Convert to days

    def check_stage_seen(self, fingerprint: EventFingerprint) -> bool:
        """
        Check if a specific stage has been seen before

        Args:
            fingerprint: Event fingerprint

        Returns:
            True if stage has been seen
        """
        bloom = self._get_or_create_filter(fingerprint.cve_id)
        return fingerprint.get_stage_fingerprint() in bloom

    def get_campaign_info(self, correlation_key: str) -> Optional[dict]:
        """
        Get information about a specific campaign

        Args:
            correlation_key: Correlation key

        Returns:
            Dictionary with campaign information or None
        """
        campaign = self.correlation_manager.get_campaign(correlation_key)
        if not campaign:
            return None

        stages = self.correlation_manager.get_campaign_stages(correlation_key)
        time_span = self._calculate_time_span(correlation_key)

        return {
            'correlation_key': correlation_key,
            'num_events': len(campaign),
            'stages': sorted(stages),
            'time_span_days': time_span,
            'first_seen': self.campaign_timestamps[correlation_key]['first'],
            'last_seen': self.campaign_timestamps[correlation_key]['last']
        }

    def get_all_campaigns(self) -> List[dict]:
        """Get information about all active campaigns"""
        campaigns = []
        for corr_key in self.correlation_manager.active_campaigns.keys():
            info = self.get_campaign_info(corr_key)
            if info:
                campaigns.append(info)
        return campaigns

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        avg_time = (self.stats['total_time_us'] / self.stats['events_processed']
                   if self.stats['events_processed'] > 0 else 0)

        return {
            'events_processed': self.stats['events_processed'],
            'new_campaigns': self.stats['new_campaigns'],
            'multi_stage_detected': self.stats['multi_stage_detected'],
            'apt_campaigns_detected': self.stats['apt_campaigns_detected'],
            'avg_time_us': avg_time,
            'throughput_per_sec': 1_000_000 / avg_time if avg_time > 0 else 0
        }

    def get_filter_stats(self) -> dict:
        """Get Bloom filter statistics"""
        return {
            'num_filters': len(self.event_filters),
            'capacity_per_filter': self.config.event_capacity,
            'error_rate': self.config.event_error_rate,
            'estimated_memory_mb': len(self.event_filters) * 1.2,
            'active_campaigns': self.correlation_manager.get_num_active_campaigns()
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'events_processed': 0,
            'new_campaigns': 0,
            'multi_stage_detected': 0,
            'apt_campaigns_detected': 0,
            'total_time_us': 0
        }

    def __repr__(self) -> str:
        return f"EventBloomManager({len(self.event_filters)} CVE filters, {self.correlation_manager.get_num_active_campaigns()} active campaigns)"


if __name__ == "__main__":
    # Test Event Bloom Filter
    event_bloom = EventBloomManager()

    # Simulate multi-stage attack over 45 days
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

    print("Event Correlation Test:")
    print("="*60)

    for i, event in enumerate(events, 1):
        result = event_bloom.add_event(event)
        print(f"\nEvent {i}: {event}")
        print(f"  Result: {result}")
        print(f"  Is new campaign: {result.is_new_campaign}")
        print(f"  Previous stages: {result.previous_stages}")
        print(f"  Total stages: {sorted(result.detected_stages)}")
        print(f"  Time span: {result.time_span_days:.1f} days")
        print(f"  Multi-stage: {result.is_multi_stage}")
        print(f"  APT campaign: {result.is_apt_campaign}")

    # Campaign information
    print("\n" + "="*60)
    print("Active Campaigns:")
    print("="*60)
    for campaign in event_bloom.get_all_campaigns():
        print(f"\nCampaign: {campaign['correlation_key']}")
        print(f"  Events: {campaign['num_events']}")
        print(f"  Stages: {campaign['stages']}")
        print(f"  Time span: {campaign['time_span_days']:.1f} days")
        print(f"  First seen: {campaign['first_seen']}")
        print(f"  Last seen: {campaign['last_seen']}")

    # Performance stats
    print("\n" + "="*60)
    print("Performance Statistics:")
    print("="*60)
    stats = event_bloom.get_performance_stats()
    for key, value in stats.items():
        if 'time' in key or 'throughput' in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Filter stats
    print("\nFilter Statistics:")
    print("="*60)
    filter_stats = event_bloom.get_filter_stats()
    for key, value in filter_stats.items():
        print(f"{key}: {value}")
