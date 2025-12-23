"""
Change Detection Service
Analyzes NDVI changes and generates alerts
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

from app.config import settings, FOREST_AREAS, AreaOfInterest
from app.services.gee_service import GEEService

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ChangeDetectionResult:
    """Result of change detection analysis"""

    def __init__(
        self,
        area_id: str,
        area_info: AreaOfInterest,
        baseline_period: tuple,
        current_period: tuple,
        mean_ndvi_change: float,
        loss_area_hectares: float,
        alert_level: AlertLevel,
        confidence: float,
        change_map_url: Optional[str] = None,
        loss_map_url: Optional[str] = None
    ):
        self.area_id = area_id
        self.area_info = area_info
        self.baseline_period = baseline_period
        self.current_period = current_period
        self.mean_ndvi_change = mean_ndvi_change
        self.loss_area_hectares = loss_area_hectares
        self.alert_level = alert_level
        self.confidence = confidence
        self.change_map_url = change_map_url
        self.loss_map_url = loss_map_url
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area_id": self.area_id,
            "area_name": self.area_info.name,
            "state": self.area_info.state,
            "bbox": self.area_info.bbox,
            "baseline_period": {
                "start": self.baseline_period[0],
                "end": self.baseline_period[1]
            },
            "current_period": {
                "start": self.current_period[0],
                "end": self.current_period[1]
            },
            "analysis": {
                "mean_ndvi_change": self.mean_ndvi_change,
                "loss_area_hectares": self.loss_area_hectares,
                "alert_level": self.alert_level.value,
                "confidence": self.confidence
            },
            "visualization": {
                "change_map_url": self.change_map_url,
                "loss_map_url": self.loss_map_url
            },
            "timestamp": self.timestamp.isoformat()
        }


class ChangeDetectionService:
    """Service for detecting vegetation changes in monitored areas"""

    def __init__(self):
        self.threshold = settings.ndvi_change_threshold
        self.min_area = settings.min_change_area_hectares
        self.cloud_cover_max = settings.cloud_cover_max

    def _calculate_alert_level(
        self,
        mean_change: float,
        loss_area: float
    ) -> tuple[AlertLevel, float]:
        """
        Calculate alert level and confidence based on change metrics.

        Alert logic:
        - HIGH: Large area affected (>10 ha) OR severe NDVI drop (< -0.3)
        - MEDIUM: Moderate area (>3 ha) OR moderate drop (< -0.2)
        - LOW: Small but detectable changes above threshold
        - NONE: Changes below detection threshold
        """
        # Base confidence from NDVI change magnitude
        if mean_change >= 0:
            # No vegetation loss
            return AlertLevel.NONE, 0.0

        # Calculate confidence based on magnitude of change
        # More negative = higher confidence of real change
        magnitude = abs(mean_change)
        confidence = min(1.0, magnitude / 0.4)  # Cap at 1.0

        # Adjust for area affected
        if loss_area >= 10.0:
            confidence = min(1.0, confidence + 0.2)
        elif loss_area >= 3.0:
            confidence = min(1.0, confidence + 0.1)

        # Determine alert level
        if (loss_area >= 10.0 or mean_change < -0.3) and confidence >= settings.alert_confidence_high:
            return AlertLevel.HIGH, confidence
        elif (loss_area >= 3.0 or mean_change < -0.2) and confidence >= settings.alert_confidence_medium:
            return AlertLevel.MEDIUM, confidence
        elif loss_area >= self.min_area and mean_change < self.threshold:
            return AlertLevel.LOW, confidence
        else:
            return AlertLevel.NONE, confidence

    def analyze_area(
        self,
        area_id: str,
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        generate_maps: bool = True
    ) -> ChangeDetectionResult:
        """
        Perform change detection analysis for a specific area.

        Args:
            area_id: ID of the forest area (from FOREST_AREAS)
            baseline_start/end: Baseline period dates (YYYY-MM-DD)
            current_start/end: Current period dates (YYYY-MM-DD)
            generate_maps: Whether to generate visualization URLs

        Returns:
            ChangeDetectionResult with analysis findings
        """
        if area_id not in FOREST_AREAS:
            raise ValueError(f"Unknown area: {area_id}. Available: {list(FOREST_AREAS.keys())}")

        area = FOREST_AREAS[area_id]
        logger.info(f"Analyzing {area.name} ({area_id})")

        # Compute NDVI change
        change_result = GEEService.compute_ndvi_change(
            bbox=area.bbox,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            current_start=current_start,
            current_end=current_end,
            cloud_cover_max=self.cloud_cover_max
        )

        # Get statistics
        stats = GEEService.get_change_statistics(
            ndvi_change=change_result['ndvi_change'],
            bbox=area.bbox,
            threshold=self.threshold
        )

        # Extract values from GEE results
        change_stats = stats['change_stats'].getInfo()
        loss_area = stats['loss_area_hectares'].getInfo()

        mean_change = change_stats.get('NDVI_change_mean', 0) or 0
        loss_hectares = loss_area.get('area', 0) or 0

        # Calculate alert level
        alert_level, confidence = self._calculate_alert_level(mean_change, loss_hectares)

        # Generate visualization URLs if requested
        change_map_url = None
        loss_map_url = None
        if generate_maps:
            try:
                map_urls = GEEService.get_change_map_url(
                    ndvi_change=change_result['ndvi_change'],
                    bbox=area.bbox,
                    threshold=self.threshold
                )
                change_map_url = map_urls['change_map_url']
                loss_map_url = map_urls['loss_map_url']
            except Exception as e:
                logger.warning(f"Failed to generate maps: {e}")

        result = ChangeDetectionResult(
            area_id=area_id,
            area_info=area,
            baseline_period=(baseline_start, baseline_end),
            current_period=(current_start, current_end),
            mean_ndvi_change=round(mean_change, 4),
            loss_area_hectares=round(loss_hectares, 2),
            alert_level=alert_level,
            confidence=round(confidence, 3),
            change_map_url=change_map_url,
            loss_map_url=loss_map_url
        )

        logger.info(
            f"Analysis complete: {area.name} - "
            f"NDVI change: {mean_change:.4f}, "
            f"Loss area: {loss_hectares:.2f} ha, "
            f"Alert: {alert_level.value}"
        )

        return result

    def analyze_all_areas(
        self,
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        generate_maps: bool = False
    ) -> List[ChangeDetectionResult]:
        """Analyze all configured forest areas"""
        results = []
        for area_id in FOREST_AREAS:
            try:
                result = self.analyze_area(
                    area_id=area_id,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    current_start=current_start,
                    current_end=current_end,
                    generate_maps=generate_maps
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {area_id}: {e}")

        return results

    def quick_scan(
        self,
        area_id: str,
        days_back: int = 30,
        baseline_days: int = 30
    ) -> ChangeDetectionResult:
        """
        Quick scan comparing recent period to baseline.

        Args:
            area_id: Forest area to scan
            days_back: How many days back for current period
            baseline_days: Duration of baseline period (same # of days, but from previous year)
        """
        today = datetime.utcnow()

        # Current period: last N days
        current_end = today.strftime('%Y-%m-%d')
        current_start = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')

        # Baseline: same period last year
        baseline_end = (today - timedelta(days=365)).strftime('%Y-%m-%d')
        baseline_start = (today - timedelta(days=365 + baseline_days)).strftime('%Y-%m-%d')

        return self.analyze_area(
            area_id=area_id,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            current_start=current_start,
            current_end=current_end,
            generate_maps=True
        )
