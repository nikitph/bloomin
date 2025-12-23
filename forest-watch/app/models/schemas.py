"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AlertLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# Request schemas

class AnalysisRequest(BaseModel):
    """Request for change detection analysis"""
    area_id: str = Field(..., description="ID of the forest area to analyze")
    baseline_start: str = Field(..., description="Baseline period start (YYYY-MM-DD)")
    baseline_end: str = Field(..., description="Baseline period end (YYYY-MM-DD)")
    current_start: str = Field(..., description="Current period start (YYYY-MM-DD)")
    current_end: str = Field(..., description="Current period end (YYYY-MM-DD)")
    generate_maps: bool = Field(default=True, description="Generate visualization URLs")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "area_id": "tadoba",
                "baseline_start": "2023-01-01",
                "baseline_end": "2023-01-31",
                "current_start": "2024-01-01",
                "current_end": "2024-01-31",
                "generate_maps": True
            }]
        }
    }


class QuickScanRequest(BaseModel):
    """Request for quick scan analysis"""
    area_id: str = Field(..., description="ID of the forest area to scan")
    days_back: int = Field(default=30, ge=7, le=90, description="Days to look back for current period")
    baseline_days: int = Field(default=30, ge=7, le=90, description="Duration of baseline period")


class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis of multiple areas"""
    area_ids: Optional[List[str]] = Field(default=None, description="List of area IDs (None = all)")
    baseline_start: str
    baseline_end: str
    current_start: str
    current_end: str
    generate_maps: bool = Field(default=False)


# Response schemas

class PeriodInfo(BaseModel):
    start: str
    end: str


class AnalysisMetrics(BaseModel):
    mean_ndvi_change: float = Field(..., description="Mean NDVI change (negative = loss)")
    loss_area_hectares: float = Field(..., description="Area of vegetation loss in hectares")
    alert_level: AlertLevel
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")


class VisualizationUrls(BaseModel):
    change_map_url: Optional[str] = Field(None, description="URL to NDVI change visualization")
    loss_map_url: Optional[str] = Field(None, description="URL to loss mask visualization")


class AreaInfo(BaseModel):
    id: str
    name: str
    state: str
    bbox: List[float]
    center: List[float]


class AnalysisResponse(BaseModel):
    """Response from change detection analysis"""
    area: AreaInfo
    baseline_period: PeriodInfo
    current_period: PeriodInfo
    analysis: AnalysisMetrics
    visualization: VisualizationUrls
    timestamp: datetime


class BatchAnalysisResponse(BaseModel):
    """Response from batch analysis"""
    total_areas: int
    alerts_high: int
    alerts_medium: int
    alerts_low: int
    results: List[AnalysisResponse]


class AreaListResponse(BaseModel):
    """List of available forest areas"""
    areas: List[AreaInfo]


class ImageryInfoRequest(BaseModel):
    """Request for imagery availability info"""
    area_id: str
    start_date: str
    end_date: str
    cloud_cover_max: float = Field(default=20.0, ge=0, le=100)


class ImageryInfoResponse(BaseModel):
    """Information about available imagery"""
    area_id: str
    area_name: str
    image_count: int
    dates: List[str]
    period: PeriodInfo


class HealthResponse(BaseModel):
    status: str
    gee_initialized: bool
    timestamp: datetime
