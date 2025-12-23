"""
FastAPI routes for Forest Watch API
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import logging

from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    QuickScanRequest,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AreaListResponse,
    AreaInfo,
    ImageryInfoRequest,
    ImageryInfoResponse,
    HealthResponse,
    PeriodInfo,
    AnalysisMetrics,
    VisualizationUrls,
    AlertLevel
)
from app.config import FOREST_AREAS
from app.services.gee_service import GEEService
from app.services.change_detection import ChangeDetectionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
change_detector = ChangeDetectionService()


def result_to_response(result) -> AnalysisResponse:
    """Convert ChangeDetectionResult to AnalysisResponse"""
    return AnalysisResponse(
        area=AreaInfo(
            id=result.area_id,
            name=result.area_info.name,
            state=result.area_info.state,
            bbox=result.area_info.bbox,
            center=result.area_info.center
        ),
        baseline_period=PeriodInfo(
            start=result.baseline_period[0],
            end=result.baseline_period[1]
        ),
        current_period=PeriodInfo(
            start=result.current_period[0],
            end=result.current_period[1]
        ),
        analysis=AnalysisMetrics(
            mean_ndvi_change=result.mean_ndvi_change,
            loss_area_hectares=result.loss_area_hectares,
            alert_level=AlertLevel(result.alert_level.value),
            confidence=result.confidence
        ),
        visualization=VisualizationUrls(
            change_map_url=result.change_map_url,
            loss_map_url=result.loss_map_url
        ),
        timestamp=result.timestamp
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and GEE connection health"""
    return HealthResponse(
        status="healthy",
        gee_initialized=GEEService._initialized,
        timestamp=datetime.utcnow()
    )


@router.get("/areas", response_model=AreaListResponse, tags=["Areas"])
async def list_areas():
    """List all configured forest areas for monitoring"""
    areas = [
        AreaInfo(
            id=area_id,
            name=area.name,
            state=area.state,
            bbox=area.bbox,
            center=area.center
        )
        for area_id, area in FOREST_AREAS.items()
    ]
    return AreaListResponse(areas=areas)


@router.get("/areas/{area_id}", response_model=AreaInfo, tags=["Areas"])
async def get_area(area_id: str):
    """Get details for a specific forest area"""
    if area_id not in FOREST_AREAS:
        raise HTTPException(
            status_code=404,
            detail=f"Area '{area_id}' not found. Available: {list(FOREST_AREAS.keys())}"
        )

    area = FOREST_AREAS[area_id]
    return AreaInfo(
        id=area_id,
        name=area.name,
        state=area.state,
        bbox=area.bbox,
        center=area.center
    )


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_area(request: AnalysisRequest):
    """
    Perform change detection analysis for a specific area.

    Compares NDVI between baseline and current periods to detect vegetation loss.
    """
    try:
        result = change_detector.analyze_area(
            area_id=request.area_id,
            baseline_start=request.baseline_start,
            baseline_end=request.baseline_end,
            current_start=request.current_start,
            current_end=request.current_end,
            generate_maps=request.generate_maps
        )
        return result_to_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/quick", response_model=AnalysisResponse, tags=["Analysis"])
async def quick_scan(request: QuickScanRequest):
    """
    Quick scan comparing recent imagery to same period last year.

    Useful for rapid checks without specifying exact date ranges.
    """
    try:
        result = change_detector.quick_scan(
            area_id=request.area_id,
            days_back=request.days_back,
            baseline_days=request.baseline_days
        )
        return result_to_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quick scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick scan failed: {str(e)}")


@router.post("/analyze/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def batch_analyze(request: BatchAnalysisRequest):
    """
    Analyze multiple forest areas in batch.

    If area_ids is not provided, analyzes all configured areas.
    """
    area_ids = request.area_ids or list(FOREST_AREAS.keys())

    # Validate all area IDs
    invalid = [a for a in area_ids if a not in FOREST_AREAS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid area IDs: {invalid}. Available: {list(FOREST_AREAS.keys())}"
        )

    results = []
    for area_id in area_ids:
        try:
            result = change_detector.analyze_area(
                area_id=area_id,
                baseline_start=request.baseline_start,
                baseline_end=request.baseline_end,
                current_start=request.current_start,
                current_end=request.current_end,
                generate_maps=request.generate_maps
            )
            results.append(result_to_response(result))
        except Exception as e:
            logger.error(f"Failed to analyze {area_id}: {e}")

    # Count alerts by level
    alerts_high = sum(1 for r in results if r.analysis.alert_level == AlertLevel.HIGH)
    alerts_medium = sum(1 for r in results if r.analysis.alert_level == AlertLevel.MEDIUM)
    alerts_low = sum(1 for r in results if r.analysis.alert_level == AlertLevel.LOW)

    return BatchAnalysisResponse(
        total_areas=len(results),
        alerts_high=alerts_high,
        alerts_medium=alerts_medium,
        alerts_low=alerts_low,
        results=results
    )


@router.post("/imagery/info", response_model=ImageryInfoResponse, tags=["Imagery"])
async def get_imagery_info(request: ImageryInfoRequest):
    """
    Check available Sentinel-2 imagery for an area and date range.

    Useful for understanding data availability before running analysis.
    """
    if request.area_id not in FOREST_AREAS:
        raise HTTPException(
            status_code=404,
            detail=f"Area '{request.area_id}' not found"
        )

    area = FOREST_AREAS[request.area_id]

    try:
        info = GEEService.get_collection_info(
            bbox=area.bbox,
            start_date=request.start_date,
            end_date=request.end_date,
            cloud_cover_max=request.cloud_cover_max
        )

        return ImageryInfoResponse(
            area_id=request.area_id,
            area_name=area.name,
            image_count=info['image_count'],
            dates=info['dates'],
            period=PeriodInfo(
                start=request.start_date,
                end=request.end_date
            )
        )
    except Exception as e:
        logger.error(f"Failed to get imagery info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get imagery info: {str(e)}")


@router.get("/alerts", tags=["Alerts"])
async def get_recent_alerts(
    min_level: AlertLevel = Query(default=AlertLevel.LOW, description="Minimum alert level"),
    state: Optional[str] = Query(default=None, description="Filter by state")
):
    """
    Get placeholder for recent alerts.

    Note: In production, this would query a database of stored analysis results.
    For PoC, this returns guidance on how to run analyses.
    """
    return {
        "message": "Alert history not implemented in PoC",
        "guidance": {
            "to_run_analysis": "POST /analyze with area_id and date ranges",
            "to_quick_scan": "POST /analyze/quick with area_id",
            "to_batch_scan": "POST /analyze/batch with date ranges (optionally filter by area_ids)"
        },
        "available_areas": list(FOREST_AREAS.keys())
    }
