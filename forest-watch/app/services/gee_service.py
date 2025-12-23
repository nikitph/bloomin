"""
Google Earth Engine Service
Handles authentication, image fetching, and NDVI computation
"""
import ee
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class GEEService:
    """Service for interacting with Google Earth Engine"""

    _initialized = False

    @classmethod
    def initialize(cls, project_id: Optional[str] = None):
        """
        Initialize Earth Engine.
        For PoC, uses default credentials (authenticate via `earthengine authenticate` CLI)

        Args:
            project_id: Google Cloud project ID with Earth Engine enabled
        """
        if cls._initialized:
            return

        # Import settings here to avoid circular imports
        from app.config import settings

        # Use provided project_id or fall back to settings
        project = project_id or settings.gee_project or None

        try:
            # Try to initialize with existing credentials
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            cls._initialized = True
            logger.info(f"Google Earth Engine initialized successfully (project: {project or 'default'})")
        except Exception as e:
            logger.warning(f"GEE initialization failed: {e}")
            logger.info("Attempting authentication...")
            # This will open a browser for authentication
            ee.Authenticate()
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            cls._initialized = True
            logger.info("Google Earth Engine authenticated and initialized")

    @staticmethod
    def get_sentinel2_collection(
        bbox: List[float],
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 20.0
    ) -> ee.ImageCollection:
        """
        Get Sentinel-2 imagery for a given bounding box and date range.

        Args:
            bbox: [west, south, east, north] coordinates
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            cloud_cover_max: Maximum cloud cover percentage

        Returns:
            Filtered ImageCollection
        """
        # Create geometry from bbox
        geometry = ee.Geometry.Rectangle(bbox)

        # Get Sentinel-2 Surface Reflectance collection
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
        )

        return collection

    @staticmethod
    def compute_ndvi(image: ee.Image) -> ee.Image:
        """
        Compute NDVI for a Sentinel-2 image.
        NDVI = (NIR - Red) / (NIR + Red)
        For Sentinel-2: NIR = B8, Red = B4
        """
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    @staticmethod
    def mask_clouds_sentinel2(image: ee.Image) -> ee.Image:
        """
        Mask clouds in Sentinel-2 imagery using the QA60 band.
        """
        qa = image.select('QA60')

        # Bits 10 and 11 are clouds and cirrus
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be zero, indicating clear conditions
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask)

    @classmethod
    def get_ndvi_composite(
        cls,
        bbox: List[float],
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 20.0
    ) -> ee.Image:
        """
        Get a cloud-free NDVI composite for a given period.
        Uses median compositing to reduce cloud/noise effects.
        """
        collection = cls.get_sentinel2_collection(
            bbox, start_date, end_date, cloud_cover_max
        )

        # Apply cloud masking and compute NDVI
        processed = (
            collection
            .map(cls.mask_clouds_sentinel2)
            .map(cls.compute_ndvi)
        )

        # Create median composite of NDVI
        ndvi_composite = processed.select('NDVI').median()

        return ndvi_composite

    @classmethod
    def compute_ndvi_change(
        cls,
        bbox: List[float],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        cloud_cover_max: float = 20.0
    ) -> Dict[str, ee.Image]:
        """
        Compute NDVI change between baseline and current period.

        Returns dict with:
            - baseline_ndvi: NDVI composite for baseline period
            - current_ndvi: NDVI composite for current period
            - ndvi_change: Difference (current - baseline)
        """
        baseline_ndvi = cls.get_ndvi_composite(
            bbox, baseline_start, baseline_end, cloud_cover_max
        )

        current_ndvi = cls.get_ndvi_composite(
            bbox, current_start, current_end, cloud_cover_max
        )

        # Change = current - baseline (negative means vegetation loss)
        ndvi_change = current_ndvi.subtract(baseline_ndvi).rename('NDVI_change')

        return {
            'baseline_ndvi': baseline_ndvi,
            'current_ndvi': current_ndvi,
            'ndvi_change': ndvi_change
        }

    @classmethod
    def get_change_statistics(
        cls,
        ndvi_change: ee.Image,
        bbox: List[float],
        threshold: float = -0.15
    ) -> Dict[str, Any]:
        """
        Compute statistics for vegetation change.

        Args:
            ndvi_change: NDVI change image
            bbox: Bounding box
            threshold: NDVI change threshold (negative for loss)

        Returns:
            Dictionary with change statistics
        """
        geometry = ee.Geometry.Rectangle(bbox)

        # Create mask for significant vegetation loss
        loss_mask = ndvi_change.lt(threshold)

        # Calculate area of change (in hectares)
        # Sentinel-2 pixel is 10m x 10m = 100 sq meters = 0.01 hectares
        pixel_area = ee.Image.pixelArea().divide(10000)  # Convert to hectares
        loss_area = pixel_area.updateMask(loss_mask)

        # Compute statistics
        stats = ndvi_change.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), '', True
            ).combine(
                ee.Reducer.min(), '', True
            ).combine(
                ee.Reducer.max(), '', True
            ),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        )

        # Compute area of vegetation loss
        area_stats = loss_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        )

        return {
            'change_stats': stats,
            'loss_area_hectares': area_stats
        }

    @classmethod
    def get_change_map_url(
        cls,
        ndvi_change: ee.Image,
        bbox: List[float],
        threshold: float = -0.15
    ) -> Dict[str, str]:
        """
        Generate tile URLs for visualization.

        Returns URLs for:
            - NDVI change visualization
            - Significant loss areas (binary mask)
        """
        geometry = ee.Geometry.Rectangle(bbox)

        # Visualization parameters for NDVI change (-1 to 1 range)
        change_vis = {
            'min': -0.5,
            'max': 0.5,
            'palette': ['red', 'orange', 'yellow', 'white', 'lightgreen', 'green', 'darkgreen']
        }

        # Create loss mask
        loss_mask = ndvi_change.lt(threshold).selfMask()
        loss_vis = {
            'palette': ['red']
        }

        # Get map URLs
        change_url = ndvi_change.clip(geometry).getThumbURL({
            'min': change_vis['min'],
            'max': change_vis['max'],
            'palette': change_vis['palette'],
            'dimensions': 512,
            'region': geometry
        })

        loss_url = loss_mask.clip(geometry).getThumbURL({
            'palette': loss_vis['palette'],
            'dimensions': 512,
            'region': geometry
        })

        return {
            'change_map_url': change_url,
            'loss_map_url': loss_url
        }

    @classmethod
    def get_collection_info(
        cls,
        bbox: List[float],
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 20.0
    ) -> Dict[str, Any]:
        """Get information about available imagery for a region/period"""
        collection = cls.get_sentinel2_collection(
            bbox, start_date, end_date, cloud_cover_max
        )

        count = collection.size().getInfo()

        # Get dates of available images
        if count > 0:
            dates = collection.aggregate_array('system:time_start').getInfo()
            dates = [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in dates]
        else:
            dates = []

        return {
            'image_count': count,
            'dates': dates,
            'bbox': bbox,
            'start_date': start_date,
            'end_date': end_date
        }
