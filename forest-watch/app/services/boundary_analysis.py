"""
Boundary Change Detection Service
Analyzes changes along the boundary of a region to detect encroachment/construction
"""
import ee
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class TileResult:
    """Result for a single tile along the boundary"""
    tile_id: int
    center_lon: float
    center_lat: float
    mean_ndvi_change: float
    min_ndvi_change: float
    loss_area_sqm: float
    baseline_url: str
    current_url: str
    change_url: str


class BoundaryAnalyzer:
    """Analyzes changes along region boundaries"""

    def __init__(self, project_id: str):
        """Initialize with GEE project"""
        try:
            ee.Initialize(project=project_id)
        except:
            pass  # Already initialized

    def parse_kml_boundary(self, kml_path: str) -> List[List[float]]:
        """
        Extract boundary coordinates from KML file.
        Returns list of [lon, lat] coordinates.
        """
        tree = ET.parse(kml_path)
        root = tree.getroot()

        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        all_coords = []
        for coords_elem in root.iter('{http://www.opengis.net/kml/2.2}coordinates'):
            if coords_elem.text:
                points = coords_elem.text.strip().split()
                for point in points:
                    parts = point.split(',')
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            all_coords.append([lon, lat])
                        except ValueError:
                            pass

        return all_coords

    def create_boundary_buffer(
        self,
        coords: List[List[float]],
        buffer_meters: int = 200
    ) -> ee.Geometry:
        """
        Create a buffer zone around the boundary polygon.
        """
        # Create polygon from coordinates
        polygon = ee.Geometry.Polygon([coords])

        # Get the boundary (perimeter) and buffer it
        boundary = polygon.buffer(buffer_meters).difference(polygon.buffer(-buffer_meters))

        return boundary

    def generate_boundary_tiles(
        self,
        coords: List[List[float]],
        tile_size_meters: int = 500,
        buffer_meters: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Generate tiles along the boundary for analysis.
        Each tile is a square centered on points along the boundary.
        """
        tiles = []
        tile_id = 0

        # Sample points along the boundary
        # Calculate approximate spacing based on tile size
        total_points = len(coords)

        # Degrees per meter (approximate at this latitude ~19°N)
        deg_per_meter_lon = 1 / 111320 * 1 / math.cos(math.radians(19))
        deg_per_meter_lat = 1 / 110540

        tile_size_deg_lon = tile_size_meters * deg_per_meter_lon
        tile_size_deg_lat = tile_size_meters * deg_per_meter_lat

        # Sample every N points to get reasonable spacing
        step = max(1, total_points // 100)  # Get ~100 sample points max

        sampled_points = []
        for i in range(0, total_points, step):
            sampled_points.append(coords[i])

        # Remove duplicate/very close points
        filtered_points = []
        min_dist = tile_size_meters * 0.5  # Minimum distance between tile centers

        for point in sampled_points:
            is_far_enough = True
            for existing in filtered_points:
                dist_lon = abs(point[0] - existing[0]) / deg_per_meter_lon
                dist_lat = abs(point[1] - existing[1]) / deg_per_meter_lat
                dist = math.sqrt(dist_lon**2 + dist_lat**2)
                if dist < min_dist:
                    is_far_enough = False
                    break
            if is_far_enough:
                filtered_points.append(point)

        # Create tiles centered on each point
        half_size_lon = tile_size_deg_lon / 2
        half_size_lat = tile_size_deg_lat / 2

        for point in filtered_points:
            lon, lat = point
            tile = {
                'id': tile_id,
                'center': [lon, lat],
                'bbox': [
                    lon - half_size_lon,
                    lat - half_size_lat,
                    lon + half_size_lon,
                    lat + half_size_lat
                ]
            }
            tiles.append(tile)
            tile_id += 1

        return tiles

    def analyze_tile(
        self,
        tile: Dict[str, Any],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        cloud_cover_max: float = 30.0
    ) -> TileResult:
        """
        Analyze a single tile for vegetation change.
        """
        geometry = ee.Geometry.Rectangle(tile['bbox'])

        # Cloud masking function
        def mask_clouds(image):
            qa = image.select('QA60')
            cloud_bit = 1 << 10
            cirrus_bit = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
            return image.updateMask(mask)

        # NDVI function
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        # Get baseline collection
        baseline_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geometry)
            .filterDate(baseline_start, baseline_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
            .map(mask_clouds)
            .map(add_ndvi))

        # Get current collection
        current_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geometry)
            .filterDate(current_start, current_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
            .map(mask_clouds)
            .map(add_ndvi))

        # Create composites
        baseline_ndvi = baseline_col.select('NDVI').median()
        current_ndvi = current_col.select('NDVI').median()

        # Also create RGB composites for visualization
        baseline_rgb = baseline_col.select(['B4', 'B3', 'B2']).median()
        current_rgb = current_col.select(['B4', 'B3', 'B2']).median()

        # Compute change
        ndvi_change = current_ndvi.subtract(baseline_ndvi)

        # Get statistics
        stats = ndvi_change.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True),
            geometry=geometry,
            scale=10,
            maxPixels=1e8
        ).getInfo()

        mean_change = stats.get('NDVI_mean', 0) or 0
        min_change = stats.get('NDVI_min', 0) or 0

        # Calculate loss area
        loss_mask = ndvi_change.lt(-0.15)
        pixel_area = ee.Image.pixelArea()
        loss_area = pixel_area.updateMask(loss_mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=10,
            maxPixels=1e8
        ).getInfo()
        loss_sqm = loss_area.get('area', 0) or 0

        # Generate thumbnail URLs (true color for better visualization of construction)
        vis_params_rgb = {
            'min': 0,
            'max': 3000,
            'bands': ['B4', 'B3', 'B2'],
            'dimensions': 400,
            'region': geometry
        }

        vis_params_change = {
            'min': -0.5,
            'max': 0.5,
            'palette': ['red', 'orange', 'yellow', 'white', 'lightgreen', 'green', 'darkgreen'],
            'dimensions': 400,
            'region': geometry
        }

        baseline_url = baseline_rgb.clip(geometry).getThumbURL(vis_params_rgb)
        current_url = current_rgb.clip(geometry).getThumbURL(vis_params_rgb)
        change_url = ndvi_change.clip(geometry).getThumbURL(vis_params_change)

        return TileResult(
            tile_id=tile['id'],
            center_lon=tile['center'][0],
            center_lat=tile['center'][1],
            mean_ndvi_change=mean_change,
            min_ndvi_change=min_change,
            loss_area_sqm=loss_sqm,
            baseline_url=baseline_url,
            current_url=current_url,
            change_url=change_url
        )

    def find_top_change_tiles(
        self,
        kml_path: str,
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        tile_size_meters: int = 500,
        top_n: int = 10,
        progress_callback=None
    ) -> List[TileResult]:
        """
        Main function: Find the top N tiles with maximum vegetation change.

        Args:
            kml_path: Path to KML file with boundary
            baseline_start/end: Baseline period dates
            current_start/end: Current period dates
            tile_size_meters: Size of each tile
            top_n: Number of top changed tiles to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of TileResult for top N changed tiles, sorted by change magnitude
        """
        print(f"Parsing KML boundary from {kml_path}...")
        coords = self.parse_kml_boundary(kml_path)
        print(f"  Found {len(coords)} boundary points")

        print(f"Generating {tile_size_meters}m tiles along boundary...")
        tiles = self.generate_boundary_tiles(coords, tile_size_meters)
        print(f"  Created {len(tiles)} tiles")

        print(f"Analyzing tiles for change...")
        results = []
        for i, tile in enumerate(tiles):
            if progress_callback:
                progress_callback(i + 1, len(tiles))
            else:
                print(f"  Processing tile {i+1}/{len(tiles)}...", end='\r')

            try:
                result = self.analyze_tile(
                    tile,
                    baseline_start,
                    baseline_end,
                    current_start,
                    current_end
                )
                results.append(result)
            except Exception as e:
                print(f"  Error on tile {tile['id']}: {e}")

        print(f"\nAnalyzed {len(results)} tiles successfully")

        # Sort by mean NDVI change (most negative first = most loss)
        results.sort(key=lambda x: x.mean_ndvi_change)

        # Return top N
        top_results = results[:top_n]

        print(f"\nTop {len(top_results)} tiles with maximum vegetation loss:")
        for i, r in enumerate(top_results):
            print(f"  {i+1}. Tile {r.tile_id}: NDVI change = {r.mean_ndvi_change:.4f}, "
                  f"Loss = {r.loss_area_sqm:.0f} sqm @ ({r.center_lon:.4f}, {r.center_lat:.4f})")

        return top_results


def run_boundary_analysis(
    kml_path: str,
    output_dir: str,
    project_id: str,
    baseline_start: str = "2023-12-01",
    baseline_end: str = "2023-12-31",
    current_start: str = "2024-11-01",
    current_end: str = "2024-12-15",
    tile_size: int = 500,
    top_n: int = 10
) -> List[TileResult]:
    """
    Convenience function to run boundary analysis and download images.
    """
    import os
    import urllib.request

    analyzer = BoundaryAnalyzer(project_id)

    results = analyzer.find_top_change_tiles(
        kml_path=kml_path,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=current_end,
        tile_size_meters=tile_size,
        top_n=top_n
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download images
    print(f"\nDownloading images to {output_dir}...")
    for i, result in enumerate(results):
        rank = i + 1
        print(f"  Downloading tile {rank} (change: {result.mean_ndvi_change:.4f})...")

        # Download baseline
        baseline_path = os.path.join(output_dir, f"tile_{rank:02d}_baseline.png")
        urllib.request.urlretrieve(result.baseline_url, baseline_path)

        # Download current
        current_path = os.path.join(output_dir, f"tile_{rank:02d}_current.png")
        urllib.request.urlretrieve(result.current_url, current_path)

        # Download change map
        change_path = os.path.join(output_dir, f"tile_{rank:02d}_change.png")
        urllib.request.urlretrieve(result.change_url, change_path)

    print(f"\nDone! Downloaded {len(results) * 3} images")

    return results


if __name__ == "__main__":
    # Test run
    import sys

    if len(sys.argv) < 2:
        print("Usage: python boundary_analysis.py <kml_file>")
        sys.exit(1)

    results = run_boundary_analysis(
        kml_path=sys.argv[1],
        output_dir="boundary_tiles",
        project_id="bubbly-mantis-452004-i3"
    )
