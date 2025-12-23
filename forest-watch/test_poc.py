#!/usr/bin/env python3
"""
Test script for Forest Watch PoC
Run this to verify GEE connection and basic functionality
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_gee_connection():
    """Test Google Earth Engine connection"""
    print("=" * 60)
    print("Testing Google Earth Engine Connection")
    print("=" * 60)

    try:
        import ee
        from app.config import settings

        project = settings.gee_project or None
        print(f"Using project: {project}")
        ee.Initialize(project=project)
        print("✓ GEE initialized successfully")

        # Quick test - get image count
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
        count = collection.size().getInfo()
        print(f"✓ Can query collections (found {count} test images)")

        return True
    except Exception as e:
        print(f"✗ GEE connection failed: {e}")
        print("\nTo authenticate, run:")
        print("  earthengine authenticate")
        return False


def test_imagery_availability():
    """Test imagery availability for sample area"""
    print("\n" + "=" * 60)
    print("Testing Imagery Availability (Tadoba Tiger Reserve)")
    print("=" * 60)

    from app.services.gee_service import GEEService
    from app.config import FOREST_AREAS, settings

    # Ensure GEE is initialized
    GEEService.initialize(settings.gee_project)

    area = FOREST_AREAS['tadoba']
    print(f"Area: {area.name}")
    print(f"Bbox: {area.bbox}")

    # Check recent imagery
    info = GEEService.get_collection_info(
        bbox=area.bbox,
        start_date='2024-01-01',
        end_date='2024-01-31',
        cloud_cover_max=30.0
    )

    print(f"✓ Found {info['image_count']} images for Jan 2024")
    if info['dates']:
        print(f"  Dates: {', '.join(info['dates'][:5])}...")

    return info['image_count'] > 0


def test_ndvi_computation():
    """Test NDVI computation for sample area"""
    print("\n" + "=" * 60)
    print("Testing NDVI Computation")
    print("=" * 60)

    from app.services.gee_service import GEEService
    from app.config import FOREST_AREAS, settings

    # Ensure GEE is initialized
    GEEService.initialize(settings.gee_project)

    area = FOREST_AREAS['tadoba']

    print("Computing NDVI composite for Jan 2024...")
    ndvi = GEEService.get_ndvi_composite(
        bbox=area.bbox,
        start_date='2024-01-01',
        end_date='2024-01-31',
        cloud_cover_max=30.0
    )

    # Get a sample value to verify computation worked
    import ee
    geometry = ee.Geometry.Rectangle(area.bbox)
    stats = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=100,  # Coarse scale for speed
        maxPixels=1e6
    ).getInfo()

    mean_ndvi = stats.get('NDVI', None)
    if mean_ndvi is not None:
        print(f"✓ NDVI computed successfully")
        print(f"  Mean NDVI: {mean_ndvi:.3f}")
        print(f"  (Expected range for forests: 0.4 - 0.8)")
        return True
    else:
        print("✗ NDVI computation returned None")
        return False


def test_change_detection():
    """Test full change detection workflow"""
    print("\n" + "=" * 60)
    print("Testing Change Detection (Tadoba: Jan 2023 vs Jan 2024)")
    print("=" * 60)

    from app.services.change_detection import ChangeDetectionService
    from app.services.gee_service import GEEService
    from app.config import settings

    # Ensure GEE is initialized
    GEEService.initialize(settings.gee_project)

    detector = ChangeDetectionService()

    print("Running analysis (this may take 30-60 seconds)...")
    result = detector.analyze_area(
        area_id='tadoba',
        baseline_start='2023-01-01',
        baseline_end='2023-01-31',
        current_start='2024-01-01',
        current_end='2024-01-31',
        generate_maps=True
    )

    print(f"\n✓ Analysis complete!")
    print(f"  Area: {result.area_info.name}")
    print(f"  Mean NDVI change: {result.mean_ndvi_change:.4f}")
    print(f"  Loss area: {result.loss_area_hectares:.2f} hectares")
    print(f"  Alert level: {result.alert_level.value}")
    print(f"  Confidence: {result.confidence:.3f}")

    if result.change_map_url:
        print(f"\n  Change map URL (valid for ~1 hour):")
        print(f"  {result.change_map_url[:100]}...")

    return True


def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# FOREST WATCH - Proof of Concept Test")
    print("#" * 60)

    # Test 1: GEE Connection
    if not test_gee_connection():
        print("\n❌ GEE connection failed. Please authenticate first.")
        return False

    # Test 2: Imagery availability
    try:
        test_imagery_availability()
    except Exception as e:
        print(f"✗ Imagery test failed: {e}")

    # Test 3: NDVI computation
    try:
        test_ndvi_computation()
    except Exception as e:
        print(f"✗ NDVI test failed: {e}")

    # Test 4: Full change detection
    try:
        test_change_detection()
    except Exception as e:
        print(f"✗ Change detection test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 60)
    print("# Tests Complete")
    print("#" * 60)
    print("\nTo start the API server:")
    print("  cd forest-watch")
    print("  uvicorn app.main:app --reload")
    print("\nThen visit: http://localhost:8000/docs")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
