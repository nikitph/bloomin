from pydantic_settings import BaseSettings
from typing import Dict, List
from pydantic import BaseModel


class AreaOfInterest(BaseModel):
    """Represents a monitored forest area"""
    name: str
    state: str
    # Bounding box: [west, south, east, north] (lon, lat, lon, lat)
    bbox: List[float]
    # Optional: center point for reference
    center: List[float]


# Known forest areas in Maharashtra and Madhya Pradesh
FOREST_AREAS: Dict[str, AreaOfInterest] = {
    # Maharashtra
    "tadoba": AreaOfInterest(
        name="Tadoba-Andhari Tiger Reserve",
        state="Maharashtra",
        bbox=[79.2, 20.0, 79.6, 20.4],
        center=[79.4, 20.2]
    ),
    "melghat": AreaOfInterest(
        name="Melghat Tiger Reserve",
        state="Maharashtra",
        bbox=[76.8, 21.3, 77.3, 21.7],
        center=[77.05, 21.5]
    ),
    "navegaon": AreaOfInterest(
        name="Navegaon-Nagzira Tiger Reserve",
        state="Maharashtra",
        bbox=[79.8, 20.8, 80.3, 21.2],
        center=[80.05, 21.0]
    ),

    # Madhya Pradesh
    "kanha": AreaOfInterest(
        name="Kanha National Park",
        state="Madhya Pradesh",
        bbox=[80.4, 22.1, 81.1, 22.5],
        center=[80.75, 22.3]
    ),
    "bandhavgarh": AreaOfInterest(
        name="Bandhavgarh National Park",
        state="Madhya Pradesh",
        bbox=[80.8, 23.6, 81.2, 23.9],
        center=[81.0, 23.75]
    ),
    "pench_mp": AreaOfInterest(
        name="Pench National Park (MP)",
        state="Madhya Pradesh",
        bbox=[79.1, 21.6, 79.5, 22.0],
        center=[79.3, 21.8]
    ),
    "satpura": AreaOfInterest(
        name="Satpura National Park",
        state="Madhya Pradesh",
        bbox=[78.1, 22.4, 78.6, 22.8],
        center=[78.35, 22.6]
    ),
}


class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "Forest Watch"
    debug: bool = True

    # GEE settings
    gee_project: str = ""  # Your Google Cloud project ID for Earth Engine
    gee_service_account: str = ""
    gee_credentials_file: str = ""

    # Change detection thresholds
    ndvi_change_threshold: float = -0.15  # NDVI drop threshold (negative = vegetation loss)
    min_change_area_hectares: float = 1.0  # Minimum area to trigger alert
    cloud_cover_max: float = 20.0  # Maximum cloud cover percentage

    # Alert settings
    alert_confidence_high: float = 0.8
    alert_confidence_medium: float = 0.5

    class Config:
        env_file = ".env"


settings = Settings()
