"""
Forest Watch - Satellite Image Change Detection System
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from app.config import settings
from app.api.routes import router
from app.services.gee_service import GEEService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Forest Watch API...")
    try:
        # Initialize Google Earth Engine
        # Note: For first run, you may need to authenticate via CLI:
        # earthengine authenticate
        GEEService.initialize()
        logger.info("GEE initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GEE: {e}")
        logger.warning("API will start but GEE features will not work")
        logger.info("Run 'earthengine authenticate' to set up credentials")

    yield

    # Shutdown
    logger.info("Shutting down Forest Watch API...")


# Create FastAPI app
app = FastAPI(
    title="Forest Watch API",
    description="""
## Satellite Image Change Detection System

Monitor forest areas in Maharashtra and Madhya Pradesh for potential deforestation
or illegal activities using Sentinel-2 satellite imagery.

### Features
- **NDVI-based change detection** - Compare vegetation indices between time periods
- **Configurable thresholds** - Tune sensitivity for your use case
- **Multiple forest areas** - Pre-configured tiger reserves and national parks
- **Alert levels** - HIGH, MEDIUM, LOW based on change magnitude and area

### How it works
1. Fetches Sentinel-2 imagery from Google Earth Engine
2. Computes NDVI (Normalized Difference Vegetation Index) for each period
3. Calculates change between baseline and current periods
4. Identifies areas with significant vegetation loss
5. Generates alerts based on configurable thresholds

### Typical workflow
1. `GET /areas` - List available forest areas
2. `POST /imagery/info` - Check data availability for your date range
3. `POST /analyze` - Run change detection analysis
4. Review results and visualization URLs
    """,
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Forest Watch API",
        "version": "0.1.0",
        "description": "Satellite image change detection for forest monitoring",
        "docs": "/docs",
        "health": "/api/v1/health",
        "areas": "/api/v1/areas"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
