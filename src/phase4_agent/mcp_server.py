import sys

# ---------------------------------------------------------
# Add Phase 1, Phase 2, and Phase 3 source folders
# ---------------------------------------------------------

sys.path.append("src/phase1_ml")
sys.path.append("src/phase2_dl")
sys.path.append("src/phase3_lidar")


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from site_scorer import score_site
from image_classifier import classify_imagery
from lidar_analyzer import analyze_bbox
from streetview_semantic import analyze_location as streetview_analyze


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="GeoSense Agent MCP Server",
    version="1.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8501",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Latest GeoAI agent result
# ---------------------------------------------------------

LAST_AGENT_RESULT = {}


@app.get(
    "/agent/latest_result",
    summary="Get the latest GeoAI agent result"
)
def get_latest_result():
    return LAST_AGENT_RESULT


# ---------------------------------------------------------
# Input models
# ---------------------------------------------------------

class LocationInput(BaseModel):
    latitude: float = Field(
        ...,
        description="Latitude in WGS84 decimal degrees"
    )

    longitude: float = Field(
        ...,
        description="Longitude in WGS84 decimal degrees"
    )


class BboxInput(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


# ---------------------------------------------------------
# Tool 1 — Site suitability scoring
# ---------------------------------------------------------

@app.post(
    "/tools/score_location",
    summary="Score a location for site suitability using ML"
)
def tool_score_location(payload: LocationInput):

    return score_site(
        payload.latitude,
        payload.longitude
    )


# ---------------------------------------------------------
# Tool 2 — Satellite imagery classification
# ---------------------------------------------------------

@app.post(
    "/tools/classify_imagery",
    summary="Classify satellite imagery land cover at a location"
)
def tool_classify_imagery(payload: LocationInput):

    return classify_imagery(
        payload.latitude,
        payload.longitude,
        radius_m=500
    )


# ---------------------------------------------------------
# Tool 3 — LiDAR profile
# ---------------------------------------------------------

@app.post(
    "/tools/lidar_profile",
    summary="Get LiDAR-derived building height, canopy, and slope metrics"
)
def tool_lidar_profile(payload: BboxInput):

    return analyze_bbox(
        payload.min_lat,
        payload.min_lon,
        payload.max_lat,
        payload.max_lon
    )


# ---------------------------------------------------------
# Tool 4 — Street-level semantic scene analysis
# ---------------------------------------------------------

@app.post(
    "/tools/streetview_scene",
    summary="Get 360-degree street scene composition and walkability score"
)
def tool_streetview_scene(payload: LocationInput):

    return streetview_analyze(
        payload.latitude,
        payload.longitude
    )


# ---------------------------------------------------------
# Tool 5 — Climate risk
# ---------------------------------------------------------

@app.post(
    "/tools/get_climate_risk",
    summary="Get temperature anomaly and drought risk for a location"
)
def tool_get_climate_risk(payload: LocationInput):

    return {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "temperature_anomaly_c": 1.4,
        "drought_index": 0.3,
    }


# ---------------------------------------------------------
# Tool 6 — Demographics
# ---------------------------------------------------------

@app.post(
    "/tools/get_demographics",
    summary="Get population density and healthcare access near a location"
)
def tool_get_demographics(payload: LocationInput):

    return {
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "population_density_per_km2": 8200,
        "nearest_hospital_km": 1.2,
    }


# ---------------------------------------------------------
# Tool 7 — Find similar sites
# ---------------------------------------------------------

@app.post(
    "/tools/find_similar_sites",
    summary="Find sites with a similar profile to a reference location"
)
def tool_find_similar_sites(payload: LocationInput):

    reference = score_site(
        payload.latitude,
        payload.longitude
    )

    return {
        "reference": reference,
        "similar_sites": [],
    }


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )