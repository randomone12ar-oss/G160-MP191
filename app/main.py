"""
FastAPI Prediction Service for MP191: Tiffin Relay Delivery Time Prediction.

Client: Kochi Dabba Relay Union
Endpoints:
- GET  /health   -> Status check returning {"status": "ok"}
- POST /predict  -> Predicts delivery_min from the 8 operational features

Pydantic validation constraints:
- temp_c <= 75 (values above 75 are physically impossible in Kochi)
- wait_at_station_min >= 0 (negative wait time is always a data entry mistake)
- extra="forbid" (rejects record_id, date, delivery_min, or unexpected fields with HTTP 422)

Logging:
- Structured single-line JSON logging written to stdout
"""

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

# Base directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON strings to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra payload if provided
        if hasattr(record, "structured_data") and isinstance(
            record.structured_data, dict
        ):
            log_obj.update(record.structured_data)
        return json.dumps(log_obj)


# Setup stdout logging
logger = logging.getLogger("mp191.api")
logger.setLevel(logging.INFO)
logger.propagate = False  # Avoid duplicate handlers
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)


# Global model container
model_container: Dict[str, Any] = {}


def load_champion_model(path: Path = MODEL_PATH) -> Any:
    """Loads the fitted pipeline artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)


def get_model() -> Any:
    """Retrieves or loads the champion model singleton."""
    if "model" not in model_container:
        model_container["model"] = load_champion_model()
    return model_container["model"]


app = FastAPI(
    title="MP191 Tiffin Delivery Time Prediction Service",
    description="Production API for Kochi Dabba Relay Union",
    version="1.0.0",
)

# Eagerly load model if available at startup
try:
    get_model()
    logger.info(
        "Champion model loaded successfully",
        extra={"structured_data": {"event": "startup_model_loaded"}},
    )
except Exception as exc:
    logger.warning(
        f"Champion model deferred load: {exc}",
        extra={"structured_data": {"event": "model_deferred_load"}},
    )


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictionRequest(BaseModel):
    """
    Schema for prediction input.
    Rejects forbidden or extra fields (e.g. record_id, date, delivery_min) with HTTP 422.
    """

    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(..., description="Route length in kilometers")
    tiffins: int = Field(..., description="Count of tiffins in the batch")
    handovers: int = Field(..., description="Times the batch changes hands")
    wait_at_station_min: float = Field(
        ...,
        ge=0.0,
        description="Time spent waiting at station in minutes (must be >= 0)",
    )
    rider_age: int = Field(..., description="Age of the main runner in years")
    temp_c: float = Field(
        ...,
        le=75.0,
        description="Temperature at noon in degrees Celsius (must be <= 75)",
    )
    transport: str = Field(
        ..., description="Main transport mode: bicycle, local_train, scooter"
    )
    route_zone: str = Field(
        ..., description="Route zone: Central, Suburb, Old City"
    )


class PredictionResponse(BaseModel):
    delivery_min: float = Field(
        ..., description="Predicted delivery time in minutes"
    )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check endpoint.
    Returns HTTP 200 with {'status': 'ok'}.
    """
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict(request: PredictionRequest):
    """
    Generates a delivery time prediction for an operational batch.
    Validates input constraints, runs the fitted pipeline, and logs structured JSON.
    """
    start_time = time.perf_counter()

    model = model_container.get("model")
    if model is None:
        # Fallback to load if accessed outside lifespan (e.g. direct function call)
        model = load_champion_model()
        model_container["model"] = model

    # Construct input dataframe matching exact training column names
    input_df = pd.DataFrame(
        [
            {
                "distance_km": request.distance_km,
                "tiffins": request.tiffins,
                "handovers": request.handovers,
                "wait_at_station_min": request.wait_at_station_min,
                "rider_age": request.rider_age,
                "temp_c": request.temp_c,
                "transport": request.transport,
                "route_zone": request.route_zone,
            }
        ]
    )

    try:
        raw_pred = model.predict(input_df)[0]
        raw_pred = float(raw_pred)

        if not math.isfinite(raw_pred):
            raise ValueError("Model returned a non-finite prediction")

        delivery_min = round(raw_pred, 2)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Emit structured single-line JSON log to stdout
        logger.info(
            "Prediction executed successfully",
            extra={
                "structured_data": {
                    "event": "prediction_request",
                    "status": "success",
                    "delivery_min": delivery_min,
                    "duration_ms": duration_ms,
                    "route_zone": request.route_zone,
                    "transport": request.transport,
                }
            },
        )

        return PredictionResponse(delivery_min=delivery_min)

    except Exception as e:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            f"Prediction failed: {str(e)}",
            extra={
                "structured_data": {
                    "event": "prediction_request",
                    "status": "error",
                    "duration_ms": duration_ms,
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing prediction request",
        )
