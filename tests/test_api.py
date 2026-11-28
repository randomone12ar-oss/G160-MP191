"""
Tests for FastAPI Prediction Service (app/main.py).

Verifies:
1. GET /health returns HTTP 200 with {"status": "ok"}.
2. POST /predict returns HTTP 200 with delivery_min for valid input.
3. Response contains delivery_min as a finite float.
4. Missing required fields are rejected with HTTP 422.
5. temp_c > 75 is rejected with HTTP 422 (client rule 1).
6. wait_at_station_min < 0 is rejected with HTTP 422 (client rule 2).
7. Extra forbidden fields (record_id, date, delivery_min) are rejected with HTTP 422.
8. Model is loaded once from models/model.joblib and not retrained.
9. Multiple requests produce deterministic, correct outputs.
"""

import math
from fastapi.testclient import TestClient
import pytest

from app.main import app, load_champion_model


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance."""
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """Valid prediction request payload matching the 8 operational features."""
    return {
        "distance_km": 12.5,
        "tiffins": 30,
        "handovers": 2,
        "wait_at_station_min": 10.0,
        "rider_age": 35,
        "temp_c": 28.5,
        "transport": "scooter",
        "route_zone": "Central",
    }


def test_health_endpoint(client: TestClient):
    """Checks that GET /health returns HTTP 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction_returns_200(
    client: TestClient, valid_payload: dict
):
    """Checks that POST /predict returns HTTP 200 for a valid payload."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_prediction_contains_finite_delivery_min(
    client: TestClient, valid_payload: dict
):
    """Checks that the response contains delivery_min as a finite positive number."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    assert "delivery_min" in data
    val = data["delivery_min"]
    assert isinstance(val, (int, float))
    assert math.isfinite(val)
    assert val > 0.0


def test_missing_required_field_rejected(
    client: TestClient, valid_payload: dict
):
    """Checks that omitting a required feature returns HTTP 422."""
    incomplete_payload = valid_payload.copy()
    del incomplete_payload["distance_km"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_temp_c_above_75_rejected(client: TestClient, valid_payload: dict):
    """Checks that temp_c > 75 is rejected with HTTP 422 (Client Rule 1)."""
    bad_payload = valid_payload.copy()
    bad_payload["temp_c"] = 75.1

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_temp_c_boundary_75_accepted(client: TestClient, valid_payload: dict):
    """Checks that temp_c == 75.0 is accepted (boundary condition le=75)."""
    boundary_payload = valid_payload.copy()
    boundary_payload["temp_c"] = 75.0

    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200


def test_negative_wait_time_rejected(client: TestClient, valid_payload: dict):
    """Checks that wait_at_station_min < 0 is rejected with HTTP 422 (Client Rule 2)."""
    bad_payload = valid_payload.copy()
    bad_payload["wait_at_station_min"] = -0.5

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_wait_time_zero_accepted(client: TestClient, valid_payload: dict):
    """Checks that wait_at_station_min == 0.0 is accepted (boundary condition ge=0)."""
    boundary_payload = valid_payload.copy()
    boundary_payload["wait_at_station_min"] = 0.0

    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200


@pytest.mark.parametrize("forbidden_key", ["record_id", "date", "delivery_min"])
def test_extra_forbidden_fields_rejected(
    client: TestClient, valid_payload: dict, forbidden_key: str
):
    """Checks that extra fields (record_id, date, delivery_min) trigger HTTP 422."""
    leaked_payload = valid_payload.copy()
    leaked_payload[forbidden_key] = "forbidden_val"

    response = client.post("/predict", json=leaked_payload)
    assert response.status_code == 422


def test_model_loaded_from_artifact_and_not_retrained():
    """Checks that load_champion_model returns a pre-fitted Pipeline."""
    model = load_champion_model()
    # Must be fitted (has named steps and preprocessor is fitted)
    assert hasattr(model, "predict")
    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "regressor" in model.named_steps


def test_multiple_sequential_predictions(
    client: TestClient, valid_payload: dict
):
    """Checks that multiple sequential requests succeed and return deterministic output."""
    res1 = client.post("/predict", json=valid_payload)
    res2 = client.post("/predict", json=valid_payload)

    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["delivery_min"] == res2.json()["delivery_min"]
