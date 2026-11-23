"""
Tests for Feature Pipeline, Candidate Models, and Model Artifact.

Verifies:
1. Feature/target separation (record_id and date excluded; delivery_min isolated).
2. Preprocessor handling of missing values (median imputation).
3. Preprocessor handling of unseen categorical levels (handle_unknown='ignore').
4. The presence and validity of all seven candidate model configurations.
5. Champion model artifact loading (models/model.joblib) and inference.
6. Deterministic prediction reproducibility.
7. Isolated MLflow logging in a temporary database fixture.
"""

from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.mp191.evaluate import evaluate_regression
from src.mp191.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    get_feature_target_matrices,
)
from src.mp191.train import get_model_candidates


@pytest.fixture
def sample_raw_feature_df() -> pd.DataFrame:
    """Creates a sample DataFrame matching the 8 prediction features."""
    return pd.DataFrame(
        [
            {
                "distance_km": 12.5,
                "tiffins": 30,
                "handovers": 2,
                "wait_at_station_min": 10.0,
                "rider_age": 35,
                "temp_c": 28.5,
                "transport": "scooter",
                "route_zone": "Central",
            },
            {
                "distance_km": 5.0,
                "tiffins": 15,
                "handovers": 1,
                "wait_at_station_min": 5.0,
                "rider_age": 25,
                "temp_c": 32.0,
                "transport": "bicycle",
                "route_zone": "Old City",
            },
        ]
    )


def test_feature_target_separation():
    """Checks that record_id and date are excluded and delivery_min is not in X."""
    df = pd.DataFrame(
        {
            "record_id": ["R00001", "R00002"],
            "date": ["2026-01-01", "2026-01-02"],
            "distance_km": [10.0, 15.0],
            "tiffins": [20, 30],
            "handovers": [2, 3],
            "wait_at_station_min": [12.0, 15.0],
            "rider_age": [30, 40],
            "temp_c": [28.0, 30.0],
            "transport": ["scooter", "bicycle"],
            "route_zone": ["Central", "Suburb"],
            "delivery_min": [45.0, 60.0],
        }
    )

    X, y = get_feature_target_matrices(df)

    assert "record_id" not in X.columns
    assert "date" not in X.columns
    assert "delivery_min" not in X.columns
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert y is not None
    assert len(y) == 2


def test_preprocessor_handles_missing_values(sample_raw_feature_df: pd.DataFrame):
    """Checks that SimpleImputer successfully handles missing numerical values."""
    preprocessor = build_preprocessor()
    preprocessor.fit(sample_raw_feature_df)

    # Introduce NaNs in numeric columns
    df_with_nan = sample_raw_feature_df.copy()
    df_with_nan.loc[0, "distance_km"] = np.nan
    df_with_nan.loc[1, "temp_c"] = np.nan

    transformed = preprocessor.transform(df_with_nan)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_categories(
    sample_raw_feature_df: pd.DataFrame,
):
    """Checks that OneHotEncoder with handle_unknown='ignore' silently handles unseen levels."""
    preprocessor = build_preprocessor()
    preprocessor.fit(sample_raw_feature_df)

    # Introduce an unseen category not in the training sample
    df_unseen = sample_raw_feature_df.copy()
    df_unseen.loc[0, "route_zone"] = "Industrial_Port"
    df_unseen.loc[1, "transport"] = "metro_tram"

    # Must transform without raising an exception
    transformed = preprocessor.transform(df_unseen)
    assert transformed.shape[0] == 2
    assert not np.isnan(transformed).any()


def test_seven_model_configurations_defined():
    """Verifies that all seven required candidate configurations are present."""
    candidates = get_model_candidates(random_seed=42)
    assert len(candidates) == 7

    expected_names = [
        "linear_regression_ols",
        "ridge_l2_regularized",
        "rf_shallow_conservative",
        "rf_deep_capacity",
        "hgb_baseline",
        "hgb_conservative",
        "hgb_deep_fast",
    ]
    for name in expected_names:
        assert name in candidates
        assert "family" in candidates[name]
        assert "model" in candidates[name]
        assert "params" in candidates[name]


def test_model_artifact_loading_and_inference(
    sample_raw_feature_df: pd.DataFrame,
):
    """Verifies that the saved models/model.joblib loads and generates valid predictions."""
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "models/model.joblib must exist"

    pipeline = joblib.load(model_path)
    assert isinstance(pipeline, Pipeline)

    preds = pipeline.predict(sample_raw_feature_df)
    assert len(preds) == len(sample_raw_feature_df)
    # Physical validity: delivery time must be positive
    assert (preds > 0).all()


def test_model_prediction_reproducibility(sample_raw_feature_df: pd.DataFrame):
    """Verifies that running predictions on the same input produces identical results."""
    model_path = Path("models/model.joblib")
    pipeline = joblib.load(model_path)

    pred1 = pipeline.predict(sample_raw_feature_df)
    pred2 = pipeline.predict(sample_raw_feature_df)
    np.testing.assert_array_almost_equal(pred1, pred2)


def test_evaluate_regression_metrics():
    """Tests the calculation of MAE, RMSE, and R2."""
    y_true = np.array([40.0, 50.0, 60.0])
    y_pred = np.array([42.0, 48.0, 61.0])

    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(1.6667, rel=1e-3)
    assert metrics["rmse"] == pytest.approx(1.7321, rel=1e-3)
    assert metrics["r2"] > 0.9


def test_isolated_mlflow_tracking(tmp_path: Path):
    """Verifies MLflow logging using an isolated temporary SQLite database."""
    temp_db = tmp_path / "test_mlflow.db"
    temp_tracking_uri = f"sqlite:///{temp_db.as_posix()}"
    temp_exp_name = "test-experiment"

    mlflow.set_tracking_uri(temp_tracking_uri)
    mlflow.set_experiment(temp_exp_name)

    with mlflow.start_run(run_name="test_run") as run:
        mlflow.log_param("test_param", 123)
        mlflow.log_metric("mae", 2.5)

    client = mlflow.tracking.MlflowClient(temp_tracking_uri)
    exp = client.get_experiment_by_name(temp_exp_name)
    assert exp is not None

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["test_param"] == "123"
    assert runs[0].data.metrics["mae"] == 2.5
