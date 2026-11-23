"""
Model Training and MLflow Experiment Tracking for MP191.

Trains and compares seven candidate model configurations across Linear, Random Forest,
and Gradient Boosting algorithms. Tracks all experiments in a local SQLite-backed MLflow
database (sqlite:///mlflow.db), registers the champion model, sets the @champion alias,
and exports the winning pipeline to models/model.joblib.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import yaml

from src.mp191.data_cleaning import TRAIN_SHA256, load_and_clean_train_data
from src.mp191.evaluate import evaluate_regression
from src.mp191.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    get_feature_target_matrices,
)


def get_model_candidates(random_seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns the seven defined model configurations spanning Linear, Random Forest,
    and HistGradientBoosting algorithms.
    """
    return {
        "linear_regression_ols": {
            "family": "linear",
            "model": LinearRegression(),
            "params": {"fit_intercept": True},
        },
        "ridge_l2_regularized": {
            "family": "linear",
            "model": Ridge(alpha=1.0, random_state=random_seed),
            "params": {"alpha": 1.0, "fit_intercept": True},
        },
        "rf_shallow_conservative": {
            "family": "random_forest",
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=4,
                random_state=random_seed,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_leaf": 4,
            },
        },
        "rf_deep_capacity": {
            "family": "random_forest",
            "model": RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=1,
                random_state=random_seed,
            ),
            "params": {
                "n_estimators": 150,
                "max_depth": 12,
                "min_samples_leaf": 1,
            },
        },
        "hgb_baseline": {
            "family": "gradient_boosting",
            "model": HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_seed,
            ),
            "params": {"max_iter": 100, "max_depth": 6, "learning_rate": 0.1},
        },
        "hgb_conservative": {
            "family": "gradient_boosting",
            "model": HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_seed,
            ),
            "params": {"max_iter": 150, "max_depth": 4, "learning_rate": 0.05},
        },
        "hgb_deep_fast": {
            "family": "gradient_boosting",
            "model": HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=random_seed,
            ),
            "params": {"max_iter": 200, "max_depth": 8, "learning_rate": 0.1},
        },
    }


def train_and_track_models(
    config_path: str = "params.yaml",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "mp191-tiffin-delivery",
    registered_model_name: str = "mp191-delivery-predictor",
    models_dir: str = "models",
    findings_path: str = "reports/findings.yaml",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Orchestrates the complete training and MLflow tracking workflow.
    """
    # 1. Load params configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    test_size = config.get("split", {}).get("test_size", 0.2)
    random_seed = config.get("split", {}).get("random_seed", 42)

    # 2. Load and clean raw train.csv (with cryptographic SHA-256 verification)
    raw_train_path = config.get("data", {}).get(
        "raw_train_path", "data/raw/train.csv"
    )
    df = load_and_clean_train_data(raw_train_path, TRAIN_SHA256)

    # 3. Extract feature matrix X and target y
    X, y = get_feature_target_matrices(
        df,
        target_col=TARGET_COLUMN,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # 4. Train/Validation split (leak-free)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # 5. Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    candidates = get_model_candidates(random_seed=random_seed)
    run_results = []

    print("=" * 80)
    print(f"TRAINING {len(candidates)} CANDIDATE REGRESSION MODELS (MLflow: {tracking_uri})")
    print("=" * 80)

    best_pipeline = None
    best_config_name = None
    best_run_id = None
    lowest_mae = float("inf")
    best_metrics = None
    best_params = None

    for config_name, cand in candidates.items():
        with mlflow.start_run(run_name=config_name) as run:
            run_id = run.info.run_id

            # Assemble full pipeline with preprocessor + regressor
            preprocessor = build_preprocessor()
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", cand["model"]),
                ]
            )

            # Fit exclusively on X_train
            pipeline.fit(X_train, y_train)

            # Evaluate on train and validation
            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_metrics = evaluate_regression(y_train, train_preds)
            val_metrics = evaluate_regression(y_val, val_preds)

            # Log parameters
            mlflow.log_param("data_sha256", TRAIN_SHA256)
            mlflow.log_param("config_name", config_name)
            mlflow.log_param("model_family", cand["family"])
            mlflow.log_param("random_seed", random_seed)
            mlflow.log_param("test_size", test_size)
            for p_k, p_v in cand["params"].items():
                mlflow.log_param(p_k, p_v)

            # Log metrics (including standard mae metric required by handbook)
            mlflow.log_metric("mae", val_metrics["mae"])
            mlflow.log_metric("val_mae", val_metrics["mae"])
            mlflow.log_metric("val_rmse", val_metrics["rmse"])
            mlflow.log_metric("val_r2", val_metrics["r2"])
            mlflow.log_metric("train_mae", train_metrics["mae"])

            # Log fitted pipeline model artifact
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            result_entry = {
                "config_name": config_name,
                "family": cand["family"],
                "run_id": run_id,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "train_mae": train_metrics["mae"],
            }
            run_results.append(result_entry)

            print(
                f"[{cand['family'][:6].upper()}] {config_name.ljust(25)} | "
                f"Val MAE: {val_metrics['mae']:.4f} min | "
                f"Val RMSE: {val_metrics['rmse']:.4f} min | "
                f"Val R²: {val_metrics['r2']:.4f}"
            )

            if val_metrics["mae"] < lowest_mae:
                lowest_mae = val_metrics["mae"]
                best_pipeline = pipeline
                best_config_name = config_name
                best_run_id = run_id
                best_metrics = val_metrics
                best_params = cand["params"]

    print("=" * 80)
    print(f"CHAMPION MODEL SELECTED: {best_config_name}")
    print(f"Validation MAE: {lowest_mae:.4f} min | Run ID: {best_run_id}")
    print("=" * 80)

    # 6. Register winning model in MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(
        model_uri=model_uri, name=registered_model_name
    )

    actual_version = reg_model.version
    print(
        f"✓ Model registered under: '{registered_model_name}' (Version: {actual_version})"
    )

    # 7. Assign 'champion' alias
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=str(actual_version),
    )
    print(
        f"✓ Alias '@champion' assigned to {registered_model_name} version {actual_version}"
    )

    # 8. Export winning pipeline to models/model.joblib
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib_path = models_path / "model.joblib"
    joblib.dump(best_pipeline, joblib_path)
    print(f"✓ Serialized pipeline saved to {joblib_path}")

    # 9. Write metadata.json
    metadata = {
        "model_name": best_config_name,
        "run_id": best_run_id,
        "registered_model_name": registered_model_name,
        "champion_version": int(actual_version),
        "validation_metrics": best_metrics,
        "hyperparameters": best_params,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "target": TARGET_COLUMN,
        "data_sha256": TRAIN_SHA256,
        "random_seed": random_seed,
        "all_configurations_evaluated": run_results,
    }
    metadata_path = models_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")

    # 10. Update reports/findings.yaml tracking section
    if Path(findings_path).exists():
        with open(findings_path, "r") as f:
            findings = yaml.safe_load(f)

        if "tracking" not in findings:
            findings["tracking"] = {}

        findings["tracking"]["experiment_name"] = experiment_name
        findings["tracking"]["best_run_id"] = best_run_id
        findings["tracking"]["registered_model_name"] = registered_model_name
        findings["tracking"]["champion_version"] = int(actual_version)

        with open(findings_path, "w") as f:
            yaml.safe_dump(findings, f, sort_keys=False)
        print(f"✓ Updated tracking section in {findings_path}")

    return best_pipeline, metadata


if __name__ == "__main__":
    train_and_track_models()
