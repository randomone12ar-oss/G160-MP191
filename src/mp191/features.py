"""
Feature Engineering and Preprocessing Pipeline for MP191.

Defines the ColumnTransformer pipeline:
- Numerical features: SimpleImputer(strategy='median') + StandardScaler()
- Categorical features: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
- Excludes 'record_id' and 'date' from the feature matrix
- Prevents 'delivery_min' target leakage
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical feature definitions
NUMERIC_FEATURES: List[str] = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
]

CATEGORICAL_FEATURES: List[str] = [
    "transport",
    "route_zone",
]

TARGET_COLUMN: str = "delivery_min"
EXCLUDED_COLUMNS: List[str] = ["record_id", "date"]


def build_preprocessor(
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Builds a leak-free scikit-learn ColumnTransformer.

    Viva Explanation:
    - What it does: Combines median imputation and standardization for numeric features,
      and one-hot encoding for categorical features into a single transformer.
    - Why we need it: Standardizing distance and temperature helps linear models converge,
      median imputation provides robustness to potential missing values at test/inference time,
      and handle_unknown='ignore' allows the encoder to silently handle unseen categories without crashing.
    - How it works: ColumnTransformer directs subsets of columns to their designated sub-pipelines
      and concatenates the processed arrays.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drops any extra columns not in the feature list
    )

    return preprocessor


def get_feature_target_matrices(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the feature matrix X and target series y from a DataFrame.
    Guarantees that record_id, date, and delivery_min are never present in X.
    """
    feature_cols = numeric_features + categorical_features
    # Verify all required features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = df[feature_cols].copy()

    # Target extraction
    if target_col in df.columns:
        y = df[target_col].copy()
    else:
        y = None

    return X, y
