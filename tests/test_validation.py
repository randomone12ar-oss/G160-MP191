"""
Tests for src/mp191/data_validation.py.

Verifies the programmatic detection algorithms for the seven data quality issues:
1. leaking_column
2. unit_mixup
3. sentinel_value
4. duplicate_rows
5. stuck_sensor
6. target_typos
7. spelling_variants
"""

import pandas as pd
import pytest

from src.mp191.data_validation import (
    detect_duplicate_rows,
    detect_leaking_column,
    detect_sentinel_value,
    detect_spelling_variants,
    detect_stuck_sensor,
    detect_target_typos,
    detect_unit_mixup,
)


def test_detect_leaking_column():
    """Checks that a column present in training but absent in holdout is flagged."""
    train_df = pd.DataFrame(
        {"feat1": [1, 2], "feat2": [3, 4], "delivery_min": [30.0, 45.0]}
    )
    holdout_df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})

    res = detect_leaking_column(
        train_df, holdout_df, target_col="delivery_min"
    )
    assert res["present"] is True
    assert res["column"] == "delivery_min"

    # When target is not missing in holdout (or not present)
    non_leaking_df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    res_clean = detect_leaking_column(
        non_leaking_df, holdout_df, target_col="delivery_min"
    )
    assert res_clean["present"] is False


def test_detect_unit_mixup():
    """Checks that a 1000x difference in group medians is detected."""
    df_mixed = pd.DataFrame(
        {
            "route_zone": [
                "Central",
                "Central",
                "Old City",
                "Old City",
                "Suburb",
                "Suburb",
            ],
            "distance_km": [10.0, 11.0, 9.5, 10.5, 9500.0, 10500.0],
        }
    )
    res = detect_unit_mixup(
        df_mixed, group_col="route_zone", value_col="distance_km"
    )
    assert res["present"] is True
    assert res["column"] == "distance_km"
    assert res["group_value"] == "Suburb"
    assert res["factor"] == 1000

    # Clean uniform data
    df_uniform = pd.DataFrame(
        {
            "route_zone": ["Central", "Central", "Suburb", "Suburb"],
            "distance_km": [10.0, 11.0, 12.0, 13.0],
        }
    )
    res_clean = detect_unit_mixup(
        df_uniform, group_col="route_zone", value_col="distance_km"
    )
    assert res_clean["present"] is False


def test_detect_sentinel_value():
    """Checks that impossible special values (-999, 9999) are flagged."""
    df_with_sentinel = pd.DataFrame(
        {
            "tiffins": [20, 25, 30],
            "handovers": [1, 2, 3],
            "wait_at_station_min": [-999.0, 12.0, 15.0],
            "rider_age": [30, 35, 40],
            "temp_c": [28.0, 30.0, 32.0],
        }
    )
    res = detect_sentinel_value(df_with_sentinel)
    assert res["present"] is True
    assert res["column"] == "wait_at_station_min"
    assert res["value"] == -999

    # Clean data without sentinels
    df_clean = pd.DataFrame(
        {
            "tiffins": [20, 25, 30],
            "handovers": [1, 2, 3],
            "wait_at_station_min": [10.0, 12.0, 15.0],
            "rider_age": [30, 35, 40],
            "temp_c": [28.0, 30.0, 32.0],
        }
    )
    res_clean = detect_sentinel_value(df_clean)
    assert res_clean["present"] is False


def test_detect_duplicate_rows():
    """Checks that duplicated rows are accurately counted."""
    df_dups = pd.DataFrame(
        {"a": [1, 1, 2], "b": [10, 10, 20]}  # 1 duplicate row
    )
    res = detect_duplicate_rows(df_dups)
    assert res["present"] is True
    assert res["count"] == 1

    df_unique = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    res_clean = detect_duplicate_rows(df_unique)
    assert res_clean["present"] is False
    assert res_clean["count"] is None


def test_detect_stuck_sensor():
    """Checks that consecutive identical readings over a threshold are identified."""
    dates = pd.date_range("2026-05-01", periods=20, freq="D").astype(str)
    # Consecutive run of 20 identical readings
    df_stuck = pd.DataFrame(
        {
            "date": dates,
            "rider_age": [23] * 20,
            "distance_km": list(range(1, 21)),
            "tiffins": [20] * 5 + [25] * 15,
            "handovers": [2] * 20,
            "wait_at_station_min": [10.0] * 20,
            "temp_c": [30.0] * 20,
        }
    )
    res = detect_stuck_sensor(
        df_stuck,
        date_col="date",
        numeric_cols=["rider_age"],
        min_run_threshold=15,
    )
    assert res["present"] is True
    assert res["column"] == "rider_age"
    assert res["stuck_value"] == 23
    assert res["start_date"] == "2026-05-01"
    assert res["end_date"] == "2026-05-20"


def test_detect_target_typos():
    """Checks that 10x typos sitting above the distribution gap are detected."""
    df_typos = pd.DataFrame(
        {
            "delivery_min": [
                30.0,
                35.0,
                40.0,
                75.0,  # normal distribution spans [30.0, 75.0]
                400.0,  # 400.0 / 10 = 40.0 (in range)
                650.0,  # 650.0 / 10 = 65.0 (in range)
            ]
        }
    )
    res = detect_target_typos(
        df_typos, target_col="delivery_min", gap_threshold=100.0
    )
    assert res["present"] is True
    assert res["count"] == 2
    assert res["consistent_with_10x"] is True

    # Clean targets
    df_clean = pd.DataFrame({"delivery_min": [25.0, 30.0, 45.0, 60.0]})
    res_clean = detect_target_typos(
        df_clean, target_col="delivery_min", gap_threshold=100.0
    )
    assert res_clean["present"] is False


def test_detect_spelling_variants():
    """Checks detection of whitespace or casing variations in categories."""
    df_noisy = pd.DataFrame(
        {"transport": ["bicycle", "Bicycle ", "bicycle", "scooter"]}
    )
    res = detect_spelling_variants(df_noisy, categorical_cols=["transport"])
    assert res["present"] is True
    assert res["column"] == "transport"
    assert res["raw_levels"] == 3
    assert res["true_levels"] == 2

    # Clean categories
    df_clean = pd.DataFrame(
        {"transport": ["bicycle", "bicycle", "scooter", "local_train"]}
    )
    res_clean = detect_spelling_variants(
        df_clean, categorical_cols=["transport"]
    )
    assert res_clean["present"] is False
