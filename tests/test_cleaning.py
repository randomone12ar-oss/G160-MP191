"""
Tests for src/mp191/data_cleaning.py.

Verifies:
1. Distance unit correction (/1000 for Suburb in meters).
2. Target typo correction (/10 for >= 200 min).
3. Preservation of legitimate age 23 riders outside the stuck sensor period.
4. Input immutability (functions do not mutate original DataFrames).
5. Cleaning reproducibility.
6. Row count and column schema preservation.
7. SHA-256 data hash verification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.mp191.data_cleaning import (
    TRAIN_SHA256,
    clean_dataset,
    clean_distance_units,
    clean_stuck_sensor,
    clean_target_typos,
    verify_raw_data_hash,
)


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Creates a controlled synthetic DataFrame with known anomalies."""
    return pd.DataFrame(
        {
            "record_id": [f"R{i:05d}" for i in range(1, 7)],
            "date": [
                "2026-04-15",  # Outside stuck window, rider_age = 23 (should be preserved)
                "2026-05-05",  # Inside stuck window, rider_age = 23 (should be imputed)
                "2026-05-10",  # Inside stuck window, rider_age = 35 (should be preserved)
                "2026-06-01",  # Outside stuck window, rider_age = 23 (should be preserved)
                "2026-02-10",  # Normal record
                "2026-03-12",  # Normal record
            ],
            "distance_km": [
                15000.0,  # Suburb in meters -> should become 15.0
                12.5,  # Suburb already in km -> should remain 12.5
                10.0,  # Central -> should remain 10.0
                8.0,  # Old City -> should remain 8.0
                18500.0,  # Suburb in meters -> should become 18.5
                5.0,  # Central -> should remain 5.0
            ],
            "tiffins": [20, 30, 40, 15, 25, 35],
            "handovers": [2, 3, 1, 2, 4, 2],
            "wait_at_station_min": [10.0, 15.0, 5.0, 12.0, 8.0, 14.0],
            "rider_age": [23, 23, 35, 23, 45, 55],
            "temp_c": [28.0, 32.0, 30.0, 27.0, 29.0, 31.0],
            "transport": [
                "bicycle",
                "scooter",
                "local_train",
                "bicycle",
                "scooter",
                "local_train",
            ],
            "route_zone": [
                "Suburb",
                "Suburb",
                "Central",
                "Old City",
                "Suburb",
                "Central",
            ],
            "delivery_min": [
                450.0,  # Typo (45.0 * 10) -> should become 45.0
                55.0,  # Normal -> should remain 55.0
                763.0,  # Typo (76.3 * 10) -> should become 76.3
                35.0,  # Normal -> should remain 35.0
                139.1,  # Edge legitimate later value (< 200) -> should remain 139.1
                40.0,  # Normal -> should remain 40.0
            ],
        }
    )


def test_clean_distance_units_scaling(synthetic_raw_df: pd.DataFrame):
    """Checks that distance in meters for Suburb (>100) is converted to km, while others remain unchanged."""
    cleaned = clean_distance_units(synthetic_raw_df)

    # Row 0: Suburb 15000.0 m -> 15.0 km
    assert cleaned.loc[0, "distance_km"] == pytest.approx(15.0)

    # Row 1: Suburb 12.5 km (already in km, not > 100) -> remains 12.5
    assert cleaned.loc[1, "distance_km"] == pytest.approx(12.5)

    # Row 2: Central 10.0 km -> remains 10.0
    assert cleaned.loc[2, "distance_km"] == pytest.approx(10.0)

    # Row 4: Suburb 18500.0 m -> 18.5 km
    assert cleaned.loc[4, "distance_km"] == pytest.approx(18.5)


def test_clean_target_typos(synthetic_raw_df: pd.DataFrame):
    """Checks that 10x typographical errors (>= 200) are corrected and normal values preserved."""
    cleaned = clean_target_typos(synthetic_raw_df)

    # Row 0: 450.0 -> 45.0
    assert cleaned.loc[0, "delivery_min"] == pytest.approx(45.0)

    # Row 1: 55.0 -> remains 55.0
    assert cleaned.loc[1, "delivery_min"] == pytest.approx(55.0)

    # Row 2: 763.0 -> 76.3
    assert cleaned.loc[2, "delivery_min"] == pytest.approx(76.3)

    # Row 4: 139.1 (legitimate production value < 200) -> remains 139.1
    assert cleaned.loc[4, "delivery_min"] == pytest.approx(139.1)


def test_clean_stuck_sensor_preservation_outside_window(
    synthetic_raw_df: pd.DataFrame,
):
    """Checks that rider_age=23 is imputed ONLY within 2026-05-02 to 2026-05-19, and preserved elsewhere."""
    cleaned = clean_stuck_sensor(synthetic_raw_df)

    # Row 0: 2026-04-15 with rider_age=23 (before stuck window) -> MUST remain 23
    assert cleaned.loc[0, "rider_age"] == 23

    # Row 1: 2026-05-05 with rider_age=23 (inside window) -> MUST be imputed with median of valid readings
    # Valid readings: rows 0 (23), 2 (35), 3 (23), 4 (45), 5 (55) -> median is 35
    assert cleaned.loc[1, "rider_age"] != 23
    assert cleaned.loc[1, "rider_age"] == 35

    # Row 2: 2026-05-10 with rider_age=35 (inside window, but not stuck_value 23) -> remains 35
    assert cleaned.loc[2, "rider_age"] == 35

    # Row 3: 2026-06-01 with rider_age=23 (after stuck window) -> MUST remain 23
    assert cleaned.loc[3, "rider_age"] == 23


def test_clean_dataset_input_immutability(synthetic_raw_df: pd.DataFrame):
    """Verifies that clean_dataset does not mutate the original input DataFrame."""
    original_copy = synthetic_raw_df.copy(deep=True)
    _ = clean_dataset(synthetic_raw_df, is_train=True)
    pd.testing.assert_frame_equal(synthetic_raw_df, original_copy)


def test_clean_dataset_reproducibility(synthetic_raw_df: pd.DataFrame):
    """Verifies that running clean_dataset multiple times produces deterministic, identical results."""
    run1 = clean_dataset(synthetic_raw_df, is_train=True)
    run2 = clean_dataset(synthetic_raw_df, is_train=True)
    pd.testing.assert_frame_equal(run1, run2)


def test_clean_dataset_preserves_row_count_and_columns(
    synthetic_raw_df: pd.DataFrame,
):
    """Verifies that no rows are dropped and all schema columns are preserved."""
    cleaned = clean_dataset(synthetic_raw_df, is_train=True)
    assert len(cleaned) == len(synthetic_raw_df)
    assert list(cleaned.columns) == list(synthetic_raw_df.columns)
    assert not cleaned.isnull().values.any()


def test_verify_raw_data_hash_success_and_failure(tmp_path: Path):
    """Tests the cryptographic verification function for exact match and tamper detection."""
    test_file = tmp_path / "sample.csv"
    test_file.write_bytes(b"col1,col2\n1,2\n")

    # Correct hash
    import hashlib

    expected = hashlib.sha256(b"col1,col2\n1,2\n").hexdigest()
    assert verify_raw_data_hash(test_file, expected) is True

    # Tampered hash
    with pytest.raises(ValueError, match="Integrity check failed"):
        verify_raw_data_hash(test_file, "00000000000000000000000000000000")

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        verify_raw_data_hash(tmp_path / "non_existent.csv", expected)
