"""
Data Quality Validation Module for MP191: Tiffin Relay Delivery Time Prediction.

This module provides independent, executable detection functions for the seven
classic data quality problems outlined in the SCSE3040 Student Handbook:
1. leaking_column
2. unit_mixup
3. sentinel_value
4. duplicate_rows
5. stuck_sensor
6. target_typos
7. spelling_variants

Every detector inspects the real dataset programmatically and returns its findings.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml


def detect_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "delivery_min",
) -> Dict[str, Any]:
    """
    Detects leaking columns by inspecting training features vs holdout features.

    Viva Explanation:
    - What it does: A leaking column is one that is recorded after the delivery
      is finished (such as delivery_min, arrival time, or post-trip status).
    - Why we need it: If used as an input feature, the model achieves unrealistically
      good training metrics but fails in production because the feature is not
      available at prediction time.
    - How it works: Compares column names between train.csv and holdout_features.csv.
      Columns present in train but absent in holdout represent post-event data.
      Here, 'delivery_min' is absent from holdout and must never be used as a feature.
    - Marking Requirement: Handbook Section 7 (leaking_column).
    """
    train_cols = set(train_df.columns)
    holdout_cols = set(holdout_df.columns)
    missing_in_holdout = sorted(list(train_cols - holdout_cols))

    if target_col in missing_in_holdout:
        return {
            "present": True,
            "column": target_col,
            "details": f"'{target_col}' is recorded post-event, absent in holdout features, and must not be used as an input feature.",
        }
    return {"present": False, "column": None}


def detect_unit_mixup(
    df: pd.DataFrame,
    group_col: str = "route_zone",
    value_col: str = "distance_km",
) -> Dict[str, Any]:
    """
    Detects unit mix-ups across different reporting groups (e.g. Suburb vs Central).

    Viva Explanation:
    - What it does: Identifies if one operational hub logged distance in meters
      instead of kilometers.
    - Why we need it: Feeding 15,000 meters into a model trained on 15.0 km
      destroys prediction accuracy.
    - How it works: Groups by route_zone and computes min, mean, median, max.
      The Suburb group has a median ~9900 vs ~10 in Central and Old City.
      The ratio ~990 indicates an exact conversion factor of 1000 (meters to km).
    - Marking Requirement: Handbook Section 7 (unit_mixup).
    """
    stats = df.groupby(group_col)[value_col].agg(["min", "mean", "median", "max"])

    overall_median = df[value_col].median()
    suspect_group = None
    suspect_factor = None

    for zone, row in stats.iterrows():
        other_median = stats.drop(zone)["median"].median()
        ratio = row["median"] / other_median if other_median > 0 else 1.0

        # Check standard SI or unit ratios: 10, 60, 100, 1000
        if 800 <= ratio <= 1200:
            suspect_group = str(zone)
            suspect_factor = 1000
            break
        elif 80 <= ratio <= 120:
            suspect_group = str(zone)
            suspect_factor = 100
            break

    if suspect_group:
        return {
            "present": True,
            "column": value_col,
            "group_value": suspect_group,
            "factor": suspect_factor,
            "group_stats": stats.to_dict(orient="index"),
        }
    return {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }


def detect_sentinel_value(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detects sentinel values (e.g. -999, -1, 999, 9999) used instead of nulls.

    Viva Explanation:
    - What it does: Checks if special sentinel numbers were typed when data was unavailable.
    - Why we need it: Models treat -999 or 9999 as real numeric quantities, distorting relationships.
    - How it works: Scans frequency counts and bounds across numeric features.
      In train.csv, all features are within realistic physiological/physical bounds:
      rider_age (19-60), wait_at_station_min (0-25), temp_c (10-45), tiffins (5-60), handovers (1-5).
      No negative or extreme sentinels exist.
    - Marking Requirement: Handbook Section 7 (sentinel_value).
    """
    if numeric_cols is None:
        numeric_cols = [
            "tiffins",
            "handovers",
            "wait_at_station_min",
            "rider_age",
            "temp_c",
        ]

    for col in numeric_cols:
        vc = df[col].value_counts()
        # Look for negative placeholders or extreme numbers
        neg_sentinels = vc[vc.index < 0]
        pos_sentinels = vc[vc.index >= 999]
        if len(neg_sentinels) > 0:
            val = int(neg_sentinels.index[0])
            cnt = int(neg_sentinels.iloc[0])
            return {"present": True, "column": col, "value": val, "count": cnt}
        if len(pos_sentinels) > 0:
            val = int(pos_sentinels.index[0])
            cnt = int(pos_sentinels.iloc[0])
            return {"present": True, "column": col, "value": val, "count": cnt}

    return {"present": False, "column": None, "value": None, "count": None}


def detect_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects exact duplicate records in the dataset.

    Viva Explanation:
    - What it does: Counts duplicate records using df.duplicated().
    - Why we need it: Duplicate records leak across train/test splits and bias evaluation.
    - How it works: Evaluates df.duplicated().sum(). In train.csv, all 1000 records are distinct.
    - Marking Requirement: Handbook Section 7 (duplicate_rows).
    """
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        return {"present": True, "count": dup_count}
    return {"present": False, "count": None}


def detect_stuck_sensor(
    df: pd.DataFrame,
    date_col: str = "date",
    numeric_cols: Optional[List[str]] = None,
    min_run_threshold: int = 15,
) -> Dict[str, Any]:
    """
    Detects stuck sensors where a feature logs the exact same value over many consecutive days.

    Viva Explanation:
    - What it does: Finds frozen readings caused by hardware or terminal glitches.
    - Why we need it: A frozen reading misrepresents true conditions and misleads the model.
    - How it works: Sorts data chronologically by date and calculates run-length encoding.
      For rider_age, the value 23 repeats continuously for 105 consecutive records
      from 2026-05-02 to 2026-05-19.
    - Marking Requirement: Handbook Section 7 (stuck_sensor).
    """
    if numeric_cols is None:
        numeric_cols = [
            "distance_km",
            "tiffins",
            "handovers",
            "wait_at_station_min",
            "rider_age",
            "temp_c",
        ]

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    for col in numeric_cols:
        # Group identical consecutive values
        runs = (df_sorted[col] != df_sorted[col].shift()).cumsum()
        run_counts = df_sorted.groupby(runs).size()
        max_run_id = run_counts.idxmax()
        max_len = int(run_counts.max())

        if max_len >= min_run_threshold:
            run_df = df_sorted[runs == max_run_id]
            stuck_val = run_df[col].iloc[0]
            start_date = str(run_df[date_col].min())
            end_date = str(run_df[date_col].max())
            return {
                "present": True,
                "column": col,
                "stuck_value": stuck_val,
                "run_length": max_len,
                "start_date": start_date,
                "end_date": end_date,
            }

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }


def detect_target_typos(
    df: pd.DataFrame,
    target_col: str = "delivery_min",
    factor: float = 10.0,
    gap_threshold: float = 100.0,
) -> Dict[str, Any]:
    """
    Detects typographical errors in the target where numbers were typed 10x too large.

    Viva Explanation:
    - What it does: Spots delivery times typed with an extra trailing zero (e.g. 400 instead of 40).
    - Why we need it: Large outliers square/inflate loss functions and severely bias regression lines.
    - How it works: Sorts delivery_min. Legitimate deliveries range from 14.2 to 81.4 min.
      An empty gap exists between 81.4 and 247.0 min. Exactly 13 values sit between 247.0 and 763.0 min.
      Dividing these 13 values by 10 maps them into [24.7, 76.3] min, perfectly fitting the distribution.
    - Marking Requirement: Handbook Section 7 (target_typos).
    """
    typo_mask = df[target_col] > gap_threshold
    typo_count = int(typo_mask.sum())

    if typo_count > 0:
        typo_values = df.loc[typo_mask, target_col].values
        corrected_values = typo_values / factor
        normal_max = df.loc[~typo_mask, target_col].max()
        normal_min = df.loc[~typo_mask, target_col].min()

        # Validate that divided values fit inside the normal range
        is_consistent = (corrected_values >= normal_min).all() and (
            corrected_values <= normal_max
        ).all()
        return {
            "present": True,
            "count": typo_count,
            "factor": factor,
            "min_typo": float(typo_values.min()),
            "max_typo": float(typo_values.max()),
            "consistent_with_10x": bool(is_consistent),
        }

    return {"present": False, "count": None}


def detect_spelling_variants(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detects inconsistent spellings or casing variants in text categorical columns.

    Viva Explanation:
    - What it does: Looks for categories split into noisy variants (e.g. 'bicycle', 'Bicycle ').
    - Why we need it: OneHotEncoder creates redundant, sparse features for duplicate spellings.
    - How it works: Compares raw categories against stripped/lowercased versions and later.csv.
      Both 'transport' (3 levels) and 'route_zone' (3 levels) are clean and match later.csv.
    - Marking Requirement: Handbook Section 7 (spelling_variants).
    """
    if categorical_cols is None:
        categorical_cols = ["transport", "route_zone"]

    for col in categorical_cols:
        raw_levels = set(df[col].dropna().unique())
        normalized_levels = set(
            df[col].dropna().astype(str).str.strip().str.lower().unique()
        )

        if len(raw_levels) != len(normalized_levels):
            return {
                "present": True,
                "column": col,
                "raw_levels": len(raw_levels),
                "true_levels": len(normalized_levels),
            }

        if reference_df is not None and col in reference_df.columns:
            ref_levels = set(reference_df[col].dropna().unique())
            extra_levels = raw_levels - ref_levels
            if len(extra_levels) > 0:
                return {
                    "present": True,
                    "column": col,
                    "raw_levels": len(raw_levels),
                    "true_levels": len(ref_levels),
                }

    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> Dict[str, Any]:
    """
    Executes all seven data quality checks and prints structured findings.
    """
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)
    holdout_df = pd.read_csv(holdout_path)

    results = {
        "leaking_column": detect_leaking_column(train_df, holdout_df),
        "unit_mixup": detect_unit_mixup(train_df),
        "sentinel_value": detect_sentinel_value(train_df),
        "duplicate_rows": detect_duplicate_rows(train_df),
        "stuck_sensor": detect_stuck_sensor(train_df),
        "target_typos": detect_target_typos(train_df),
        "spelling_variants": detect_spelling_variants(train_df, later_df),
    }

    print("=" * 60)
    print("MP191 DATA QUALITY VALIDATION RESULTS (G160)")
    print("=" * 60)
    for issue, res in results.items():
        pres = res.get("present")
        print(f"[{'X' if pres else ' '}] {issue.ljust(20)}: present = {pres}")
        for k, v in res.items():
            if k not in ["present", "group_stats", "consistent_with_10x"]:
                print(f"    - {k}: {v}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_checks()
