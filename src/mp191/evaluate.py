"""
Model Evaluation Metrics for MP191: Tiffin Relay Delivery Time Prediction.

Calculates:
- Mean Absolute Error (MAE) — Primary evaluation metric
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes regression performance metrics.

    Viva Explanation:
    - MAE: Measures the average magnitude of errors in raw units (minutes).
      MAE is robust to outliers and represents the expected arrival mistake for customers.
    - RMSE: Penalizes larger timing errors more heavily due to squaring.
    - R²: Proportion of delivery time variance explained by the model features.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
