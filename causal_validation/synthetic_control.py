import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import nnls
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def fit_synthetic_control(
    pre_period_df: pd.DataFrame, treated_geo_id: str, donor_geo_ids: List[str]
) -> Dict[str, float]:
    """
    Fits donor weights to minimize pre-period RMSE mathematically using non-negative least squares (or simple bounded OLS).

    Args:
        pre_period_df: DataFrame containing structural rows matching 'week', 'geo_id', and 'revenue'.
        treated_geo_id: The identifier of the treated market region.
        donor_geo_ids: The list of candidate proxy geo_ids generated from a GeoMatcher.

    Returns:
        dict: A strict mapping defining the geometric distribution of the donor_geo_id scalar weights.
    """
    # 1. Pivot timeline dimensions to isolate the independent geo matrices
    df_pivot = pre_period_df.pivot(
        index="week", columns="geo_id", values="revenue"
    ).fillna(0)

    if treated_geo_id not in df_pivot.columns:
        raise ValueError(
            f"Target treated dimensional key '{treated_geo_id}' not found entirely in pre_period_df DataFrame."
        )

    y_actual = df_pivot[treated_geo_id].values

    # Secure donor candidates ensuring column structures actually persist locally
    available_donors = [g for g in donor_geo_ids if g in df_pivot.columns]
    if not available_donors:
        raise ValueError(
            "None of the specified `donor_geo_ids` physically exist within the `pre_period_df` matrix blocks."
        )

    X_matrix = df_pivot[available_donors].values

    # 2. Extract Mathematical Regression
    # Standard SC best-practice utilizes strictly NNLS to enforce valid causal bounding constraints
    weights, rnorm = nnls(X_matrix, y_actual)

    y_synth_pre = X_matrix @ weights
    rmse = float(np.sqrt(np.mean((y_actual - y_synth_pre) ** 2)))
    r_squared = float(
        1
        - np.sum((y_actual - y_synth_pre) ** 2)
        / np.sum((y_actual - np.mean(y_actual)) ** 2)
    )

    logger.info(
        f"Synthetic control pre-period fit — RMSE: {rmse:.2f}, R²: {r_squared:.4f}"
    )

    if r_squared < 0.7:
        logger.warning(
            f"Poor pre-period fit (R²={r_squared:.3f}) — ATT estimate may be unreliable"
        )

    return {geo_id: float(weight) for geo_id, weight in zip(available_donors, weights)}


def estimate_incrementality(
    pre_period_df: pd.DataFrame,
    post_period_df: pd.DataFrame,
    treated_geo_id: str,
    donor_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Fires the fitted target weights against structural post-period empirical constraints to statically
    resolve the underlying Causal Treatment efficacy vs the projected Synthetic baseline.

    Args:
        pre_period_df: Historical context mapping preceding causal interventions (optional param logic tracking, skipped if implicit).
        post_period_df: Post-intervention context to extrapolate performance divergence over.
        treated_geo_id: True string-index identifier of the explicitly targeted market.
        donor_weights: Explicit scalar dictionary representing bounds optimized via `fit_synthetic_control`.

    Returns:
        Dict returning isolated causal validations matching explicitly:
        - att: Average pointwise Treatment Effect strictly on the Treated (ATT)
        - std_err: Geometric distribution error associated internally to the outcome variance
        - p_value: Statistical validity resolving independent identical distribution t-tests
        - synthetic_control_series: Computed scalar baseline payload vector evaluating identical timeframe
        - actual_series: True empirically observed revenue outputs payload matching treated_geo_id directly
    """
    df_post_pivot = post_period_df.pivot(
        index="week", columns="geo_id", values="revenue"
    ).fillna(0)

    if treated_geo_id not in df_post_pivot.columns:
        raise ValueError(
            f"Treated geo target '{treated_geo_id}' does not appear within the structural post_period_df bounds."
        )

    y_actual = df_post_pivot[treated_geo_id].values

    y_synth = np.zeros(len(y_actual))
    for donor, weight in donor_weights.items():
        if donor in df_post_pivot.columns:
            y_synth += df_post_pivot[donor].values * weight

    # Pointwise isolated differential scalar series
    treatment_effect = y_actual - y_synth

    # 1. ATT metric resolving geometric average scalar divergence
    att = float(np.mean(treatment_effect))

    n_obs = len(treatment_effect)

    # 2. Standard Error representing local target statistical confidence divergence variance
    if n_obs > 1:
        # Utilize internal degrees of freedom = 1 (sample error standard representation natively)
        std_err = float(np.std(treatment_effect, ddof=1) / np.sqrt(n_obs))
    else:
        std_err = 0.0

    # 3. Two-Sided Null Hypothesis Statistical P-Value resolution
    if std_err > 0:
        t_stat = att / std_err
        # P-value resolves the identical theoretical null-constraint distance bounds (alpha)
        p_val = float(stats.t.sf(np.abs(t_stat), df=n_obs - 1) * 2)
    else:
        p_val = 1.0

    return {
        "att": att,
        "std_err": std_err,
        "p_value": p_val,
        "synthetic_control_series": y_synth.tolist(),
        "actual_series": y_actual.tolist(),
    }
