import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from causal_validation.synthetic_control import fit_synthetic_control, estimate_incrementality

logger = logging.getLogger(__name__)

def run_incrementality_audit(
    global_summary: Dict[str, Dict[str, float]], 
    geo_data: pd.DataFrame, 
    matched_geos: Dict[str, Any], 
    channel_to_audit: str
) -> Dict[str, Any]:
    """
    Empirically bridges the theoretical federated MMM outputs structurally against raw causal Synthetic 
    Control results mapping mathematically identical interventions.
    
    Args:
        global_summary: The global FedAvg mathematical posterior summary dictionary.
        geo_data: Structural pandas DataFrame holding continuous chronological evaluation arrays.
        matched_geos: Dictionary boundary mapping {"treated_geo_id": str, "donor_geo_ids": List[str]}.
        channel_to_audit: The target programmatic scalar variable establishing the test intersection.
        
    Returns:
        dict: A completely structured map returning absolute validations:
              {'channel', 'mmm_beta_mean', 'mmm_beta_ci', 'att_estimate', 'coverage', 'gap'}
    """
    treated_geo_id = matched_geos.get("treated_geo_id")
    donor_geo_ids = matched_geos.get("donor_geo_ids", [])
    
    if not treated_geo_id or not donor_geo_ids:
        raise ValueError("The 'matched_geos' object strictly requires configured 'treated_geo_id' and 'donor_geo_ids' definitions explicitly.")
        
    # Intelligent Chronological Splits (Extrapolating implicitly over structural timelines natively if explicit columns are missing)
    if "is_treatment_period" in geo_data.columns:
        pre_period_df = geo_data[~geo_data["is_treatment_period"]]
        post_period_df = geo_data[geo_data["is_treatment_period"]]
    elif "period" in geo_data.columns:
        pre_period_df = geo_data[geo_data["period"] == "pre"]
        post_period_df = geo_data[geo_data["period"] == "post"]
    else:
        # Bound mathematically splitting independent distributions halfway
        weeks = sorted(geo_data["week"].unique())
        midpoint = len(weeks) // 2
        pre_weeks = weeks[:midpoint]
        pre_period_df = geo_data[geo_data["week"].isin(pre_weeks)]
        post_period_df = geo_data[~geo_data["week"].isin(pre_weeks)]
        
    # 1. Trigger robust theoretical Causal validation bounding 
    donor_weights = fit_synthetic_control(pre_period_df, treated_geo_id, donor_geo_ids)
    inc_res = estimate_incrementality(pre_period_df, post_period_df, treated_geo_id, donor_weights)
    
    # 2. Extract structurally modeled Treatment Metrics
    att_estimate = float(inc_res["att"])
    
    # 3. Harvest analytical constraints directly mapped off our federated model aggregation endpoints
    ch_summary = global_summary.get(channel_to_audit, {})
    if not ch_summary:
        logger.warning(f"Target inference metric scalar '{channel_to_audit}' missing from global aggregate evaluations!")
        mmm_beta_mean = 0.0
        mmm_beta_std = 0.0
    else:
        mmm_beta_mean = ch_summary.get("mean", 0.0)
        mmm_beta_std = ch_summary.get("std", 0.15)
        
    # Gaussian 90% Bound Z-Scores
    p5 = ch_summary.get("p5", mmm_beta_mean - 1.645 * mmm_beta_std)
    p95 = ch_summary.get("p95", mmm_beta_mean + 1.645 * mmm_beta_std)
    
    mmm_beta_ci = [float(p5), float(p95)]
    
    # 4. Resolve absolute inference gaps identifying whether correlations actually trace independent causality symmetrically
    coverage = bool(p5 <= att_estimate <= p95)
    gap = float(att_estimate - mmm_beta_mean)
    
    audit_result = {
        "channel": channel_to_audit,
        "mmm_beta_mean": float(mmm_beta_mean),
        "mmm_beta_ci": mmm_beta_ci,
        "att_estimate": att_estimate,
        "coverage": coverage,
        "gap": gap
    }
    
    return audit_result
