import pandas as pd
import numpy as np


def make_synthetic_geo_data(n_weeks=104, geo_ids=None, seed=42):
    """Creates synthetic geo-level revenue data for testing Phase 8."""
    if geo_ids is None:
        geo_ids = ["geo_treated", "geo_A", "geo_B", "geo_C", "geo_D"]

    rng = np.random.default_rng(seed)
    rows = []

    # Ground truth: treated geo has a real lift in post period
    treatment_start = n_weeks // 2
    true_att = 200.0  # known lift for validation

    # Generate a shared base signal all geos follow (market-wide trend)
    shared_signal = rng.normal(1000, 30, size=n_weeks)

    for week_idx in range(n_weeks):
        week = week_idx + 1
        for geo in geo_ids:
            # Each geo = shared signal + geo-specific noise
            geo_noise = rng.normal(0, 15)
            base = shared_signal[week_idx] + geo_noise

            # Inject known treatment effect in post-period for treated geo only
            if geo == "geo_treated" and week >= treatment_start:
                base += true_att

            rows.append(
                {
                    "week": week,
                    "geo_id": geo,
                    "revenue": max(0, base),
                    "is_treatment_period": week >= treatment_start,
                }
            )

    return pd.DataFrame(rows), true_att


def make_geo_metadata(geo_ids=None):
    """Creates synthetic geo metadata."""
    if geo_ids is None:
        geo_ids = ["geo_treated", "geo_A", "geo_B", "geo_C", "geo_D"]

    rows = []
    descriptions = {
        "geo_treated": "Urban, high income, Northeast US, population 2M",
        "geo_A": "Urban, high income, Mid-Atlantic, population 1.8M",
        "geo_B": "Suburban, medium income, Northeast US, population 1.2M",
        "geo_C": "Rural, low income, Southeast US, population 0.5M",
        "geo_D": "Urban, medium income, Midwest US, population 1.5M",
    }
    for geo in geo_ids:
        rows.append(
            {
                "geo_id": geo,
                "region": (
                    "Northeast" if "treated" in geo or geo == "geo_A" else "Other"
                ),
                "population": 2000000 if "treated" in geo else 1000000,
                "median_income": 75000,
                "urbanization_level": "urban",
                "description": descriptions.get(geo, f"{geo} market"),
            }
        )

    return pd.DataFrame(rows)
