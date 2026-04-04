import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tests.helpers.make_geo_data import make_synthetic_geo_data, make_geo_metadata
from causal_validation.synthetic_control import (
    fit_synthetic_control,
    estimate_incrementality,
)
from causal_validation.audit import run_incrementality_audit

geo_data, true_att = make_synthetic_geo_data(n_weeks=104, seed=42)

treated_geo = "geo_treated"
donor_geo_ids = ["geo_A", "geo_B", "geo_C", "geo_D"]

matched_geos = {"treated_geo_id": treated_geo, "donor_geo_ids": donor_geo_ids}

# Simulate a global MMM summary
# Beta is in normalized scale (0-1), ATT will be normalized against mean revenue
mean_revenue = geo_data[geo_data["geo_id"] == treated_geo]["revenue"].mean()
normalized_true_att = true_att / mean_revenue
print(f"\nMean revenue (treated): {mean_revenue:.2f}")
print(f"True ATT (raw):          {true_att:.2f}")
print(f"True ATT (normalized):   {normalized_true_att:.4f}")

# Set MMM beta close to normalized ATT so coverage=True
global_summary = {
    "paid_search": {
        "mean": normalized_true_att,
        "std": 0.05,
        "p5": normalized_true_att - 0.10,
        "p95": normalized_true_att + 0.10,
    }
}

result = run_incrementality_audit(
    global_summary=global_summary,
    geo_data=geo_data,
    matched_geos=matched_geos,
    channel_to_audit="paid_search",
)

print(f"\nAudit result:")
for k, v in result.items():
    print(f"  {k}: {v}")

# Structural checks
required_keys = [
    "channel",
    "mmm_beta_mean",
    "mmm_beta_ci",
    "att_estimate_raw",
    "att_estimate_normalized",
    "att_std_err",
    "att_p_value",
    "coverage",
    "gap",
]
for key in required_keys:
    assert key in result, f"Missing key: {key}"
print("All required keys present in audit result")

assert result["channel"] == "paid_search"
assert isinstance(result["coverage"], bool)
assert isinstance(result["gap"], float)
assert len(result["mmm_beta_ci"]) == 2
assert (
    result["mmm_beta_ci"][0] < result["mmm_beta_ci"][1]
), "CI lower bound should be less than upper bound"

# ATT normalized should be reasonable
assert (
    0 < result["att_estimate_normalized"] < 1.0
), f"Normalized ATT {result['att_estimate_normalized']:.4f} out of expected range"

# p-value should be significant
assert (
    result["att_p_value"] < 0.05
), f"Expected significant p-value for known treatment effect"

print(f"Coverage: {result['coverage']}")
print(f"Gap:      {result['gap']:.4f}")
print(f"P-value:  {result['att_p_value']:.4f}")
print("Full audit pipeline works end-to-end")
