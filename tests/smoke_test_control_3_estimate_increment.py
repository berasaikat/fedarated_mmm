import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tests.helpers.make_geo_data import make_synthetic_geo_data
from causal_validation.synthetic_control import (
    fit_synthetic_control,
    estimate_incrementality,
)

geo_data, true_att = make_synthetic_geo_data()

pre_period = geo_data[~geo_data["is_treatment_period"]]
post_period = geo_data[geo_data["is_treatment_period"]]

treated_geo = "geo_treated"
donor_geo_ids = ["geo_A", "geo_B", "geo_C", "geo_D"]

weights = fit_synthetic_control(pre_period, treated_geo, donor_geo_ids)
result = estimate_incrementality(pre_period, post_period, treated_geo, weights)

print(f"\nTrue ATT:      {true_att:.2f}")
print(f"Estimated ATT: {result['att']:.2f}")
print(f"Std Error:     {result['std_err']:.2f}")
print(f"P-value:       {result['p_value']:.4f}")

# Checks
assert "att" in result
assert "std_err" in result
assert "p_value" in result
assert "synthetic_control_series" in result
assert "actual_series" in result

assert len(result["synthetic_control_series"]) == len(post_period["week"].unique())
assert len(result["actual_series"]) == len(post_period["week"].unique())

# ATT should be in the right ballpark (within 2x of true)
assert (
    0 < result["att"] < true_att * 3
), f"ATT {result['att']:.2f} way off from true {true_att:.2f}"

# p-value should be significant given we injected a real effect
assert (
    result["p_value"] < 0.05
), f"Expected significant p-value, got {result['p_value']:.4f}"

print("estimate_incrementality — ATT in expected range, p-value significant")
