import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tests.helpers.make_geo_data import make_synthetic_geo_data
from causal_validation.synthetic_control import fit_synthetic_control

geo_data, true_att = make_synthetic_geo_data()

# Use only pre-period for fitting
pre_period = geo_data[~geo_data["is_treatment_period"]]

treated_geo = "geo_treated"
donor_geo_ids = ["geo_A", "geo_B", "geo_C", "geo_D"]

weights = fit_synthetic_control(pre_period, treated_geo, donor_geo_ids)

print(f"\nDonor weights: { {k: round(v, 4) for k, v in weights.items()} }")

# Checks
assert set(weights.keys()).issubset(
    set(donor_geo_ids)
), "Weights contain unknown geo IDs"
assert all(v >= 0 for v in weights.values()), "NNLS weights must be non-negative"

# Weights should NOT be forced to sum to 1 after your fix
weight_sum = sum(weights.values())
print(f"Weight sum: {weight_sum:.4f} (not forced to 1.0 after NNLS fix)")

# Pre-period reconstruction quality check
pre_pivot = pre_period.pivot(index="week", columns="geo_id", values="revenue").fillna(0)

y_actual = pre_pivot[treated_geo].values
y_synth = sum(
    pre_pivot[g].values * w for g, w in weights.items() if g in pre_pivot.columns
)

rmse = np.sqrt(np.mean((y_actual - y_synth) ** 2))
r_sq = 1 - np.sum((y_actual - y_synth) ** 2) / np.sum(
    (y_actual - np.mean(y_actual)) ** 2
)

print(f"Pre-period RMSE: {rmse:.2f}")
print(f"Pre-period R²:   {r_sq:.4f}")
assert r_sq > 0.5, f"Poor pre-period fit R²={r_sq:.3f} — synthetic control unreliable"
print("fit_synthetic_control — weights valid, pre-period fit acceptable")
