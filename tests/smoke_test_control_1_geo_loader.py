import tempfile, os
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tests.helpers.make_geo_data import make_synthetic_geo_data, make_geo_metadata
from causal_validation.geo_loader import load_geo_data

geo_data, _ = make_synthetic_geo_data()
geo_meta = make_geo_metadata()

# Write temp CSVs
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
    geo_data.to_csv(f1, index=False)
    data_path = f1.name

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
    geo_meta.to_csv(f2, index=False)
    meta_path = f2.name

merged = load_geo_data(data_path, meta_path)

# Checks
assert "geo_id" in merged.columns, "geo_id missing after merge"
assert "revenue" in merged.columns, "revenue missing after merge"
assert "region" in merged.columns, "region missing — metadata not joined"
assert "population" in merged.columns, "population missing — metadata not joined"
assert merged.isnull().sum().sum() == 0, "NaN values found after merge"

n_geos = geo_data["geo_id"].nunique()
n_weeks = geo_data["week"].nunique()
assert (
    len(merged) == n_geos * n_weeks
), f"Expected {n_geos * n_weeks} rows, got {len(merged)}"

print(f"geo_loader — merged shape: {merged.shape}")
print(f"Columns: {merged.columns.tolist()}")

# Test missing geo_id fallback
geo_data_no_geo = geo_data.drop(columns=["geo_id"])
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f3:
    geo_data_no_geo.to_csv(f3, index=False)
    no_geo_path = f3.name

merged_fallback = load_geo_data(no_geo_path, meta_path)
assert "geo_id" in merged_fallback.columns, "Should assign default geo_id when missing"
print("geo_loader fallback — default geo_id assigned when missing")

for p in [data_path, meta_path, no_geo_path]:
    os.unlink(p)
