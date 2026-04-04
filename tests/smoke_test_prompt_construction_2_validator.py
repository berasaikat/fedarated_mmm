import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm_prior.validator import validate_priors

channels = ["paid_search", "social", "tv", "ooh"]

# Test 1 — all valid
valid_input = {
    "paid_search": {"mu": 0.35, "sigma": 0.08},
    "social": {"mu": 0.18, "sigma": 0.10},
    "tv": {"mu": 0.25, "sigma": 0.12},
    "ooh": {"mu": 0.10, "sigma": 0.09},
}
result = validate_priors(valid_input, channels)
assert all(ch in result for ch in channels)
print("✓ Test 1 passed — valid input")

# Test 2 — missing channel gets default
partial_input = {"paid_search": {"mu": 0.35, "sigma": 0.08}}
result = validate_priors(partial_input, channels)
assert result["social"]["mu"] == 0.2
assert result["social"]["sigma"] == 0.15
print("✓ Test 2 passed — missing channel filled with default")

# Test 3 — out of bounds mu gets default
bad_input = {"paid_search": {"mu": 99.0, "sigma": 0.08}}
result = validate_priors(bad_input, channels)
assert result["paid_search"]["mu"] == 0.2
print("✓ Test 3 passed — invalid mu replaced with default")

# Test 4 — negative sigma gets default
bad_sigma = {"paid_search": {"mu": 0.3, "sigma": -0.5}}
result = validate_priors(bad_sigma, channels)
assert result["paid_search"]["sigma"] == 0.15
print("✓ Test 4 passed — invalid sigma replaced with default")
