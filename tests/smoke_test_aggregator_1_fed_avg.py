import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from aggregator.fed_avg_posterior import fedavg_posterior
import math

# Test 1 — basic averaging
summaries = [
    {"paid_search": {"mean": 0.4, "std": 0.08}, "social": {"mean": 0.2, "std": 0.10}},
    {"paid_search": {"mean": 0.6, "std": 0.08}, "social": {"mean": 0.4, "std": 0.10}},
]

result = fedavg_posterior(summaries)
print(f"paid_search mean: {result['paid_search']['mean']}")  # expected: 0.5
print(f"social mean: {result['social']['mean']}")  # expected: 0.3

assert abs(result["paid_search"]["mean"] - 0.5) < 1e-9, "Mean averaging wrong"
assert abs(result["social"]["mean"] - 0.3) < 1e-9, "Mean averaging wrong"
print("Test 1 passed — basic averaging correct")

# Test 2 — sigma uses law of total variance
# Var = mean(variances) + variance(means)
# means = [0.4, 0.6], variance of means = 0.01
# variances = [0.0064, 0.0064], mean of variances = 0.0064
# total var = 0.0064 + 0.01 = 0.0164, std = sqrt(0.0164)
expected_std = math.sqrt(0.0064 + 0.01)
assert (
    abs(result["paid_search"]["std"] - expected_std) < 1e-6
), f"Expected std {expected_std:.4f}, got {result['paid_search']['std']:.4f}"
print(f"Test 2 passed — sigma uses law of total variance correctly")

# Test 3 — empty input
assert fedavg_posterior([]) == {}
print("Test 3 passed — empty input returns empty dict")

# Test 4 — single participant
single = [{"paid_search": {"mean": 0.35, "std": 0.08}}]
result_single = fedavg_posterior(single)
assert abs(result_single["paid_search"]["mean"] - 0.35) < 1e-9
print("Test 4 passed — single participant works")

# Test 5 — missing channel in one participant
partial = [
    {"paid_search": {"mean": 0.4, "std": 0.08}, "social": {"mean": 0.2, "std": 0.10}},
    {"paid_search": {"mean": 0.6, "std": 0.08}},  # no social
]
result_partial = fedavg_posterior(partial)
assert "paid_search" in result_partial
assert "social" in result_partial  # averaged over 1 participant only
print("Test 5 passed — partial channel coverage handled")
