import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from aggregator.convergence import check_convergence, compute_convergence_metrics

# Test 1 — converged when all relative changes below tol
prev = {
    "paid_search": {"mean": 0.350, "std": 0.08},
    "social": {"mean": 0.180, "std": 0.10},
}
curr = {
    "paid_search": {"mean": 0.351, "std": 0.08},  # tiny change
    "social": {"mean": 0.181, "std": 0.10},
}
assert check_convergence(prev, curr, tol=0.05) == True
print("Test 1 — converged when all relative changes below tol")

# Test 2 — not converged when any channel exceeds tol
curr_big = {
    "paid_search": {"mean": 0.40, "std": 0.08},  # large change
    "social": {"mean": 0.181, "std": 0.10},
}
assert check_convergence(prev, curr_big, tol=0.05) == False
print("Test 2 — not converged when any channel exceeds tol")

# Test 3 — empty summaries return False
assert check_convergence({}, curr, tol=0.05) == False
assert check_convergence(prev, {}, tol=0.05) == False
assert check_convergence({}, {}, tol=0.05) == False
print("Test 3 — empty summaries return False")

# Test 4 — new channel in curr returns False
curr_new = {
    "paid_search": {"mean": 0.351, "std": 0.08},
    "social": {"mean": 0.181, "std": 0.10},
    "tv": {"mean": 0.250, "std": 0.09},  # new channel
}
assert check_convergence(prev, curr_new, tol=0.05) == False
print("Test 4 — new channel in curr triggers False")

# Test 5 — compute_convergence_metrics returns correct structure
history = [
    {"paid_search": {"mean": 0.40, "std": 0.08}, "social": {"mean": 0.20, "std": 0.10}},
    {"paid_search": {"mean": 0.35, "std": 0.08}, "social": {"mean": 0.18, "std": 0.10}},
    {"paid_search": {"mean": 0.34, "std": 0.08}, "social": {"mean": 0.18, "std": 0.10}},
]
curves = compute_convergence_metrics(history)

assert "paid_search" in curves
assert "social" in curves
assert len(curves["paid_search"]) == 2  # 3 rounds → 2 transitions
assert len(curves["social"]) == 2
print("Test 5 — convergence curves have correct length (n_rounds - 1)")

# Test 6 — relative change values are correct
# paid_search round 0→1: |0.40 - 0.35| / (0.08 + 1e-6) = 0.625
expected_r1 = abs(0.40 - 0.35) / (0.08 + 1e-6)
assert (
    abs(curves["paid_search"][0] - expected_r1) < 1e-4
), f"Expected {expected_r1:.4f}, got {curves['paid_search'][0]:.4f}"
print(f"Test 6 — relative change correct: {curves['paid_search'][0]:.4f}")

# Test 7 — all values are non-negative floats
for ch, vals in curves.items():
    for v in vals:
        if v is not None:
            assert (
                isinstance(v, float) and v >= 0
            ), f"Expected non-negative float for {ch}, got {v}"
print("Test 7 — all convergence values are non-negative floats")

# Test 8 — missing channel gets None, all lists same length
history_missing = [
    {"paid_search": {"mean": 0.40, "std": 0.08}, "social": {"mean": 0.20, "std": 0.10}},
    {"paid_search": {"mean": 0.35, "std": 0.08}},  # social missing
    {"paid_search": {"mean": 0.34, "std": 0.08}, "social": {"mean": 0.18, "std": 0.10}},
]
curves_missing = compute_convergence_metrics(history_missing)
lengths = [len(v) for v in curves_missing.values()]
assert len(set(lengths)) == 1, f"All channel lists must be same length, got {lengths}"
assert (
    curves_missing["social"][0] is None
), "Missing channel should produce None at that index"
print("Test 8 — missing channel produces None, all lists same length")

# Test 9 — single entry history returns empty
assert compute_convergence_metrics([history[0]]) == {}
assert compute_convergence_metrics([]) == {}
print("Test 9 — single or empty history returns empty dict")

# Test 10 — relative change exactly at tol boundary
prev_boundary = {"paid_search": {"mean": 0.0, "std": 1.0}}
curr_boundary = {"paid_search": {"mean": 0.05, "std": 1.0}}
# relative_change = 0.05 / (1.0 + 1e-6) = 0.04999... which is < 0.05
# so this IS converged
assert check_convergence(prev_boundary, curr_boundary, tol=0.05) == True
print("Test 10a — relative change just below tol is converged")

# Test 10b — relative change strictly above tol is not converged
prev_above = {"paid_search": {"mean": 0.0, "std": 1.0}}
curr_above = {"paid_search": {"mean": 0.06, "std": 1.0}}
# relative_change = 0.06 / (1.0 + 1e-6) = 0.05999... which is >= 0.05
assert check_convergence(prev_above, curr_above, tol=0.05) == False
print("Test 10b — relative change above tol is not converged")

print("\nAll Convergence tests passed")
