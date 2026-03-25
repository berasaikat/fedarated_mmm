import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from privacy.sensitivity import compute_l2_sensitivity, clip_posterior

# Test 1 — sensitivity scales with channel count
for n in [1, 2, 4, 8]:
    s = compute_l2_sensitivity(n)
    expected = math.sqrt(n) * 0.5
    assert abs(s - expected) < 1e-10, f"Wrong sensitivity for {n} channels"
print("✓ Sensitivity formula correct")

# Test 2 — clipping reduces norm when exceeded
posterior = {
    "paid_search": {"mean": 0.8, "std": 0.08},
    "social":      {"mean": 0.8, "std": 0.10},
    "tv":          {"mean": 0.8, "std": 0.12},
    "ooh":         {"mean": 0.8, "std": 0.09},
}

# L2 norm of [0.8, 0.8, 0.8, 0.8] = 1.6 — exceeds clip_norm=1.0
clipped = clip_posterior(posterior, clip_norm=1.0)

means = [clipped[ch]["mean"] for ch in posterior]
l2_norm = math.sqrt(sum(m**2 for m in means))
print(f"  L2 norm after clipping: {l2_norm:.6f} (target: 1.0)")
assert l2_norm <= 1.0 + 1e-9, f"Clipping failed — norm is {l2_norm}"
print("✓ Clipping correctly limits L2 norm")

# Test 3 — clipping does not affect std
for ch in posterior:
    assert clipped[ch]["std"] == posterior[ch]["std"], \
        f"std should not be clipped for {ch}"
print("Clipping only affects means, not std")

# Test 4 — no clipping when norm is already within bounds
small_posterior = {
    "paid_search": {"mean": 0.1, "std": 0.05},
    "social":      {"mean": 0.1, "std": 0.05},
}
clipped_small = clip_posterior(small_posterior, clip_norm=1.0)
assert clipped_small["paid_search"]["mean"] == 0.1, "Should not clip small vectors"
print("No clipping applied when norm already within bounds")

# Test 5 — original not mutated
assert posterior["paid_search"]["mean"] == 0.8, "Original mutated by clip_posterior"
print("Original dict not mutated by clipping")