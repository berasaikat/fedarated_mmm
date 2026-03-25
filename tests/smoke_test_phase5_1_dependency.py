import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from privacy.gaussian_mechanism import add_gaussian_noise

# Test 1 — basic noise is added to mean
posterior = {
    "paid_search": {"mean": 0.35, "std": 0.08, "p5": 0.20, "p95": 0.50},
    "social":      {"mean": 0.18, "std": 0.10, "p5": 0.05, "p95": 0.35},
}

noisy = add_gaussian_noise(posterior, sensitivity=1.0, epsilon=1.0, delta=1e-5)

print("--- Noise addition check ---")
for ch in posterior:
    original_mean = posterior[ch]["mean"]
    noisy_mean = noisy[ch]["mean"]
    mean_changed = abs(noisy_mean - original_mean) > 1e-10
    std_unchanged = noisy[ch]["std"] == posterior[ch]["std"]
    print(f"  {ch}: mean changed={mean_changed}, std unchanged={std_unchanged}")
    assert mean_changed, f"{ch} mean was not noised"
    assert std_unchanged, f"{ch} std should not be noised"

# Test 2 — original dict is not mutated
assert posterior["paid_search"]["mean"] == 0.35, "Original dict was mutated!"
print("Original dict not mutated")

# Test 3 — stronger epsilon = less noise
noisy_tight = add_gaussian_noise(posterior, sensitivity=1.0, epsilon=10.0, delta=1e-5)
noisy_loose = add_gaussian_noise(posterior, sensitivity=1.0, epsilon=0.1, delta=1e-5)

# Run 100 samples to compare variance
tight_deltas, loose_deltas = [], []
for _ in range(100):
    t = add_gaussian_noise(posterior, sensitivity=1.0, epsilon=10.0, delta=1e-5)
    l = add_gaussian_noise(posterior, sensitivity=1.0, epsilon=0.1,  delta=1e-5)
    tight_deltas.append(abs(t["paid_search"]["mean"] - posterior["paid_search"]["mean"]))
    loose_deltas.append(abs(l["paid_search"]["mean"] - posterior["paid_search"]["mean"]))

assert np.mean(tight_deltas) < np.mean(loose_deltas), \
    "Higher epsilon should produce less noise"
print(f"Noise scale correct — tight eps avg delta: {np.mean(tight_deltas):.4f}, "
      f"loose eps avg delta: {np.mean(loose_deltas):.4f}")

# Test 4 — invalid epsilon/delta raises
try:
    add_gaussian_noise(posterior, sensitivity=1.0, epsilon=0, delta=1e-5)
    assert False, "Should have raised"
except ValueError:
    print("Zero epsilon raises ValueError")