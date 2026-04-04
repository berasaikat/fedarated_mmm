import numpy as np
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from privacy.gaussian_mechanism import add_gaussian_noise

# For epsilon=1.0, delta=1e-5, sensitivity=1.0
# Expected noise_std = 1.0 * sqrt(2 * ln(1.25/1e-5)) / 1.0
epsilon, delta, sensitivity = 1.0, 1e-5, 1.0
expected_std = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
print(f"Expected noise std: {expected_std:.4f}")

posterior = {"paid_search": {"mean": 0.0, "std": 0.1}}

# Sample 1000 noisy means — should have std close to expected_std
samples = []
for _ in range(1000):
    noisy = add_gaussian_noise(posterior, sensitivity, epsilon, delta)
    samples.append(noisy["paid_search"]["mean"])

observed_std = np.std(samples)
print(f"Observed noise std: {observed_std:.4f}")
print(f"Ratio (observed/expected): {observed_std/expected_std:.3f}")

# Should be within 10% of expected
assert (
    0.9 < (observed_std / expected_std) < 1.1
), f"Noise std is off — expected ~{expected_std:.3f}, got {observed_std:.3f}"
print("Noise scale is statistically correct")
