import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm_prior.surprise import compute_surprise, aggregate_surprise

# Simulate prior and posterior with known KL
prior = {
    "paid_search": {"mu": 0.35, "sigma": 0.08},
    "social": {"mu": 0.18, "sigma": 0.10},
}

# Using "mean" and "std" keys — matching posterior.py output from Phase 3
posterior = {
    "paid_search": {"mean": 0.40, "std": 0.06},
    "social": {"mean": 0.18, "std": 0.10},  # identical to prior → KL ≈ 0
}

scores = compute_surprise(prior, posterior)
print(f"\nSurprise scores: {scores}")

# Checks
assert "paid_search" in scores, "paid_search missing from scores"
assert "social" in scores, "social missing from scores"
assert scores["paid_search"] > 0, "KL should be positive when prior != posterior"
assert scores["social"] < 0.01, "KL should be near zero when prior == posterior"
assert all(v >= 0 for v in scores.values()), "KL cannot be negative"

mean_kl = aggregate_surprise(scores)
print(f"Mean KL: {mean_kl:.4f}")
print("Surprise scores look correct")
