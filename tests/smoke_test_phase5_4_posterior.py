import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from privacy.budget_tracker import PrivacyBudgetTracker, PrivacyBudgetExhausted
from privacy.dp_sharing import dp_share_posterior

posterior = {
    "paid_search": {"mean": 0.35, "std": 0.08, "p5": 0.20, "p95": 0.50, "r_hat": 1.01},
    "social":      {"mean": 0.18, "std": 0.10, "p5": 0.05, "p95": 0.35, "r_hat": 1.00},
    "tv":          {"mean": 0.25, "std": 0.12, "p5": 0.08, "p95": 0.42, "r_hat": 1.02},
    "ooh":         {"mean": 0.10, "std": 0.09, "p5": 0.02, "p95": 0.25, "r_hat": 1.01},
}

tracker = PrivacyBudgetTracker(
    total_epsilon=1.0,
    total_delta=1e-4,
    participant_ids=["participant_1"]
)

# Test 1 — successful share
noisy = dp_share_posterior(
    posterior_summary=posterior,
    budget_tracker=tracker,
    participant_id="participant_1",
    epsilon_per_round=0.2,
    delta_per_round=1e-5,
)

print("--- DP Share output ---")
for ch in posterior:
    print(f"  {ch}: original mean={posterior[ch]['mean']:.4f}, "
          f"noisy mean={noisy[ch]['mean']:.4f}")

assert set(noisy.keys()) == set(posterior.keys()), "Output keys don't match input"
assert noisy["paid_search"]["mean"] != posterior["paid_search"]["mean"], \
    "Noise was not applied"
print("dp_share_posterior works end-to-end")

# Test 2 — budget is tracked correctly after share
rem_eps, _ = tracker.remaining("participant_1")
assert abs(rem_eps - 0.8) < 1e-9, f"Expected 0.8 remaining, got {rem_eps}"
print(f"Budget correctly reduced to {rem_eps:.1f}")

# Test 3 — exhausted budget blocks sharing
tracker2 = PrivacyBudgetTracker(
    total_epsilon=0.1,
    total_delta=1e-4,
    participant_ids=["participant_1"]
)
tracker2.spend("participant_1", 0.1, 1e-5)

try:
    dp_share_posterior(posterior, tracker2, "participant_1", 0.2, 1e-5)
    assert False, "Should have raised"
except PrivacyBudgetExhausted:
    print("Exhausted budget correctly blocks sharing")

# Test 4 — original posterior not mutated
assert posterior["paid_search"]["mean"] == 0.35, "Original posterior was mutated"
print("Original posterior not mutated")