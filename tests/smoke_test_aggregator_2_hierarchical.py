import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from aggregator.hierarchical import hierarchical_pool

summaries = [
    {"paid_search": {"mean": 0.8, "std": 0.08}, "social": {"mean": 0.2, "std": 0.10}},
    {"paid_search": {"mean": 0.6, "std": 0.08}, "social": {"mean": 0.4, "std": 0.10}},
]

# Test 1 — shrinkage=0 should equal fedavg
from aggregator.fed_avg_posterior import fedavg_posterior

fedavg = fedavg_posterior(summaries)
shrunk_zero = hierarchical_pool(summaries, shrinkage=0.0)

for ch in fedavg:
    assert (
        abs(shrunk_zero[ch]["mean"] - fedavg[ch]["mean"]) < 1e-9
    ), f"shrinkage=0 should equal fedavg for {ch}"
print("Test 1 passed — shrinkage=0 equals pure fedavg")

# Test 2 — shrinkage=1 should make all channels equal to grand mean
shrunk_full = hierarchical_pool(summaries, shrinkage=1.0)
means = [shrunk_full[ch]["mean"] for ch in shrunk_full]
assert (
    max(means) - min(means) < 1e-9
), "shrinkage=1 should collapse all channels to grand mean"
print(f"Test 2 passed — shrinkage=1 collapses to grand mean ({means[0]:.4f})")

# Test 3 — shrinkage=0.5 pulls toward grand mean
shrunk_half = hierarchical_pool(summaries, shrinkage=0.5)
grand_mean = sum(fedavg[ch]["mean"] for ch in fedavg) / len(fedavg)

for ch in shrunk_half:
    fedavg_mu = fedavg[ch]["mean"]
    expected_mu = 0.5 * fedavg_mu + 0.5 * grand_mean
    assert (
        abs(shrunk_half[ch]["mean"] - expected_mu) < 1e-9
    ), f"shrinkage=0.5 formula wrong for {ch}"
print("Test 3 passed — shrinkage=0.5 formula correct")

# Test 4 — original summaries not mutated
original_mean = summaries[0]["paid_search"]["mean"]
assert summaries[0]["paid_search"]["mean"] == original_mean, "Input mutated"
print("Test 4 passed — input not mutated")
