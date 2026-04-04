"""
Regression tests for all critical math functions.
Pins exact values so silent formula changes are caught immediately.
"""

import math
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ── 1. KL Divergence (surprise.py) ───────────────────────────────────────────


def test_kl_identical_distributions_is_zero():
    from llm_prior.surprise import compute_surprise

    prior = {"paid_search": {"mu": 0.35, "sigma": 0.08}}
    post = {"paid_search": {"mean": 0.35, "std": 0.08}}
    scores = compute_surprise(prior, post)
    assert (
        abs(scores["paid_search"]) < 1e-6
    ), f"KL of identical distributions should be 0, got {scores['paid_search']}"


def test_kl_known_value():
    from llm_prior.surprise import compute_surprise

    # KL(N(0.4, 0.1) || N(0.2, 0.2))
    # = log(0.2/0.1) + (0.01 + (0.4-0.2)^2) / (2*0.04) - 0.5
    # = log(2) + (0.01 + 0.04) / 0.08 - 0.5
    # = 0.6931 + 0.625 - 0.5 = 0.8181
    prior = {"ch": {"mu": 0.2, "sigma": 0.2}}
    post = {"ch": {"mean": 0.4, "std": 0.1}}
    expected = math.log(0.2 / 0.1) + (0.01 + 0.04) / (2 * 0.04) - 0.5
    scores = compute_surprise(prior, post)
    assert (
        abs(scores["ch"] - expected) < 1e-6
    ), f"Expected KL={expected:.6f}, got {scores['ch']:.6f}"


def test_kl_always_non_negative():
    from llm_prior.surprise import compute_surprise

    rng = np.random.default_rng(0)
    for _ in range(50):
        mu_p, sig_p = rng.uniform(0.1, 1.0), rng.uniform(0.05, 0.5)
        mu_q, sig_q = rng.uniform(0.1, 1.0), rng.uniform(0.05, 0.5)
        prior = {"ch": {"mu": mu_p, "sigma": sig_p}}
        post = {"ch": {"mean": mu_q, "std": sig_q}}
        kl = compute_surprise(prior, post).get("ch", 0.0)
        assert (
            kl >= 0
        ), f"KL negative: {kl:.6f} for prior=({mu_p},{sig_p}) post=({mu_q},{sig_q})"


def test_kl_missing_channel_skipped():
    from llm_prior.surprise import compute_surprise

    prior = {
        "paid_search": {"mu": 0.3, "sigma": 0.1},
        "social": {"mu": 0.2, "sigma": 0.1},
    }
    post = {"paid_search": {"mean": 0.35, "std": 0.08}}
    scores = compute_surprise(prior, post)
    assert "paid_search" in scores
    assert "social" not in scores


def test_aggregate_surprise_mean():
    from llm_prior.surprise import aggregate_surprise

    assert aggregate_surprise({}) == 0.0
    scores = {"a": 0.4, "b": 0.6, "c": 0.2}
    expected = (0.4 + 0.6 + 0.2) / 3
    assert abs(aggregate_surprise(scores) - expected) < 1e-10


# ── 2. Gaussian Mechanism (gaussian_mechanism.py) ────────────────────────────


def test_gaussian_noise_std_formula():
    """Observed noise std must match theoretical: sensitivity * sqrt(2*ln(1.25/delta)) / epsilon"""
    from privacy.gaussian_mechanism import add_gaussian_noise

    epsilon, delta, sensitivity = 1.0, 1e-5, 1.0
    expected_std = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    posterior = {"ch": {"mean": 0.0, "std": 0.1}}
    samples = [
        add_gaussian_noise(posterior, sensitivity, epsilon, delta, seed=i)["ch"]["mean"]
        for i in range(2000)
    ]
    observed_std = float(np.std(samples))
    ratio = observed_std / expected_std
    assert (
        0.92 < ratio < 1.08
    ), f"Noise std ratio {ratio:.3f} outside [0.92, 1.08] — formula may be wrong"


def test_gaussian_noise_higher_epsilon_means_less_noise():
    from privacy.gaussian_mechanism import add_gaussian_noise

    posterior = {"ch": {"mean": 0.0, "std": 0.1}}
    tight_samples = [
        add_gaussian_noise(posterior, 1.0, 10.0, 1e-5, seed=i)["ch"]["mean"]
        for i in range(500)
    ]
    loose_samples = [
        add_gaussian_noise(posterior, 1.0, 0.1, 1e-5, seed=i)["ch"]["mean"]
        for i in range(500)
    ]
    assert np.std(tight_samples) < np.std(
        loose_samples
    ), "Higher epsilon must produce less noise"


def test_gaussian_noise_only_affects_mean_not_std():
    from privacy.gaussian_mechanism import add_gaussian_noise

    posterior = {"ch": {"mean": 0.5, "std": 0.08, "p5": 0.1, "p95": 0.9}}
    noisy = add_gaussian_noise(posterior, 1.0, 1.0, 1e-5, seed=0)
    assert noisy["ch"]["std"] == 0.08, "std must not be noised"
    assert noisy["ch"]["p5"] == 0.1, "p5 must not be noised"
    assert noisy["ch"]["p95"] == 0.9, "p95 must not be noised"


def test_gaussian_noise_does_not_mutate_input():
    from privacy.gaussian_mechanism import add_gaussian_noise

    posterior = {"ch": {"mean": 0.5, "std": 0.08}}
    _ = add_gaussian_noise(posterior, 1.0, 1.0, 1e-5, seed=0)
    assert posterior["ch"]["mean"] == 0.5, "Input posterior was mutated"


# ── 3. FedAvg Posterior (fed_avg_posterior.py) ───────────────────────────────


def test_fedavg_mean_is_arithmetic_mean():
    from aggregator.fed_avg_posterior import fedavg_posterior

    summaries = [
        {"ch": {"mean": 0.2, "std": 0.1}},
        {"ch": {"mean": 0.4, "std": 0.1}},
        {"ch": {"mean": 0.6, "std": 0.1}},
    ]
    result = fedavg_posterior(summaries)
    assert (
        abs(result["ch"]["mean"] - 0.4) < 1e-9
    ), f"Expected mean=0.4, got {result['ch']['mean']}"


def test_fedavg_sigma_law_of_total_variance():
    """
    Var(X) = E[Var(X|Z)] + Var(E[X|Z])
    means = [0.2, 0.6], var of means = 0.04
    variances = [0.01, 0.01], mean of variances = 0.01
    total var = 0.05, std = sqrt(0.05)
    """
    from aggregator.fed_avg_posterior import fedavg_posterior

    summaries = [
        {"ch": {"mean": 0.2, "std": 0.1}},
        {"ch": {"mean": 0.6, "std": 0.1}},
    ]
    expected_std = math.sqrt(0.01 + 0.04)
    result = fedavg_posterior(summaries)
    assert (
        abs(result["ch"]["std"] - expected_std) < 1e-9
    ), f"Expected std={expected_std:.6f}, got {result['ch']['std']:.6f}"


def test_fedavg_single_participant_passthrough():
    from aggregator.fed_avg_posterior import fedavg_posterior

    summaries = [{"ch": {"mean": 0.35, "std": 0.08}}]
    result = fedavg_posterior(summaries)
    assert abs(result["ch"]["mean"] - 0.35) < 1e-9
    assert abs(result["ch"]["std"] - 0.08) < 1e-9


def test_fedavg_empty_returns_empty():
    from aggregator.fed_avg_posterior import fedavg_posterior

    assert fedavg_posterior([]) == {}


# ── 4. Hierarchical Pool (hierarchical.py) ───────────────────────────────────


def test_hierarchical_shrinkage_zero_equals_fedavg():
    from aggregator.fed_avg_posterior import fedavg_posterior
    from aggregator.hierarchical import hierarchical_pool

    summaries = [
        {"a": {"mean": 0.8, "std": 0.1}, "b": {"mean": 0.2, "std": 0.1}},
        {"a": {"mean": 0.4, "std": 0.1}, "b": {"mean": 0.6, "std": 0.1}},
    ]
    fedavg = fedavg_posterior(summaries)
    shrunk = hierarchical_pool(summaries, shrinkage=0.0)
    for ch in fedavg:
        assert (
            abs(shrunk[ch]["mean"] - fedavg[ch]["mean"]) < 1e-9
        ), f"shrinkage=0 diverges from fedavg for {ch}"


def test_hierarchical_shrinkage_one_collapses_to_grand_mean():
    from aggregator.hierarchical import hierarchical_pool

    summaries = [
        {"a": {"mean": 0.8, "std": 0.1}, "b": {"mean": 0.2, "std": 0.1}},
        {"a": {"mean": 0.4, "std": 0.1}, "b": {"mean": 0.6, "std": 0.1}},
    ]
    shrunk = hierarchical_pool(summaries, shrinkage=1.0)
    means = [shrunk[ch]["mean"] for ch in shrunk]
    assert (
        max(means) - min(means) < 1e-9
    ), f"shrinkage=1 should collapse all channels to grand mean, got {means}"


def test_hierarchical_shrinkage_half_formula():
    from aggregator.fed_avg_posterior import fedavg_posterior
    from aggregator.hierarchical import hierarchical_pool

    summaries = [
        {"a": {"mean": 0.8, "std": 0.1}, "b": {"mean": 0.2, "std": 0.1}},
        {"a": {"mean": 0.4, "std": 0.1}, "b": {"mean": 0.6, "std": 0.1}},
    ]
    fedavg = fedavg_posterior(summaries)
    grand_mean = np.mean([fedavg[ch]["mean"] for ch in fedavg])
    shrunk = hierarchical_pool(summaries, shrinkage=0.5)
    for ch in fedavg:
        expected = 0.5 * fedavg[ch]["mean"] + 0.5 * grand_mean
        assert (
            abs(shrunk[ch]["mean"] - expected) < 1e-9
        ), f"{ch}: expected {expected:.4f}, got {shrunk[ch]['mean']:.4f}"


# ── 5. L2 Clipping (sensitivity.py) ──────────────────────────────────────────


def test_clip_posterior_reduces_norm():
    from privacy.sensitivity import clip_posterior

    posterior = {ch: {"mean": 0.8, "std": 0.08} for ch in ["a", "b", "c", "d"]}
    clipped = clip_posterior(posterior, clip_norm=1.0)
    means = [clipped[ch]["mean"] for ch in clipped]
    l2 = math.sqrt(sum(m**2 for m in means))
    assert l2 <= 1.0 + 1e-9, f"L2 norm after clipping = {l2:.6f} > 1.0"


def test_clip_posterior_no_clip_when_within_bound():
    from privacy.sensitivity import clip_posterior

    posterior = {"a": {"mean": 0.1, "std": 0.05}, "b": {"mean": 0.1, "std": 0.05}}
    clipped = clip_posterior(posterior, clip_norm=1.0)
    assert abs(clipped["a"]["mean"] - 0.1) < 1e-9, "Should not clip when within bound"


def test_clip_does_not_affect_std():
    from privacy.sensitivity import clip_posterior

    posterior = {ch: {"mean": 0.9, "std": 0.08} for ch in ["a", "b", "c", "d"]}
    clipped = clip_posterior(posterior, clip_norm=1.0)
    for ch in posterior:
        assert clipped[ch]["std"] == 0.08, f"std should not be clipped for {ch}"


def test_l2_sensitivity_formula():
    from privacy.sensitivity import compute_l2_sensitivity

    for n in [1, 2, 4, 8, 16]:
        expected = math.sqrt(n) * 0.5
        got = compute_l2_sensitivity(n)
        assert abs(got - expected) < 1e-10, f"n={n}: expected {expected}, got {got}"


# ── 6. Convergence (convergence.py) ──────────────────────────────────────────


def test_convergence_formula_exact():
    from aggregator.convergence import check_convergence

    # relative_change = |0.3 - 0.35| / (0.1 + 1e-6) = 0.05 / 0.1000001 = 0.4999...
    # tol = 0.05 → 0.4999 >= 0.05 → not converged
    prev = {"ch": {"mean": 0.30, "std": 0.1}}
    curr = {"ch": {"mean": 0.35, "std": 0.1}}
    assert check_convergence(prev, curr, tol=0.05) == False


def test_convergence_passes_when_tiny_change():
    from aggregator.convergence import check_convergence

    prev = {"ch": {"mean": 0.300, "std": 0.1}}
    curr = {"ch": {"mean": 0.301, "std": 0.1}}
    # relative_change = 0.001 / 0.1 = 0.01 < 0.05 → converged
    assert check_convergence(prev, curr, tol=0.05) == True


def test_convergence_curves_length():
    from aggregator.convergence import compute_convergence_metrics

    history = [
        {"ch": {"mean": 0.4, "std": 0.1}},
        {"ch": {"mean": 0.3, "std": 0.1}},
        {"ch": {"mean": 0.29, "std": 0.1}},
        {"ch": {"mean": 0.288, "std": 0.1}},
    ]
    curves = compute_convergence_metrics(history)
    assert len(curves["ch"]) == 3, f"4 rounds → 3 transitions, got {len(curves['ch'])}"


def test_convergence_curve_values_exact():
    from aggregator.convergence import compute_convergence_metrics

    history = [
        {"ch": {"mean": 0.4, "std": 0.08}},
        {"ch": {"mean": 0.3, "std": 0.08}},
    ]
    curves = compute_convergence_metrics(history)
    expected = abs(0.4 - 0.3) / (0.08 + 1e-6)
    assert (
        abs(curves["ch"][0] - expected) < 1e-6
    ), f"Expected {expected:.6f}, got {curves['ch'][0]:.6f}"


# ── 7. Synthetic Control (synthetic_control.py) ───────────────────────────────


def test_nnls_weights_non_negative():
    from causal_validation.synthetic_control import fit_synthetic_control
    import pandas as pd

    rng = np.random.default_rng(42)
    shared = rng.normal(1000, 20, 30)
    rows = []
    for w in range(30):
        rows.append(
            {
                "week": w + 1,
                "geo_id": "treated",
                "revenue": shared[w] + rng.normal(0, 5),
            }
        )
        for g in ["A", "B", "C"]:
            rows.append(
                {"week": w + 1, "geo_id": g, "revenue": shared[w] + rng.normal(0, 5)}
            )
    df = pd.DataFrame(rows)
    weights = fit_synthetic_control(df, "treated", ["A", "B", "C"])
    assert all(
        v >= 0 for v in weights.values()
    ), f"NNLS weights must be non-negative: {weights}"


def test_att_recovers_known_effect():
    from causal_validation.synthetic_control import (
        fit_synthetic_control,
        estimate_incrementality,
    )
    import pandas as pd

    rng = np.random.default_rng(42)
    n_pre, n_post = 40, 20
    true_att = 150.0
    shared_pre = rng.normal(1000, 15, n_pre)
    shared_post = rng.normal(1000, 15, n_post)
    rows = []
    for i, s in enumerate(shared_pre):
        rows.append(
            {"week": i + 1, "geo_id": "treated", "revenue": s + rng.normal(0, 5)}
        )
        for g in ["A", "B", "C"]:
            rows.append({"week": i + 1, "geo_id": g, "revenue": s + rng.normal(0, 5)})
    for i, s in enumerate(shared_post):
        rows.append(
            {
                "week": n_pre + i + 1,
                "geo_id": "treated",
                "revenue": s + true_att + rng.normal(0, 5),
            }
        )
        for g in ["A", "B", "C"]:
            rows.append(
                {"week": n_pre + i + 1, "geo_id": g, "revenue": s + rng.normal(0, 5)}
            )

    df = pd.DataFrame(rows)
    pre = df[df["week"] <= n_pre]
    post = df[df["week"] > n_pre]
    w = fit_synthetic_control(pre, "treated", ["A", "B", "C"])
    result = estimate_incrementality(pre, post, "treated", w)

    assert (
        abs(result["att"] - true_att) < 30
    ), f"ATT {result['att']:.1f} too far from true {true_att}"
    assert (
        result["p_value"] < 0.01
    ), f"Expected significant p-value, got {result['p_value']:.4f}"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Run with pytest if available, else run manually
    try:
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        # Manual runner
        import inspect

        tests = [
            (n, f)
            for n, f in globals().items()
            if n.startswith("test_") and callable(f)
        ]
        passed = failed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"  ✓ {name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed out of {passed+failed} tests")
        if failed:
            sys.exit(1)
