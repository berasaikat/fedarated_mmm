"""
Full pipeline integration test — runs 2 rounds with 3 participants.
Mocks LLM API and uses reduced MCMC settings.
Asserts all components wire together correctly end-to-end.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

# ── Helpers ──────────────────────────────────────────────────────────────────

CHANNELS = ["paid_search", "social", "tv", "ooh"]
PARTICIPANTS = ["participant_1", "participant_2", "participant_3"]
N_WEEKS = 52


def make_participant_csv(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    shared = rng.normal(1000, 30, size=N_WEEKS)
    data = {"week": list(range(1, N_WEEKS + 1))}
    data["revenue"] = [max(0, shared[i] + rng.normal(0, 15)) for i in range(N_WEEKS)]
    for ch in CHANNELS:
        spend_mean = rng.uniform(100, 400)
        data[ch] = list(
            np.maximum(0, rng.normal(spend_mean, spend_mean * 0.2, N_WEEKS))
        )
    pd.DataFrame(data).to_csv(path, index=False)


def make_participant_yaml(path: Path, pid: str, data_path: str):
    cfg = {
        "participant_id": pid,
        "industry_vertical": "retail",
        "region": "US-Northeast",
        "seasonality_pattern": "retail",
        "budget_share": {ch: 0.25 for ch in CHANNELS},
        "channel_descriptions": {ch: f"{ch} description" for ch in CHANNELS},
        "data_path": data_path,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def make_global_yaml(path: Path, num_participants: int, data_dir: Path):
    cfg = {
        "num_participants": num_participants,
        "num_rounds": 2,
        "channels": CHANNELS,
        "seed": 42,
        "total_epsilon": 10.0,
        "total_delta": 1e-3,
        "epsilon_per_round": 0.5,
        "delta_per_round": 1e-5,
        "convergence_tolerance": 0.05,
        "llm_model": "claude-sonnet-4-5",
        "aggregation": {"method": "fedavg_posterior"},
        "privacy": {"epsilon": 10.0, "delta": 1e-3},
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def mock_llm_elicit(participant_config, channels, posterior_history=None):
    return {
        "priors": {
            ch: {"mu": 0.2, "sigma": 0.15, "reasoning": "mocked"}
            for ch in (channels.keys() if isinstance(channels, dict) else channels)
        },
        "confidence": "medium",
        "notes": "mocked elicitation",
    }


# ── Test ─────────────────────────────────────────────────────────────────────


def test_full_pipeline():
    tmp = tempfile.mkdtemp()
    try:
        root = Path(tmp)
        data_dir = root / "data" / "synthetic"
        config_dir = root / "config"
        results_dir = root / "results"
        logs_dir = root / "logs" / "exp_001"
        data_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)

        # ── 1. Write synthetic CSVs and participant configs ───────────────
        for i, pid in enumerate(PARTICIPANTS, start=1):
            csv_path = data_dir / f"{pid}.csv"
            make_participant_csv(csv_path, seed=42 + i)
            make_participant_yaml(
                config_dir / f"participant_{i}.yaml",
                pid,
                str(csv_path),
            )

        global_cfg_path = config_dir / "global.yaml"
        make_global_yaml(global_cfg_path, len(PARTICIPANTS), data_dir)

        print("✓ Step 1 — synthetic data and configs written")

        # ── 2. Local trainer smoke (reduced MCMC) ─────────────────────────
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        from participants.local_trainer import LocalTrainer

        trainer = LocalTrainer(
            participant_id="participant_1",
            config_path=str(config_dir / "participant_1.yaml"),
        )
        spend_matrix, revenue, spend_cols = trainer.load_data()
        assert spend_matrix.shape == (
            N_WEEKS,
            len(CHANNELS),
        ), f"Unexpected spend shape: {spend_matrix.shape}"
        assert revenue.shape == (N_WEEKS,)
        assert set(spend_cols) == set(CHANNELS)
        print("✓ Step 2 — LocalTrainer.load_data() correct shape")

        # ── 3. Run MCMC on one participant (reduced settings) ─────────────
        priors = {ch: {"mu": 0.2, "sigma": 0.5} for ch in CHANNELS}
        import jax.numpy as jnp
        from participants.inference import run_mcmc
        from participants.mmm_model import mmm_numpyro

        result = run_mcmc(
            model=mmm_numpyro,
            spend_matrix=jnp.array(spend_matrix),
            revenue=jnp.array(revenue),
            priors_dict=priors,
            channel_names=spend_cols,
            num_warmup=100,
            num_samples=200,
            num_chains=1,
            seed=42,
        )
        samples = result
        expected_keys = [f"beta_{i}" for i in range(len(CHANNELS))]
        assert all(
            k in samples for k in expected_keys
        ), f"Missing beta samples. Got: {list(samples.keys())}"
        print("✓ Step 3 — MCMC ran successfully for participant_1")

        # ── 4. Posterior extraction ───────────────────────────────────────
        from participants.posterior import extract_posterior_summary

        summary = extract_posterior_summary(samples)
        assert len(summary) == len(
            CHANNELS
        ), f"Expected {len(CHANNELS)} betas, got {len(summary)}"
        for param, stats in summary.items():
            assert np.isfinite(stats["mean"]), f"{param} mean is not finite"
            assert np.isfinite(stats["std"]), f"{param} std is not finite"
            assert stats["r_hat"] > 0, f"{param} r_hat is zero"
            if stats["r_hat"] > 1.1:
                print(f"  ⚠ WARNING: {param} r_hat={stats['r_hat']:.3f}")
        print("✓ Step 4 — posterior summary extracted correctly")

        # ── 5. LLM prior elicitation (mocked) ────────────────────────────
        with patch(
            "llm_prior.elicitor.PriorElicitor.elicit", side_effect=mock_llm_elicit
        ):
            from llm_prior.elicitor import PriorElicitor
            from llm_prior.validator import validate_priors

            elicitor = PriorElicitor()
            with open(config_dir / "participant_1.yaml") as f:
                p_cfg = yaml.safe_load(f)

            raw = elicitor.elicit(p_cfg, p_cfg["channel_descriptions"])
            validated = validate_priors(raw["priors"], CHANNELS)

            assert set(validated.keys()) == set(CHANNELS)
            for ch in CHANNELS:
                assert 0 < validated[ch]["mu"] < 5
                assert 0.01 < validated[ch]["sigma"] < 2.0
        print("✓ Step 5 — LLM prior elicitation (mocked) and validation correct")

        # ── 6. Surprise scores ────────────────────────────────────────────
        from llm_prior.surprise import compute_surprise, aggregate_surprise

        # Remap summary keys from beta_X to channel names
        posterior_by_channel = {
            ch: {
                "mean": summary[f"beta_{i}"]["mean"],
                "std": summary[f"beta_{i}"]["std"],
            }
            for i, ch in enumerate(CHANNELS)
        }
        surprise = compute_surprise(validated, posterior_by_channel)
        assert len(surprise) == len(CHANNELS)
        assert all(
            v >= 0 for v in surprise.values()
        ), "KL divergence must be non-negative"
        mean_kl = aggregate_surprise(surprise)
        assert mean_kl >= 0
        print(f"✓ Step 6 — surprise scores computed (mean KL={mean_kl:.4f})")

        # ── 7. DP sharing ─────────────────────────────────────────────────
        from privacy.budget_tracker import PrivacyBudgetTracker
        from privacy.dp_sharing import dp_share_posterior

        tracker = PrivacyBudgetTracker(
            total_epsilon=10.0,
            total_delta=1e-3,
            participant_ids=PARTICIPANTS,
        )
        noisy = dp_share_posterior(
            posterior_summary=posterior_by_channel,
            budget_tracker=tracker,
            participant_id="participant_1",
            epsilon_per_round=0.5,
            delta_per_round=1e-5,
        )
        assert set(noisy.keys()) == set(CHANNELS)
        for ch in CHANNELS:
            assert (
                noisy[ch]["mean"] != posterior_by_channel[ch]["mean"]
            ), f"DP noise not applied to {ch}"
        rem_eps, _ = tracker.remaining("participant_1")
        assert abs(rem_eps - 9.5) < 1e-9, f"Expected 9.5 remaining, got {rem_eps}"
        print("✓ Step 7 — DP sharing applied, budget correctly decremented")

        # ── 8. Federated aggregation ──────────────────────────────────────
        from aggregator.fed_avg_posterior import fedavg_posterior
        from aggregator.hierarchical import hierarchical_pool

        fake_summaries = [
            {
                ch: {"mean": 0.2 + j * 0.05 + i * 0.02, "std": 0.08}
                for i, ch in enumerate(CHANNELS)
            }
            for j in range(len(PARTICIPANTS))
        ]
        global_avg = fedavg_posterior(fake_summaries)
        global_hier = hierarchical_pool(fake_summaries, shrinkage=0.5)

        assert set(global_avg.keys()) == set(CHANNELS)
        assert set(global_hier.keys()) == set(CHANNELS)
        for ch in CHANNELS:
            assert np.isfinite(global_avg[ch]["mean"])
            assert np.isfinite(global_hier[ch]["mean"])
        print("✓ Step 8 — fedavg and hierarchical pooling both produce valid summaries")

        # ── 9. Experiment logging ─────────────────────────────────────────
        from config.experiment_logger import ExperimentLogger
        from aggregator.convergence import (
            check_convergence,
            compute_convergence_metrics,
        )

        exp_logger = ExperimentLogger("exp_001", str(root / "logs"))
        exp_logger.log_round(
            round_num=1,
            global_summary=global_avg,
            surprise_scores={"participant_1": surprise},
            epsilon_spent_per_participant=0.5,
            num_active_participants=3,
        )
        exp_logger.log_priors(1, "participant_1", validated)
        exp_logger.log_audit(
            {
                "channel": "paid_search",
                "mmm_beta_mean": 0.3,
                "mmm_beta_ci": [0.2, 0.4],
                "att_estimate_normalized": 0.28,
                "coverage": True,
                "gap": -0.02,
            }
        )
        exp_logger.log_round(
            round_num=2,
            global_summary=global_hier,
            surprise_scores={"participant_1": surprise},
            epsilon_spent_per_participant=0.5,
            num_active_participants=3,
        )
        exp_logger.save_summary()

        assert exp_logger.rounds_log_path.exists()
        assert exp_logger.priors_log_path.exists()
        assert exp_logger.audits_log_path.exists()
        assert (Path(exp_logger.base_dir) / "experiment_summary.json").exists()

        rounds_back = exp_logger.read_rounds()
        assert len(rounds_back) == 2
        assert rounds_back[0]["round_num"] == 1
        assert rounds_back[1]["round_num"] == 2
        assert (
            abs(exp_logger.summary["metrics"]["epsilon_spent_cumulated"] - 3.0) < 1e-9
        )

        history = [r["global_summary"] for r in rounds_back]
        converged = check_convergence(history[0], history[1], tol=0.05)
        curves = compute_convergence_metrics(history)
        assert set(curves.keys()) == set(CHANNELS)
        assert all(len(v) == 1 for v in curves.values())
        print("✓ Step 9 — experiment logging, read-back, and convergence check correct")

        # ── 10. Visualization smoke ───────────────────────────────────────
        plots_dir = root / "plots"
        plots_dir.mkdir(exist_ok=True)

        from visualization.posterior_plots import plot_posterior_evolution
        from visualization.privacy_plots import plot_budget_consumption
        from visualization.surprise_heatmap import plot_surprise_heatmap
        from visualization.audit_chart import plot_audit_results

        plot_posterior_evolution(
            rounds_back,
            CHANNELS,
            str(plots_dir / "posterior_evolution.png"),
        )
        plot_budget_consumption(tracker, str(plots_dir / "privacy_budget.png"))
        plot_surprise_heatmap(
            rounds_back,
            ["participant_1"],
            CHANNELS,
            str(plots_dir / "surprise_heatmap.png"),
        )
        plot_audit_results(
            exp_logger.read_audits(),
            str(plots_dir / "audit_chart.png"),
        )
        for png in [
            "posterior_evolution.png",
            "privacy_budget.png",
            "surprise_heatmap.png",
            "audit_chart.png",
        ]:
            assert (plots_dir / png).exists(), f"Missing plot: {png}"
        print("✓ Step 10 — all visualization plots generated without error")

        print("\n" + "=" * 60)
        print("✓ FULL PIPELINE INTEGRATION TEST PASSED (10/10 steps)")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_full_pipeline()
