# tests/smoke_test_phase10_visualization.py
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

output_dir = tempfile.mkdtemp()

try:
    channels = ["paid_search", "social", "tv", "ooh"]
    participants = ["participant_1", "participant_2", "participant_3"]

    # ── Shared fake data ──────────────────────────────────────────────
    round_history = [
        {
            "round_num": r,
            "global_summary": {
                ch: {"mean": 0.2 + r * 0.02 + i * 0.05, "std": 0.08}
                for i, ch in enumerate(channels)
            },
            "surprise_scores": {
                p: {ch: 0.1 * r + 0.05 * j for j, ch in enumerate(channels)}
                for p in participants
            },
            "epsilon_spent_per_participant": 0.5,
            "epsilon_spent_total": 1.5,
        }
        for r in range(1, 6)
    ]

    priors_history = [
        {
            "round_num": r,
            "participant_id": p,
            "priors": {ch: {"mu": 0.2, "sigma": 0.1} for ch in channels},
        }
        for r in range(1, 6)
        for p in participants
    ]

    audit_results = [
        {
            "channel": ch,
            "mmm_beta_mean": 0.30,
            "mmm_beta_ci": [0.20, 0.40],
            "att_estimate_normalized": 0.28 + i * 0.05,
            "att_estimate_raw": 280.0,
            "coverage": i % 2 == 0,
            "gap": -0.02 + i * 0.01,
            "att_p_value": 0.001,
        }
        for i, ch in enumerate(channels)
    ]

    # Mock budget tracker
    mock_tracker = MagicMock()
    mock_tracker.total_epsilon = 2.0
    mock_tracker.spent_budgets = {
        p: {"epsilon": 0.5 * (i + 1), "delta": 1e-5} for i, p in enumerate(participants)
    }

    # ── 1. Posterior evolution ────────────────────────────────────────
    from visualization.posterior_plots import plot_posterior_evolution

    out1 = str(Path(output_dir) / "posterior_evolution.png")
    plot_posterior_evolution(
        round_history, channels, out1, priors_history=priors_history
    )

    assert Path(out1).exists(), "posterior_evolution.png not created"
    assert (
        Path(out1).stat().st_size > 1000
    ), "posterior_evolution.png suspiciously small"

    prior_vs_post = Path(output_dir) / "prior_vs_posterior.png"
    assert prior_vs_post.exists(), "prior_vs_posterior.png not created"
    print("posterior_plots — both PNGs created")

    # ── 2. Privacy budget ─────────────────────────────────────────────
    from visualization.privacy_plots import plot_budget_consumption

    out2 = str(Path(output_dir) / "privacy_budget.png")
    plot_budget_consumption(mock_tracker, out2)

    assert Path(out2).exists(), "privacy_budget.png not created"
    assert Path(out2).stat().st_size > 1000, "privacy_budget.png suspiciously small"
    print("privacy_plots — PNG created")

    # ── 3. Surprise heatmap ───────────────────────────────────────────
    from visualization.surprise_heatmap import plot_surprise_heatmap

    out3 = str(Path(output_dir) / "surprise_heatmap.png")
    plot_surprise_heatmap(round_history, participants, channels, out3)

    assert Path(out3).exists(), "surprise_heatmap.png not created"
    assert Path(out3).stat().st_size > 1000, "surprise_heatmap.png suspiciously small"
    print("surprise_heatmap — PNG created")

    # ── 4. Audit chart ────────────────────────────────────────────────
    from visualization.audit_chart import plot_audit_results

    out4 = str(Path(output_dir) / "audit_chart.png")
    plot_audit_results(audit_results, out4)

    assert Path(out4).exists(), "audit_chart.png not created"
    assert Path(out4).stat().st_size > 1000, "audit_chart.png suspiciously small"
    print("audit_chart — PNG created")

    # ── 5. Edge cases — empty inputs should not crash ─────────────────
    plot_posterior_evolution([], channels, str(Path(output_dir) / "empty.png"))
    plot_budget_consumption(
        MagicMock(total_epsilon=1.0, spent_budgets={}),
        str(Path(output_dir) / "empty_budget.png"),
    )
    plot_surprise_heatmap(
        [], participants, channels, str(Path(output_dir) / "empty_heatmap.png")
    )
    plot_audit_results([], str(Path(output_dir) / "empty_audit.png"))
    print("Edge cases — empty inputs handled without crash")

    print(f"\nAll visualization smoke tests passed")
    print(f"  Inspect outputs in: {output_dir}")

finally:
    shutil.rmtree(output_dir)
