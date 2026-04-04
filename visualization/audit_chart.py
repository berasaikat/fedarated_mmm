import os
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def plot_audit_results(audit_results_list: List[Dict[str, Any]], output_path: str):
    """
    Plots a continuous horizontal PyPlot forest-plot directly resolving causal validity limits.
    Statically frames identical MMM 90% Confidence Intervals as horizontal bars, mapping exactly
    against Synthetic Experimental Truths (ATT point estimates presented logically as diamonds).

    Mapping scale:
      - Green bounds => Evaluated ATT truth physically lands safely within modeled MMM Uncertainty regions.
      - Red bounds => Spurious structural failure mapping explicitly outside calculated Gaussian limitations.

    Args:
        audit_results_list: The direct array returned by `audit.py` containing identical schema evaluation mappings.
        output_path: Target filesystem PNG image path.
    """
    if not audit_results_list:
        logger.warning(
            "Empty evaluation arrays logically provided—Cannot successfully render structured forest plots."
        )
        return

    num_audits = len(audit_results_list)

    # Establish dynamic bounds resolving computational arrays proportionally scaling globally
    fig, ax = plt.subplots(figsize=(10, max(4, num_audits * 1.5)))

    y_positions = []
    labels = []

    # Iterate exactly backward mapping chronological indexes efficiently from the physical bottom upward
    for i, res in enumerate(reversed(audit_results_list)):
        y = i
        y_positions.append(y)

        # Pull structurally reliable schema properties safely substituting dynamically generated string defaults
        channel = res.get("channel", f"Channel_{num_audits - i}")

        # Extrapolate explicitly normalized scalar components (or safely fall back evaluating unnormalized variables natively)
        att_estimate = res.get(
            "att_estimate_normalized",
            res.get("att_estimate_raw", res.get("att_estimate", 0.0)),
        )
        mmm_beta_mean = res.get("mmm_beta_mean", 0.0)

        ci = res.get("mmm_beta_ci", [mmm_beta_mean - 0.1, mmm_beta_mean + 0.1])
        ci_lower, ci_upper = ci[0], ci[1]

        coverage = res.get("coverage", False)

        # Dynamically evaluate visual indicator paths mapping identically against conditional bounds evaluations
        color = "#2ecc71" if coverage else "#e74c3c"

        # 1. Map Global Posterior Correlation (Federated Output Baseline arrays)
        ax.plot(
            [ci_lower, ci_upper],
            [y, y],
            color=color,
            linewidth=6,
            solid_capstyle="round",
            alpha=0.9,
            label="Federated MMM 90% Error Bound" if i == num_audits - 1 else "",
        )
        ax.plot(
            mmm_beta_mean,
            y,
            marker="o",
            markersize=12,
            color="white",
            markeredgecolor=color,
            markeredgewidth=2.5,
            label="Federated Posterior Mean" if i == num_audits - 1 else "",
        )

        # 2. Embody Identical Explicit Empirical Causation (Experimental ATT Diamond Endpoint)
        ax.plot(
            att_estimate,
            y,
            marker="D",
            markersize=14,
            color="#2c3e50",
            alpha=0.85,
            label="Causal ATT Empirical Validizer" if i == num_audits - 1 else "",
        )

        labels.append(f"{channel}")

        # Optional Relative Quantitative Divergence Logics annotated horizontally past the furthest visual marker
        gap = res.get("gap", 0.0)
        # ax.text(max(ci_upper, att_estimate) + 0.05, y, f"Gap: {gap:+.3f}", va='center', fontsize=9, color=color, fontweight='bold')
        ax.text(
            max(ci_upper, att_estimate) + 0.05,
            y + 0.15,  # slight vertical nudge
            f"Gap: {gap:+.3f}",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
    # Instantiate targeted limits aligning categorical identifiers logically across the Y-Axis natively
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontweight="bold", size=11)

    ax.set_title(
        "Experimental Verification Limits: FedAvg Aggregates vs Synthetic Validations",
        fontweight="bold",
        pad=20,
        fontsize=13,
    )
    ax.set_xlabel("Relative Effectiveness Increment Array", fontsize=11)

    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Establish dynamic endpoints seamlessly capturing horizontal artifacts identically across disparate output ranges
    all_vals = []
    for r in audit_results_list:
        att = r.get(
            "att_estimate_normalized",
            r.get("att_estimate_raw", r.get("att_estimate", 0.0)),
        )
        ci = r.get("mmm_beta_ci", [0.0, 0.0])
        all_vals.extend([att, ci[0], ci[1]])

    x_min, x_max = min(all_vals), max(all_vals)
    padding = (x_max - x_min) * 0.25 + 0.05
    ax.set_xlim(x_min - padding, x_max + padding)

    # Vertical Null-Hypothesis boundary seamlessly injecting explicit mathematical centering
    ax.axvline(x=0, color="grey", linestyle=":", linewidth=2, alpha=0.6)

    # Combine legend filters eliminating structural duplicates generated by looped artifact evaluations
    handles, legends = ax.get_legend_handles_labels()
    by_label = dict(zip(legends, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Analytical Forest Causal Bounds plotted statically returning visually into {output_path}"
    )
