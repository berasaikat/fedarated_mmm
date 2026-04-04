import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def plot_posterior_evolution(
    round_history: List[Dict[str, Any]],
    channels: List[str],
    output_path: str,
    priors_history: List[Dict[str, Any]] = None,
):
    """
    Plots the structural convergence evolution mapping global parameter means and variances sequentially across rounds.

    Args:
        round_history: List representing chronological rounds (JSONL array from ExperimentLogger.read_rounds())
                       or mapped directly as identical global_summary dict bounds.
        channels: String array describing the distinct channels to visualize natively.
        output_path: Target filesystem path to export the standard Gaussian evolution PNG.
        priors_history: Optional payload mapping localized JSONL LLM inputs (from ExperimentLogger.read_priors()).
                        If structured arrays exist natively, evaluates and prints 'prior_vs_posterior.png'.
    """
    if not round_history or not channels:
        logger.warning(
            "Empty execution bounds passed natively. Cannot execute posterior plot visualizations."
        )
        return

    # Isolate strictly dimensional analytical arrays tracking parameters structurally
    plot_data = {ch: {"mu": [], "sigma": [], "rounds": []} for ch in channels}

    for i, payload in enumerate(round_history):
        # Iteratively extract parameters (whether raw dictionary bounds or NDJSON structurally nested objects)
        summary = (
            payload.get("global_summary", payload)
            if isinstance(payload, dict)
            else payload
        )
        round_idx = (
            payload.get("round_num", i + 1) if isinstance(payload, dict) else i + 1
        )

        for ch in channels:
            ch_data = summary.get(ch, {})
            # Safely capture geometric parameters preventing null execution crashes
            mu = ch_data.get("mean")
            sigma = ch_data.get("std")

            if mu is not None and sigma is not None:
                plot_data[ch]["mu"].append(float(mu))
                plot_data[ch]["sigma"].append(float(sigma))
                plot_data[ch]["rounds"].append(round_idx)

    # 1. Standard Evaluated Bounds Evolution Graph (Yielding Global Mu & explicit ±1 Global Sigma confidence)
    n_cols = min(2, len(channels))
    n_rows = (len(channels) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False
    )
    axes = axes.flatten()

    for i, ch in enumerate(channels):
        ax = axes[i]
        rounds = plot_data[ch]["rounds"]
        mus = plot_data[ch]["mu"]
        sigmas = plot_data[ch]["sigma"]

        if not rounds:
            ax.set_title(f"{ch} (Data Absent)")
            continue

        ax.plot(
            rounds,
            mus,
            marker="o",
            label="Modeled Global Mean",
            color="#2c3e50",
            linewidth=2,
        )

        # Represent exactly the structural 1 Std Dev analytical geometric zone dynamically
        lower_bound = [m - s for m, s in zip(mus, sigmas)]
        upper_bound = [m + s for m, s in zip(mus, sigmas)]
        ax.fill_between(
            rounds,
            lower_bound,
            upper_bound,
            color="#3498db",
            alpha=0.25,
            label="±1 Global Std Dev",
        )

        ax.set_title(f"Global Posterior Convergence: {ch}", fontweight="bold")
        ax.set_xlabel("Federated Iteration Round")
        ax.set_ylabel("Marginal Performance Weight")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

    # Dynamically purge un-utilized subplot artifacts guaranteeing exact dimensional output shapes
    for j in range(len(channels), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(
        f"Successfully generated dynamic Gaussian posterior limits visually mapping to {output_path}"
    )

    # 2. Comparative Generative LLM Context Alignments (Prior Vs Posterior Divergence Maps)
    if priors_history:
        # Cross-sectional aggregation dynamically averaging inputs bridging the multiple structural nodes natively per round
        prior_means = {ch: {} for ch in channels}

        for p_record in priors_history:
            r_num = p_record.get("round_num")
            priors = p_record.get("priors", {})
            if r_num is None:
                continue

            for ch in channels:
                if ch in priors and "mu" in priors[ch]:
                    if r_num not in prior_means[ch]:
                        prior_means[ch][r_num] = []
                    prior_means[ch][r_num].append(float(priors[ch]["mu"]))

        # Collapses nodes accurately down producing arithmetic mean vector natively tracking the global LLM expectation bounds
        prior_avg = {
            ch: {r: sum(vals) / len(vals) for r, vals in prior_means[ch].items()}
            for ch in channels
        }

        fig2, axes2 = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False
        )
        axes2 = axes2.flatten()

        for i, ch in enumerate(channels):
            ax2 = axes2[i]
            rounds = plot_data[ch]["rounds"]
            post_mus = plot_data[ch]["mu"]

            # Translate historical round IDs mathematically binding the average explicit prior scalar
            plot_priors = [prior_avg[ch].get(r, None) for r in rounds]

            # Reconstruct filtered dimension parameters safely blocking None interpolation crashes
            valid_x = [r for r, p in zip(rounds, plot_priors) if p is not None]
            valid_priors = [p for p in plot_priors if p is not None]

            ax2.plot(
                rounds,
                post_mus,
                marker="o",
                label="FedAgg Posterior Mean",
                color="#2c3e50",
                linewidth=2,
            )
            if valid_x:
                ax2.plot(
                    valid_x,
                    valid_priors,
                    marker="x",
                    linestyle="--",
                    label="Derived LLM Target Prior",
                    color="#e74c3c",
                    linewidth=2,
                )

            ax2.set_title(
                f"Generative Prior Alignments vs Empirical Reality: {ch}",
                fontweight="bold",
            )
            ax2.set_xlabel("Federated Sequence Frame")
            ax2.set_ylabel("Parameter Amplitude Shift")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend(loc="best")

        for j in range(i + 1, len(axes2)):
            fig2.delaxes(axes2[j])

        plt.tight_layout()

        # Statically bind outputs directly next to the canonical evolution targets evaluating within the identical environment folder
        prior_out = str(Path(output_path).parent / "prior_vs_posterior.png")

        plt.savefig(prior_out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(
            f"Successfully tracked LLM Prior divergences geometrically saving map vectors sequentially to {prior_out}"
        )
