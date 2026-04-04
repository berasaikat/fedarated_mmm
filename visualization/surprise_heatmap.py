import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def plot_surprise_heatmap(
    round_history: List[Dict[str, Any]],
    participants: List[str],
    channels: List[str],
    output_path: str,
):
    """
    Constructs a Seaborn correlation heatmap natively representing Bayesian Surprise (KL Divergence).
    Resolves independent graphical components mapping exactly 1-to-1 against isolated participants.

    Mapping scale:
      - Green bounds => Low Surprise (High alignment between LLM and Empirical realities).
      - Red bounds => High Surprise (Severe analytical hallucination/divergence, clipped at > 1.0).

    Args:
        round_history: Chronological evaluation rounds resolving `surprise_scores` dictionaries exactly.
        participants: String IDs representing the explicit target FL nodes securely isolated.
        channels: Media intervention variables defining matrix columns logically.
        output_path: Global filesystem PNG image destination safely housing stacked matrices.
    """
    if not round_history or not participants or not channels:
        logger.warning(
            "Empty evaluation arrays trapped—Cannot visualize surprise heatmaps."
        )
        return

    n_participants = len(participants)

    # Scale figure dynamically matching variable dimensional inputs
    fig, axes = plt.subplots(
        n_participants,
        1,
        figsize=(max(8, len(channels) * 1.5), max(4, 3 * n_participants)),
        squeeze=False,
    )
    axes = axes.flatten()

    # 1. Structure the 2D Data Cubes capturing iterative boundaries securely
    data_cubes = {p: [] for p in participants}
    rounds_labels = []

    for payload in round_history:
        r_num = payload.get("round_num", "Unknown")
        rounds_labels.append(f"Round {r_num}")

        # Explicit dict resolving exactly the local KL divergences observed natively by `round_manager`
        scores_dict = payload.get("surprise_scores", {})

        for p in participants:
            p_scores = scores_dict.get(p, {})
            # Initialize array safely mapping structurally missing channels with implicit zero bounds
            row = [float(p_scores.get(ch, 0.0)) for ch in channels]
            data_cubes[p].append(row)

    # 2. Iterate cleanly over each explicitly separated FL node and embed identical sub-heatmaps
    for i, p in enumerate(participants):
        ax = axes[i]
        p_rounds = []
        p_rows = []

        for payload in round_history:
            r_num = payload.get("round_num", "?")
            scores = payload.get("surprise_scores", {})
            p_scores = scores.get(p, {})
            if p_scores:  # only include rounds where this participant has scores
                p_rounds.append(f"Round {r_num}")
                p_rows.append([float(p_scores.get(ch, 0.0)) for ch in channels])

        if not p_rows:
            ax.set_title(f"{p} — no surprise data")
            continue

        # Matrix orientation natively frames Rounds (Rows) to Independent Channels (Columns)
        df = pd.DataFrame(data_cubes[p], index=rounds_labels, columns=channels)

        # The 'RdYlGn_r' divergent palette perfectly forces High evaluations (1.0+) into structural Red (Bad),
        # while bounding accurate convergence (0.0) solidly into Deep Green.
        sns.heatmap(
            df,
            ax=ax,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "KL Divergence Bounds"},
        )

        ax.set_title(
            f"Bayesian Evaluation Heatmap [KL Surprise]: {p}", fontweight="bold"
        )
        ax.set_ylabel("Temporal Execution Interval")
        ax.set_xlabel("MMM Channel Variable")
        # Add a note to the colorbar
        cbar_kws = {"label": "KL Divergence (capped at 1.0)"}

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Analytical KL severity divergences mapped statically exactly into {output_path}"
    )
