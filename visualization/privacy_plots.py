import os
import logging
import matplotlib.pyplot as plt
from typing import Any

logger = logging.getLogger(__name__)


def plot_budget_consumption(budget_tracker: Any, output_path: str):
    """
    Visualizes the cumulative Differential Privacy Epsilon consumed across all federated participants statically.

    Args:
        budget_tracker: PrivacyBudgetTracker instance isolating current execution consumption states.
        output_path: Target filesystem PNG resolution mapping final graphic.
    """
    total_epsilon_limit = getattr(budget_tracker, "total_epsilon", 10.0)
    spent_budgets = getattr(budget_tracker, "spent_budgets", {})

    if not spent_budgets:
        logger.warning(
            "No evaluated budgets found natively to plot in the targeted Privacy Tracker."
        )
        return

    participants = list(spent_budgets.keys())
    epsilons_spent = [spent_budgets[p].get("epsilon", 0.0) for p in participants]

    plt.figure(figsize=(10, 6))

    # Dynamically inject aesthetic palette cycling mappings for structural variance
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#34495e"]
    dynamic_colors = [colors[i % len(colors)] for i in range(len(participants))]

    # Plotting absolute cumulative totals per participant natively
    bars = plt.bar(
        participants, epsilons_spent, color=dynamic_colors, edgecolor="black", alpha=0.9
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)

    # Bounding exact analytical limit mathematically
    plt.axhline(
        y=total_epsilon_limit,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Total Permitted Limit ($\\epsilon$={total_epsilon_limit})",
    )

    plt.title("Differential Privacy Cumulative Epsilon Consumption", fontweight="bold")
    plt.xlabel("Federated Participant Node")
    plt.ylabel("Epsilon ($\\epsilon$) Spent")

    # Headroom mapping resolving chart dimensions intelligently against the maximum bounds limits
    max_height = max([total_epsilon_limit] + epsilons_spent)
    plt.ylim(0, max_height * 1.15)

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    # Annotate analytical floats squarely on top of graphical limits
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (max_height * 0.015),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            size=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Successfully generated visual spatial privacy limits securely resolving locally to {output_path}"
    )
