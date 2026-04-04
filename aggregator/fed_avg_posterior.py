import numpy as np
import math
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def fedavg_posterior(
    list_of_summaries: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Computes the federated average of multiple participant posterior summaries.
    Uses rigorous mathematical moment-matching for a mixture of Gaussians.

    Formula:
      E[X] = mean of all participant posterior means
      Var(X) = mean of participant posterior variances + variance of participant means

    Args:
        list_of_summaries: A list of dicts (one per participant). Each dict maps
                           channels to {"mean": float, "std": float}.

    Returns:
        A dict representing the new globally aggregated posterior summary bounding the identical schema.
    """
    if not list_of_summaries:
        return {}

    global_summary = {}

    # Dynamically extract all unique channels across all participants
    all_channels = set()
    for summary in list_of_summaries:
        all_channels.update(summary.keys())

    for ch in all_channels:
        means = []
        variances = []

        # Accumulate metrics per-participant
        for summary in list_of_summaries:
            if ch in summary:
                ch_data = summary[ch]
                if "mean" in ch_data and "std" in ch_data:
                    means.append(ch_data["mean"])
                    variances.append(ch_data["std"] ** 2)

        # Skip gracefully if channel is absent across the board
        if not means:
            continue
        if len(means) < len(list_of_summaries):
            logger.warning(
                f"Channel '{ch}' only present in {len(means)}/{len(list_of_summaries)} "
                f"participants — averaging over available only"
            )

        means_arr = np.array(means)
        variances_arr = np.array(variances)

        # 1. global_mu = mean of all participant posterior means
        global_mu = float(np.mean(means_arr))

        # 2. global_sigma = sqrt(mean of variances + variance of means)
        mean_of_variances = float(np.mean(variances_arr))
        # Population variance matches the expectation E[Var(X|Z)] + Var(E[X|Z]) theorem
        variance_of_means = float(np.var(means_arr))

        global_sigma = math.sqrt(mean_of_variances + variance_of_means)

        global_summary[ch] = {"mean": global_mu, "std": global_sigma}

    return global_summary
