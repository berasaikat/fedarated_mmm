import numpy as np
import copy
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from aggregator.fed_avg_posterior import fedavg_posterior


# shrinkage=0.0 → pure fedavg (no pooling)
# shrinkage=0.5 → balanced (default, good starting point)
# shrinkage=1.0 → all channels forced to grand mean (over-pooling)
# Tune based on aggregate_surprise scores — high surprise → increase shrinkage
def hierarchical_pool(
    list_of_summaries: List[Dict[str, Dict[str, float]]], shrinkage: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Computes a shrinkage estimator for the federated posterior means.

    For each channel:
      global_mu = (1 - shrinkage) * fedavg_mu + shrinkage * grand_mean
    where grand_mean is the mean of the fedavg_mu across all channels.

    This acts as a hierarchical prior pulling extreme idiosyncratic channel estimates
    towards the global channel average, which is particularly useful
    when some participants have high-uncertainty posteriors or scarce data.

    Args:
        list_of_summaries: A list of dicts (one per participant), each with channel keys mapped to {"mean": float, "std": float}.
        shrinkage: The shrinkage hyperparameter (0.0 to 1.0). 0 means no shrinkage (pure fedavg), 1 means full shrinkage (all channels equal grand_mean).

    Returns:
        A dict representing the new globally aggregated and hierarchically shrunk posterior summary.
    """
    if not list_of_summaries:
        return {}

    # 1. Start with the standard federated average
    # This mathematically gives us the correctly population-averaged fedavg_mu and fedavg_sigma
    fedavg_summary = fedavg_posterior(list_of_summaries)

    if not fedavg_summary:
        return {}

    # 2. Extract all the local channel means to resolve the 'grand_mean' across all modeled channels
    fedavg_mus = [
        params["mean"] for params in fedavg_summary.values() if "mean" in params
    ]

    if not fedavg_mus:
        return fedavg_summary

    grand_mean = float(np.mean(fedavg_mus))

    shrunk_summary = copy.deepcopy(fedavg_summary)

    # 3. Apply the shrinkage formula to each channel's mean
    for ch, params in shrunk_summary.items():
        if "mean" in params:
            fedavg_mu = params["mean"]
            # The shrinkage estimator
            global_mu = (1.0 - shrinkage) * fedavg_mu + shrinkage * grand_mean
            params["mean"] = global_mu

            # Widen sigma to reflect shrinkage uncertainty
            if "std" in params:
                params["std"] = params["std"] * (1.0 + shrinkage * 0.5)

    return shrunk_summary
