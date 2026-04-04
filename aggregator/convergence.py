import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def check_convergence(
    prev_global_summary: Dict[str, Dict[str, float]],
    curr_global_summary: Dict[str, Dict[str, float]],
    tol: float = 0.05,
) -> bool:
    """
    Determines mathematically if the global federated aggregation model has stabilized.

    Formula evaluated identically across all channels:
      Relative Change = |prev_mu - curr_mu| / (prev_sigma + 1e-6)

    If every available channel resolves a relative change dropping STRICTLY under the
    specified tolerance (`tol`), the loop formally achieves convergence.

    Args:
        prev_global_summary: Global posterior bounds array structurally generated in the previous round.
        curr_global_summary: Current round aggregate bounding conditions.
        tol: The tolerance threshold triggering success evaluations.

    Returns:
        bool: True if strictly all evaluated relative gradients < tol.
    """
    if not prev_global_summary or not curr_global_summary:
        return False

    for ch, curr_params in curr_global_summary.items():
        if ch not in prev_global_summary:
            logger.info(f"New channel '{ch}' detected in round — resetting convergence")
            # Introduction of completely novel variables interrupts immediate convergence mathematical assumptions
            return False

        prev_params = prev_global_summary[ch]

        curr_mu = curr_params.get("mean", 0.0)
        prev_mu = prev_params.get("mean", 0.0)
        prev_sigma = prev_params.get("std", 0.15)

        # Calculate geometric distance bridging chronological steps against prior distributions
        relative_change = abs(prev_mu - curr_mu) / (prev_sigma + 1e-6)

        # Halt evaluation immediately if a single dimension violates gradient constraints
        if relative_change >= tol:
            return False

    return True


def compute_convergence_metrics(
    history: List[Dict[str, Dict[str, float]]],
) -> Dict[str, List[float]]:
    """
    Extracts the analytical descent progression mapping chronological round-over-round relative adjustments.

    Args:
        history: Sequential List traversing `global_summary` records temporally.

    Returns:
        Dict: Mapping specific variable `channel` to an ordered list calculating iteration gradients sequentially.
    """
    if not history or len(history) < 2:
        return {}

    convergence_curves = {}

    # Establish all unique tracking vectors mathematically observed
    all_channels = set()
    for summary in history:
        all_channels.update(summary.keys())

    for ch in all_channels:
        convergence_curves[ch] = []

    for i in range(1, len(history)):
        prev_summary = history[i - 1]
        curr_summary = history[i]

        for ch in all_channels:
            if ch in prev_summary and ch in curr_summary:
                prev_mu = prev_summary[ch].get("mean", 0.0)
                prev_sigma = prev_summary[ch].get("std", 0.15)
                curr_mu = curr_summary[ch].get("mean", 0.0)

                # Execute identical normalized bounds tracking
                relative_change = abs(prev_mu - curr_mu) / (prev_sigma + 1e-6)
                convergence_curves[ch].append(float(relative_change))
            else:
                # If target parameters were absent dynamically bridging rounds, push explicit `None`s structurally
                convergence_curves[ch].append(None)

    return convergence_curves
