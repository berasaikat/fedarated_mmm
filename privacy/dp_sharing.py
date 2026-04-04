from typing import Dict, Any
from privacy.budget_tracker import PrivacyBudgetTracker, PrivacyBudgetExhausted
from privacy.sensitivity import clip_posterior, compute_l2_sensitivity
from privacy.gaussian_mechanism import add_gaussian_noise


def dp_share_posterior(
    posterior_summary: Dict[str, Dict[str, float]],
    budget_tracker: PrivacyBudgetTracker,
    participant_id: int,
    epsilon_per_round: float,
    delta_per_round: float,
) -> Dict[str, Dict[str, float]]:
    """
    Differentially Private wrapper for sharing posterior summaries from a participant to the federated aggregator.
    This encapsulates the entire DP pipeline for a sharing operation.

    Workflow:
    1. Checks if the participant has available privacy budget.
    2. Spends the allocated budget for this round.
    3. Clips the mathematical posterior summary to the bound limit prior to applying DP noise.
    4. Adds continuous Gaussian noise calibrated to the given DP parameters.
    5. Returns the bounded, noisy posterior summary safe for untrusted aggregation.

    Args:
        posterior_summary: The exact posterior summary output from local PyMC/NumPyro models.
        budget_tracker: The privacy budget tracker instance spanning all rounds.
        participant_id: Absolute numerical ID of the participant executing the share.
        epsilon_per_round: The maximum DP epsilon limit designated per aggregation iteration.
        delta_per_round: The continuous DP delta allocation per iteration.

    Returns:
        The DP-noised posterior summary dictionary with exactly the same shape.
    """
    # 1. Ensure the budget mathematically hasn't been completely depleted
    if budget_tracker.is_exhausted(participant_id):
        raise PrivacyBudgetExhausted(
            f"Participant {participant_id}'s privacy budget has been fully depleted prior to sharing."
        )

    # 2. Subtract the given iteration cost (this will also halt/throw Error if this specific action drives it under 0)
    budget_tracker.spend(participant_id, epsilon_per_round, delta_per_round)

    # 3. Clip the beta variables based on L2 formulation mathematically enforcing maximum contribution
    channel_count = len(posterior_summary)
    sensitivity = compute_l2_sensitivity(channel_count)

    try:
        clipped_summary = clip_posterior(posterior_summary, clip_norm=sensitivity)

        # 4. Integrate Gaussian Mechanism using evaluated sensitivities
        noisy_summary = add_gaussian_noise(
            posterior_summary=clipped_summary,
            sensitivity=sensitivity,
            epsilon=epsilon_per_round,
            delta=delta_per_round,
        )
    except Exception as e:
        # Roll back the spend
        budget_tracker.spent_budgets[participant_id]["epsilon"] -= epsilon_per_round
        budget_tracker.spent_budgets[participant_id]["delta"] -= delta_per_round
        raise RuntimeError(f"DP sharing failed after budget deduction: {e}")

    # 5. Pipeline successful! Yield DP representation
    return noisy_summary
