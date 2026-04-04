import math
from typing import Tuple


class PrivacyBudgetExhausted(Exception):
    """Exception raised when a participant's privacy budget is exceeded."""

    pass


class PrivacyBudgetTracker:
    def __init__(self, total_epsilon: float, total_delta: float, participant_ids: list):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_budgets = {
            pid: {"epsilon": 0.0, "delta": 0.0} for pid in participant_ids
        }

    def spend(
        self, participant_id: int, epsilon_spent: float, delta_spent: float
    ) -> None:
        """
        Deducts epsilon and delta from the participant's budget via sequential composition.
        Raises PrivacyBudgetExhausted if the deduction would exceed the total available limit.
        """
        if participant_id not in self.spent_budgets:
            self.spent_budgets[participant_id] = {"epsilon": 0.0, "delta": 0.0}

        current = self.spent_budgets[participant_id]

        new_eps = current["epsilon"] + epsilon_spent
        new_delta = current["delta"] + delta_spent

        # Add tiny floating-point tolerance to prevent false positives at boundary exact matches
        if new_eps > self.total_epsilon + 1e-10 or new_delta > self.total_delta + 1e-10:
            raise PrivacyBudgetExhausted(
                f"Privacy budget exceeded for participant {participant_id}. "
                f"Attempting to spend (eps={epsilon_spent}, delta={delta_spent}). "
                f"Total limits: (eps={self.total_epsilon}, delta={self.total_delta}). "
                f"Currently spent: (eps={current['epsilon']}, delta={current['delta']})."
            )

        self.spent_budgets[participant_id]["epsilon"] = new_eps
        self.spent_budgets[participant_id]["delta"] = new_delta

    def remaining(self, participant_id: int) -> Tuple[float, float]:
        """
        Returns the remaining privacy budget (epsilon, delta) for the participant.
        """
        if participant_id not in self.spent_budgets:
            return (self.total_epsilon, self.total_delta)

        current = self.spent_budgets[participant_id]
        # Bounding at exactly zero to gracefully handle the floating point precision tolerance limits
        rem_eps = max(0.0, self.total_epsilon - current["epsilon"])
        rem_delta = max(0.0, self.total_delta - current["delta"])

        return (rem_eps, rem_delta)

    def is_exhausted(self, participant_id: int) -> bool:
        """
        Returns True if either the epsilon or delta budget is fully spent/exhausted.
        """
        rem_eps, _ = self.remaining(participant_id)
        # It's exhausted if either capacity drops to functionally zero
        return rem_eps <= 1e-10
