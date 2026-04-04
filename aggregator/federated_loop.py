import yaml, json
import logging
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm_prior.elicitor import PriorElicitor
from privacy.budget_tracker import PrivacyBudgetTracker
from aggregator.round_manager import RoundManager

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Attempt dynamic load for the actual client-side worker class
try:
    from participants.trainer import LocalTrainer
except ImportError:
    # A lightweight mock definition if it hasn't been implemented in this environment yet
    class LocalTrainer:
        def __init__(self, participant_id, participant_config, channels=None):
            self.participant_id = participant_id
            self.participant_config = participant_config
            self.channels = channels or {}
            self.posterior_history = []

        def train(self, valid_priors):
            return {}


logger = logging.getLogger(__name__)


def run_federated_training(config_path: str) -> List[Dict[str, Any]]:
    """
    Executes the global structured federated training loop iteratively.

    Args:
        config_path: String path to the JSON configuration specifying simulation limits.

    Returns:
        List of structured round summaries:
        [
          {
            "round_num": int,
            "global_summary": dict,
            "per_participant_surprise": dict,
            "epsilon_spent": float
          }, ...
        ]
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    num_rounds = config.get("num_rounds", 5)
    total_epsilon = config.get("total_epsilon", 10.0)
    total_delta = config.get("total_delta", 1e-4)
    eps_per_round = config.get("epsilon_per_round", 1.0)

    # Dynamically structure participants ensuring backwards compatibility with indexing
    participant_configs = config.get("participants", [])
    participant_ids = [p.get("id", str(i)) for i, p in enumerate(participant_configs)]

    # 1. Instantiate the Global Federated Components
    prior_elicitor = PriorElicitor(
        model_name=config.get("llm_model", "claude-3-5-sonnet-latest")
    )

    budget_tracker = PrivacyBudgetTracker(
        total_epsilon=total_epsilon,
        total_delta=total_delta,
        participant_ids=participant_ids,
    )

    round_manager = RoundManager(config=config, budget_tracker=budget_tracker)

    # 2. Instantiate all local, mathematically isolated trainer processes
    all_local_trainers = []
    for i, p_config in enumerate(participant_configs):
        pid = participant_ids[i]
        channels = p_config.get("channels", {})

        trainer = LocalTrainer(
            participant_id=pid, participant_config=p_config, channels=channels
        )
        all_local_trainers.append(trainer)

    results = []
    previous_global_summary = None

    for round_num in range(1, num_rounds + 1):
        logger.info(f"--- Starting Federated Loop Round {round_num} ---")

        # Verify valid budget capacity; RoundManager will fatally crash if we pass exhausted nodes
        active_trainers = [
            t
            for t in all_local_trainers
            if not budget_tracker.is_exhausted(t.participant_id)
        ]

        if not active_trainers:
            logger.info(
                "All participants' global DP privacy budgets are exhausted! Terminating federated loop."
            )
            break

        # 3. Trigger Iterative Multi-step RoundManager
        global_summary, surprise_scores = round_manager.run_round(
            round_num=round_num,
            all_local_trainers=active_trainers,
            prior_elicitor=prior_elicitor,
        )

        # 4. Check Strict Scientific Convergence Criterion (Delta on global mu < 0.01 for all channels)
        converged = False
        if previous_global_summary is not None:
            max_mu_change = 0.0

            for ch, current_params in global_summary.items():
                if ch in previous_global_summary:
                    prev_mu = previous_global_summary[ch].get("mean", 0.0)
                    curr_mu = current_params.get("mean", 0.0)
                    change = abs(curr_mu - prev_mu)

                    if change > max_mu_change:
                        max_mu_change = change

            if max_mu_change < 0.01:
                logger.info(
                    f"Convergence parameter achieved at step {round_num}! Max global mu delta evaluated at: {max_mu_change:.4f}"
                )
                converged = True

        # Commit structured round metric representation
        total_eps_this_round = eps_per_round * len(active_trainers)

        results.append(
            {
                "round_num": round_num,
                "global_summary": global_summary,
                "per_participant_surprise": surprise_scores,
                "epsilon_spent_per_participant": eps_per_round,
                "epsilon_spent_total": total_eps_this_round,
            }
        )

        round_path = results_dir / f"round_{round_num}.json"
        with open(round_path, "w") as f:
            # global_summary values may contain numpy floats — convert first
            json.dump(results[-1], f, indent=2, default=float)
        logger.info(f"Round {round_num} results saved to {round_path}")

        # Halt logically prior to executing redundant iteration steps
        if converged:
            break

        previous_global_summary = global_summary

    return results
