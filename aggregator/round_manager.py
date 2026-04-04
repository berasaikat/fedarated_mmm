from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from privacy.dp_sharing import dp_share_posterior
from privacy.budget_tracker import PrivacyBudgetTracker, PrivacyBudgetExhausted
from aggregator.fed_avg_posterior import fedavg_posterior
from llm_prior.surprise import compute_surprise
from llm_prior.validator import validate_priors

import logging

logger = logging.getLogger(__name__)


class RoundManager:
    def __init__(self, config: Dict[str, Any], budget_tracker: PrivacyBudgetTracker):
        """
        Initializes the aggregator round manager.

        Args:
            config: A dictionary containing hyperparameters like 'epsilon_per_round', 'delta_per_round', etc.
            budget_tracker: A component evaluating and bounding global privacy allocation mathematically.
        """
        self.config = config
        self.budget_tracker = budget_tracker

    def run_round(
        self, round_num: int, all_local_trainers: List[Any], prior_elicitor: Any
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[Any, Dict[str, float]]]:
        """
        Orchestrates an end-to-end iteration of the Federated Model.

        Args:
            round_num: The global aggregation round index.
            all_local_trainers: A list of participant trainer artifacts that expose data configs and local .train() operations.
            prior_elicitor: A `PriorElicitor` instance evaluating valid predictive priors.

        Returns:
            global_summary: The federated averaging representation.
            surprise_scores_per_participant: Mathematical KL bound surprises mapping pid -> channel -> score.
        """
        # Load operational parameter constraints
        eps = self.config.get("epsilon_per_round", 1.0)
        delta = self.config.get("delta_per_round", 1e-5)

        shared_summaries = []
        surprise_scores_per_participant = {}

        for trainer in all_local_trainers:
            pid = trainer.participant_id
            p_config = getattr(trainer, "participant_config", {})
            channels = getattr(trainer, "channels", {})
            history = getattr(trainer, "posterior_history", None)
            logger.info(f"Round {round_num} | Participant {pid} — eliciting priors")

            # 1. Elicit LLM priors
            raw_elicitation = prior_elicitor.elicit(
                participant_config=p_config,
                channels=channels,
                posterior_history=history,
            )
            raw_priors = raw_elicitation.get("priors", {})

            # Validate schema and parameter limits
            channel_names = channels.keys() if isinstance(channels, dict) else channels
            valid_priors = validate_priors(raw_priors, channel_names)

            # 2. Run local training structurally
            # Note: We execute localized `.train()` providing mathematical constraints securely.
            local_posterior = trainer.train(valid_priors)
            logger.info(
                f"Round {round_num} | Participant {pid} — training complete, "
                f"channels: {list(local_posterior.keys())}"
            )

            # Apply dynamic channel L2 sensitivity resolving if undefined globally
            if "sensitivity" not in self.config:
                from privacy.sensitivity import compute_l2_sensitivity

                sense = compute_l2_sensitivity(len(channels))
            else:
                sense = self.config["sensitivity"]

            # 3. Apply DP sharing
            try:
                noisy_shared_posterior = dp_share_posterior(
                    posterior_summary=local_posterior,
                    budget_tracker=self.budget_tracker,
                    participant_id=pid,
                    epsilon_per_round=eps,
                    delta_per_round=delta,
                )
            except PrivacyBudgetExhausted:
                logger.warning(
                    f"Round {round_num} | Participant {pid} — "
                    f"budget exhausted mid-round, skipping"
                )
                continue

            # Cache payload for aggregation
            shared_summaries.append(noisy_shared_posterior)

            # 4. Compute surprise scores natively isolating against the observed local output
            surprise = compute_surprise(valid_priors, local_posterior)
            surprise_scores_per_participant[pid] = surprise

            if (
                not hasattr(trainer, "posterior_history")
                or trainer.posterior_history is None
            ):
                trainer.posterior_history = []

            trainer.posterior_history.append(
                {
                    "round": round_num,
                    "posteriors": {
                        ch: {"mean": local_posterior[ch]["mean"]}
                        for ch in local_posterior
                        if "mean" in local_posterior.get(ch, {})
                    },
                }
            )

        # 5. Aggregate mathematically rigorous federated posteriors
        # Defaults to fed_avg directly (hierarchical inference can overlap mathematically independent of RoundManager orchestration).
        if not shared_summaries:
            logger.warning(
                f"Round {round_num} | No participants shared posteriors — "
                f"all budgets exhausted mid-round. Returning empty."
            )
            return {}, surprise_scores_per_participant

        global_summary = fedavg_posterior(shared_summaries)
        logger.info(
            f"Round {round_num} | Global summary: "
            f"{ {ch: round(v['mean'], 4) for ch, v in global_summary.items()} }"
        )

        return global_summary, surprise_scores_per_participant
