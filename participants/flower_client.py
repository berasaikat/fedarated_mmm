import flwr as fl
import numpy as np
import logging
from typing import Dict, Any, List

from privacy.dp_sharing import dp_share_posterior
from privacy.budget_tracker import PrivacyBudgetExhausted

logger = logging.getLogger(__name__)


class MMMClient(fl.client.NumPyClient):
    def __init__(
        self, trainer, budget_tracker, epsilon_per_round: float, delta_per_round: float
    ):
        """
        Flower framework client wrapping the localized Marketing Mix Model (MMM) Trainer.

        Args:
            trainer: The LocalTrainer executing MCMC iteratively.
            budget_tracker: Evaluates privacy budgets centrally over lifetime contexts.
            epsilon_per_round: Allocation of epsilon DP loss for each `.fit` operation.
            delta_per_round: Allocation of delta DP loss.
        """
        self.trainer = trainer
        self.budget_tracker = budget_tracker
        self.epsilon_per_round = epsilon_per_round
        self.delta_per_round = delta_per_round

        # Lock in a deterministic ordering of channels to map to/from numpy arrays reliably.
        channels_obj = getattr(self.trainer, "channels", {})
        if isinstance(channels_obj, dict):
            self.channel_names = list(channels_obj.keys())
        else:
            self.channel_names = list(channels_obj)

        self.last_posterior = None

    def _priors_from_parameters(self, parameters):
        global_means = parameters[0]
        return {
            ch: {"mu": float(global_means[i]), "sigma": 0.15}
            for i, ch in enumerate(self.channel_names)
            if i < len(global_means)
        }

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Returns the beta means of the latest local posterior encoded as NumPy parameters.
        """
        if not self.last_posterior:
            # Baseline behavior upon first-time server ping
            return [
                np.zeros(len(self.channel_names), dtype=np.float32),
                np.full(len(self.channel_names), 0.15, dtype=np.float32),
            ]

        means = [
            self.last_posterior.get(ch, {}).get("mean", 0.0)
            for ch in self.channel_names
        ]
        stds = [
            self.last_posterior.get(ch, {}).get("std", 0.15)
            for ch in self.channel_names
        ]
        return [np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)]

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        """
        The principal distributed execution block for Flower FL clients.
        Receives mathematical models (priors), optimizes locally via MCMC, bounds via DP, and responds.
        """
        llm_priors_json = config.get("llm_priors")

        if not parameters or len(parameters) == 0:
            raise ValueError(
                "Expected global model parameters arrays but received an empty list from the framework server."
            )

        global_means = parameters[0]

        if len(global_means) != len(self.channel_names):
            raise ValueError(
                f"Shape error: server array width ({len(global_means)}) doesn't map exactly to participant channels dimension ({len(self.channel_names)})."
            )

        if llm_priors_json:
            try:
                import json
                from llm_prior.validator import validate_priors

                raw_priors = json.loads(llm_priors_json)
                priors = validate_priors(raw_priors, self.channel_names)
                logger.info(
                    f"Using LLM priors for {getattr(self.trainer, 'participant_id', '?')}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse LLM priors from config: {e}, falling back to global means"
                )
                priors = self._priors_from_parameters(parameters)
        else:
            priors = self._priors_from_parameters(parameters)
        # 1. Construct generalized bounded priors corresponding to received parameters
        # priors = {}
        # for i, ch in enumerate(self.channel_names):
        #     priors[ch] = {
        #         "mu": float(global_means[i]),
        #         # Inject a standard deviation variance scale since `.fit()` exclusively passes point-estimate arrays via param list
        #         "sigma": 0.15
        #     }

        # 2. Run strictly localized Bayesian MCMC operation via internal wrapped logic
        local_posterior = self.trainer.train(priors)
        self.last_posterior = local_posterior

        pid = getattr(self.trainer, "participant_id", "unknown_pid")

        # 3. Assess budgets and apply mathematical protections iteratively
        try:
            noisy_posterior = dp_share_posterior(
                posterior_summary=local_posterior,
                budget_tracker=self.budget_tracker,
                participant_id=pid,
                epsilon_per_round=self.epsilon_per_round,
                delta_per_round=self.delta_per_round,
            )
        except PrivacyBudgetExhausted as e:
            logger.warning(
                f"Rejecting payload: participant {pid} consumed mathematical limits: {e}"
            )
            # If depleted logically block upstream transmission
            return self.get_parameters(config), 0, {"exhausted": True}

        # Re-pack the modified secure response back into NumPy standards
        noisy_means = [
            noisy_posterior[ch].get("mean", 0.0) for ch in self.channel_names
        ]
        noisy_stds = [noisy_posterior[ch].get("std", 0.15) for ch in self.channel_names]

        # Extract meaningful local dimensionality metadata to scale fed_avg reliably across non-uniform local densities
        num_samples = getattr(self.trainer, "num_observations", 1)
        metrics_dict = {"status": "success", "participant_id": str(pid)}

        # 4. Mandatory Return interface
        return (
            [
                np.array(noisy_means, dtype=np.float32),
                np.array(noisy_stds, dtype=np.float32),
            ],
            num_samples,
            metrics_dict,
        )

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        """
        Computes localized statistical evaluation insights directly into FL server callbacks.
        In this paradigm, 'loss' is represented strictly by the population mean posterior variance.
        """
        if not self.last_posterior:
            return 0.0, 1, {"status": "not_trained_yet"}

        variances = []
        for ch in self.channel_names:
            ch_data = self.last_posterior.get(ch, {})
            if "std" in ch_data:
                variances.append(ch_data["std"] ** 2)

        # High 'loss' == high model uncertainty
        loss = float(np.mean(variances)) if variances else 0.0

        num_samples = getattr(self.trainer, "num_observations", 1)
        metrics_dict = {"mean_posterior_variance": loss}

        return loss, num_samples, metrics_dict
