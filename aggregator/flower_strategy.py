import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    EvaluateRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from typing import List, Tuple, Dict, Optional, Union, Any
import json
import numpy as np
import logging

from aggregator.hierarchical import hierarchical_pool

logger = logging.getLogger(__name__)


class FederatedMMMStrategy(FedAvg):
    def __init__(
        self,
        prior_elicitor,
        channels,
        participants_config=None,
        shrinkage=0.5,
        *args,
        **kwargs,
    ):
        """
        Federated target orchestrator overriding standard FedAvg strategy.
        Utilizes mathematically advanced localized prior injection and global shrinkage protocols.
        """
        super().__init__(*args, **kwargs)
        self.prior_elicitor = prior_elicitor
        self.channels = channels
        self.participants_config = participants_config or {}
        self.shrinkage = shrinkage

        # Enforce canonical dimensional arrays mapping
        if isinstance(self.channels, dict):
            self.channel_names = list(self.channels.keys())
        else:
            self.channel_names = list(self.channels)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Overrides configure_fit to robustly inject current round LLM priors serialized as JSON to each mapped client.
        """
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        configured_instructions = []
        for client_proxy, fit_ins in client_instructions:

            # Map specific nodes
            cid = getattr(client_proxy, "cid", None)
            p_config = self.participants_config.get(cid, {}) if cid else {}
            posterior_history = getattr(client_proxy, "posterior_history", None)

            try:
                # 1. Elicit customized independent LLM node priors
                raw_elicitation = self.prior_elicitor.elicit(
                    participant_config=p_config,
                    channels=self.channels,
                    posterior_history=posterior_history,
                )
                priors = raw_elicitation.get("priors", {})
            except Exception as e:
                logger.error(
                    f"Round {server_round} | Failed to elicit prior dynamically for client {cid}: {e}"
                )
                priors = {}

            # JSON serialization for configuration payloads to cross FL bridge flawlessly
            fit_ins.config["llm_priors"] = json.dumps(priors)
            configured_instructions.append((client_proxy, fit_ins))

        return configured_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Union[Tuple[Any, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Overrides aggregate_fit to securely isolate parameters and mathematically apply hierarchical_pool merging vs naïve averaging.
        """
        if not results:
            return None, {}

        list_of_summaries = []

        for client_proxy, fit_res in results:

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if not ndarrays or len(ndarrays) < 2:
                continue

            means_vec = ndarrays[0]
            stds_vec = ndarrays[1]

            # Repackage dimensional representation out from NumPy Array payload block
            summary = {}
            for i, ch in enumerate(self.channel_names):
                if i < len(means_vec):
                    # We inject a pseudo-variance placeholder representing localized output bounds
                    # given the NumPy payload structurally transports exclusively dimensional array mappings
                    summary[ch] = {
                        "mean": float(means_vec[i]),
                        "std": float(stds_vec[i]) if i < len(stds_vec) else 0.15,
                    }

            list_of_summaries.append(summary)

        # 2. Trigger Hierarchical Merge to computationally stabilize outliers
        shrunk_summary = hierarchical_pool(list_of_summaries, shrinkage=self.shrinkage)

        # Reform boundaries matching output pipeline parameters schema strictly
        aggregated_means = []
        for ch in self.channel_names:
            ch_data = shrunk_summary.get(ch, {})
            aggregated_means.append(ch_data.get("mean", 0.0))

        aggregated_ndarrays = [np.array(aggregated_means, dtype=np.float32)]

        # Back to bytes/structural boundaries safely conforming directly to flwr core architectures
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, EvaluateRes]],
        failures: List[Union[Tuple[Any, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Overrides aggregate_evaluate determining success analytically via localized mean posterior variance minimization across client bounds.
        """
        if not results:
            return None, {}

        total_variance = 0.0
        total_samples = 0

        for client_proxy, eval_res in results:
            loss = (
                eval_res.loss
            )  # The exact mean posterior variance resolved dynamically
            num_samples = eval_res.num_examples

            total_variance += loss * num_samples
            total_samples += num_samples

        mean_posterior_variance = (
            total_variance / total_samples if total_samples > 0 else 0.0
        )

        logger.info(
            f"Round {server_round} aggregated mean posterior variance derived dynamically across endpoints: {mean_posterior_variance}"
        )

        return mean_posterior_variance, {
            "mean_posterior_variance": mean_posterior_variance
        }
