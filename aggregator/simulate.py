import flwr as fl
import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

from participants.flower_client import MMMClient
from aggregator.flower_strategy import FederatedMMMStrategy
from llm_prior.elicitor import PriorElicitor
from privacy.budget_tracker import PrivacyBudgetTracker

try:
    from participants.local_trainer import LocalTrainer
except ImportError:
    # A lightweight mock definition if LocalTrainer has not been resolved from this scope
    class LocalTrainer:
        def __init__(self, participant_id, participant_config, channels=None):
            self.participant_id = participant_id
            self.participant_config = participant_config
            self.channels = channels or {}
            self.posterior_history = []

        def train(self, valid_priors):
            return {ch: {"mean": 0.25, "std": 0.08} for ch in self.channels}


logger = logging.getLogger(__name__)


def run_simulation(config_path: str):
    """
    Executes a federated learning loop using the robust Flower simulation engine
    driven concurrently by Ray multiprocessing logic.
    """
    # 1. Evaluate configuration
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    num_clients = config.get("num_participants", 2)
    num_rounds = config.get("num_rounds", 5)

    total_epsilon = config.get("total_epsilon", 10.0)
    total_delta = config.get("total_delta", 1e-4)
    epsilon_per_round = config.get("epsilon_per_round", 1.0)
    delta_per_round = config.get("delta_per_round", 1e-5)

    participants_list = config.get("participants", [])
    participants_config = {
        str(p.get("id", str(i))): p for i, p in enumerate(participants_list)
    }
    participant_ids = list(participants_config.keys())

    # Scale sequential nodes dynamically if configurations are implicit
    while len(participant_ids) < num_clients:
        pad_id = f"participant_{len(participant_ids)}"
        participant_ids.append(pad_id)
        participants_config[pad_id] = {"id": pad_id}

    global_channels = config.get("channels", ["paid_search", "social", "tv", "ooh"])

    # 2. Instantiate Stateful Shared Environment Globals
    prior_elicitor = PriorElicitor(
        model_name=config.get("llm_model", "claude-3-5-sonnet-latest")
    )

    budget_tracker = PrivacyBudgetTracker(
        total_epsilon=total_epsilon,
        total_delta=total_delta,
        participant_ids=participant_ids,
    )

    # 3. Virtual Node Sandbox mapping `cid` sequential invocations back into participant scope
    def client_fn(cid: str) -> fl.client.Client:
        try:
            p_idx = int(cid)
            p_id = (
                participant_ids[p_idx]
                if p_idx < len(participant_ids)
                else f"participant_{cid}"
            )
        except ValueError:
            p_id = cid

        p_config = participants_config.get(p_id, {})
        p_channels = p_config.get(
            "channel_descriptions", {ch: ch for ch in global_channels}
        )

        # Instantiate LocalTrainer backend independently
        trainer = LocalTrainer(participant_id=p_id, config_path=f"config/{p_id}.yaml")

        # Output the structural NumPY endpoint wrapped strictly bridging into the `fl.client.Client` interface
        client = MMMClient(
            trainer=trainer,
            budget_tracker=budget_tracker,
            epsilon_per_round=epsilon_per_round,
            delta_per_round=delta_per_round,
        )

        return client.to_client()

    # 4. Logging & Configuration Payload Engine overriding custom JSON Dumps
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    class LoggingMMMStrategy(FederatedMMMStrategy):
        def aggregate_fit(self, server_round, results, failures):
            # Exploit evaluated hierarchy boundaries
            parameters, metrics = super().aggregate_fit(server_round, results, failures)

            # Repackage the output to verify analytical shape logging explicitly to disk dynamically
            if parameters is not None:
                ndarrays = fl.common.parameters_to_ndarrays(parameters)
                if len(ndarrays) > 0:
                    means = ndarrays[0]

                    global_posterior = {}
                    for i, ch in enumerate(global_channels):
                        if i < len(means):
                            global_posterior[ch] = {"mean": float(means[i])}

                    output_path = results_dir / f"round_{server_round}.json"
                    with open(output_path, "w") as out_f:
                        json.dump(
                            {
                                "round_num": server_round,
                                "global_summary": global_posterior,
                            },
                            out_f,
                            indent=2,
                        )

                    logger.info(
                        f"Tracking dynamically saved globally aggregated mathematical posterior bounds to {output_path}"
                    )

            return parameters, metrics

    strategy = LoggingMMMStrategy(
        prior_elicitor=prior_elicitor,
        channels=global_channels,
        participants_config=participants_config,
        shrinkage=config.get("shrinkage", 0.5),
    )

    # 5. Boot Ray's Sandbox Engine
    config_server = fl.server.ServerConfig(num_rounds=num_rounds)

    # Strict deterministic resource configurations (prevents parallel thread-lockouts in simulated environments)
    ray_init_args = {"ignore_reinit_error": True, "num_cpus": os.cpu_count() or 4}

    # Initiates iterative background operations and yields the history log directly
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=config_server,
        strategy=strategy,
        ray_init_args=ray_init_args,
    )

    return history
