import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from participants.flower_client import MMMClient
from privacy.budget_tracker import PrivacyBudgetTracker
from unittest.mock import MagicMock
import numpy as np
import json

channels = ["paid_search", "social", "tv", "ooh"]


def make_mock_trainer(pid):
    trainer = MagicMock()
    trainer.participant_id = pid
    trainer.channels = {ch: f"{ch} description" for ch in channels}
    trainer.num_observations = 104
    trainer.train.return_value = {
        ch: {"mean": 0.3 + i * 0.05, "std": 0.08, "p5": 0.15, "p95": 0.45}
        for i, ch in enumerate(channels)
    }
    return trainer


tracker = PrivacyBudgetTracker(
    total_epsilon=10.0, total_delta=1e-3, participant_ids=["participant_1"]
)
trainer = make_mock_trainer("participant_1")
client = MMMClient(
    trainer=trainer, budget_tracker=tracker, epsilon_per_round=0.5, delta_per_round=1e-5
)

# Inject LLM priors via config
llm_priors = {ch: {"mu": 0.5, "sigma": 0.05, "reasoning": "test"} for ch in channels}
config_with_priors = {"llm_priors": json.dumps(llm_priors)}

global_means = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
global_stds = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32)

client.fit(parameters=[global_means, global_stds], config=config_with_priors)

# Check trainer.train was called with LLM priors (mu=0.5) not global means (0.2)
call_args = trainer.train.call_args[0][0]  # first positional arg
for ch in channels:
    assert (
        abs(call_args[ch]["mu"] - 0.5) < 1e-6
    ), f"{ch}: expected mu=0.5 from LLM priors, got {call_args[ch]['mu']}"
print("LLM priors correctly used when present in config")

# Test fallback — no LLM priors in config uses global means
trainer2 = make_mock_trainer("participant_1")
client2 = MMMClient(
    trainer=trainer2,
    budget_tracker=tracker,
    epsilon_per_round=0.5,
    delta_per_round=1e-5,
)
client2.fit(parameters=[global_means, global_stds], config={})

call_args2 = trainer2.train.call_args[0][0]
for ch in channels:
    assert (
        abs(call_args2[ch]["mu"] - 0.2) < 1e-6
    ), f"{ch}: expected mu=0.2 from global means fallback"
print("Falls back to global means when no LLM priors in config")
