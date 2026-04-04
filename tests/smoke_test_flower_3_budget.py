import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from participants.flower_client import MMMClient
from unittest.mock import MagicMock
from privacy.budget_tracker import PrivacyBudgetTracker
import numpy as np

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


# Tiny budget — exhausts after 1 round
tight_tracker = PrivacyBudgetTracker(
    total_epsilon=0.4, total_delta=1e-3, participant_ids=["participant_1"]
)
trainer = make_mock_trainer("participant_1")
client = MMMClient(
    trainer=trainer,
    budget_tracker=tight_tracker,
    epsilon_per_round=0.5,
    delta_per_round=1e-5,
)

global_means = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
global_stds = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32)

# First fit — exhausts budget
result_params, num_samples, metrics = client.fit(
    parameters=[global_means, global_stds], config={}
)

# Check exhausted flag returned and NOT zeros
assert metrics.get("exhausted") == True, "Should flag exhaustion"
assert num_samples == 0, "Exhausted client should return 0 samples"
assert not np.allclose(
    result_params[0], np.zeros(4)
), "Exhausted client should return current params, not zeros"
print("Budget exhaustion returns current params with exhausted flag")
