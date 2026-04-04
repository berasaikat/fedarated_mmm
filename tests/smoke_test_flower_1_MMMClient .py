import numpy as np
import json
from unittest.mock import MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from participants.flower_client import MMMClient
from privacy.budget_tracker import PrivacyBudgetTracker

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

# Test 1 — get_parameters before training returns zeros
params = client.get_parameters(config={})
assert len(params) == 2, f"Expected 2 arrays (means + stds), got {len(params)}"
assert len(params[0]) == 4, "Means array wrong length"
assert len(params[1]) == 4, "Stds array wrong length"
assert all(params[0] == 0.0), "Pre-training means should be zeros"
print("Test 1 passed — get_parameters returns zeros before training")

# Test 2 — fit with global means as parameters
global_means = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
global_stds = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32)

result_params, num_samples, metrics = client.fit(
    parameters=[global_means, global_stds], config={}
)

assert len(result_params) == 2, "fit() should return [means, stds]"
assert len(result_params[0]) == 4, "Output means wrong length"
assert len(result_params[1]) == 4, "Output stds wrong length"
assert num_samples == 104, f"Expected 104 samples, got {num_samples}"
assert metrics.get("status") == "success"
print("Test 2 passed — fit() returns correct structure")

# Test 3 — means are noisy (DP applied)
raw_means = np.array(
    [trainer.train.return_value[ch]["mean"] for ch in channels], dtype=np.float32
)
assert not np.allclose(
    result_params[0], raw_means
), "Output means should differ from raw posterior (DP noise not applied)"
print("Test 3 passed — DP noise applied to means")

# Test 4 — stds pass through unchanged
raw_stds = np.array(
    [trainer.train.return_value[ch]["std"] for ch in channels], dtype=np.float32
)
assert np.allclose(
    result_params[1], raw_stds, atol=1e-5
), "Stds should pass through unmodified"
print("Test 4 passed — stds pass through unchanged")

# Test 5 — get_parameters after training returns last posterior
params_after = client.get_parameters(config={})
assert not all(params_after[0] == 0.0), "Post-training means should not be zeros"
print("Test 5 passed — get_parameters reflects last posterior after training")
