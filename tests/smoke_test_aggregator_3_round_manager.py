from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from aggregator.round_manager import RoundManager
from privacy.budget_tracker import PrivacyBudgetTracker


# Build mock trainers
def make_mock_trainer(pid, channels):
    trainer = MagicMock()
    trainer.participant_id = pid
    trainer.participant_config = {
        "participant_id": pid,
        "industry_vertical": "retail",
        "seasonality_pattern": "retail",
        "budget_share": {ch: 0.25 for ch in channels},
        "channel_descriptions": {ch: f"{ch} description" for ch in channels},
    }
    trainer.channels = {ch: f"{ch} description" for ch in channels}
    trainer.posterior_history = None
    trainer.train.return_value = {
        ch: {"mean": 0.3, "std": 0.08, "p5": 0.15, "p95": 0.45, "r_hat": 1.01}
        for ch in channels
    }
    return trainer


channels = ["paid_search", "social", "tv", "ooh"]
trainers = [make_mock_trainer(f"participant_{i}", channels) for i in range(1, 4)]

tracker = PrivacyBudgetTracker(
    total_epsilon=10.0,
    total_delta=1e-3,
    participant_ids=[t.participant_id for t in trainers],
)

config = {
    "epsilon_per_round": 0.5,
    "delta_per_round": 1e-5,
}

# Mock the LLM elicitor
mock_elicitor = MagicMock()
mock_elicitor.elicit.return_value = {
    "priors": {ch: {"mu": 0.2, "sigma": 0.15, "reasoning": "test"} for ch in channels},
    "confidence": "medium",
    "notes": "mocked",
}

round_manager = RoundManager(config=config, budget_tracker=tracker)

# Run round 1
global_summary, surprise_scores = round_manager.run_round(
    round_num=1, all_local_trainers=trainers, prior_elicitor=mock_elicitor
)

print(f"\nGlobal summary channels: {list(global_summary.keys())}")
print(f"Surprise scores participants: {list(surprise_scores.keys())}")

# Checks
assert set(global_summary.keys()) == set(channels), "Missing channels in global summary"
for ch in channels:
    assert "mean" in global_summary[ch], f"Missing mean for {ch}"
    assert "std" in global_summary[ch], f"Missing std for {ch}"
    assert isinstance(global_summary[ch]["mean"], float), f"Mean not float for {ch}"
print("Global summary has correct structure")

assert set(surprise_scores.keys()) == {
    "participant_1",
    "participant_2",
    "participant_3",
}
print("Surprise scores present for all participants")

# Check posterior_history was updated
for trainer in trainers:
    assert (
        trainer.posterior_history is not None
    ), f"{trainer.participant_id} posterior_history not updated"
    assert (
        len(trainer.posterior_history) == 1
    ), f"Expected 1 history entry, got {len(trainer.posterior_history)}"
print("Posterior history updated on all trainers after round 1")

# Check budget was spent
for trainer in trainers:
    rem_eps, _ = tracker.remaining(trainer.participant_id)
    assert rem_eps < 10.0, f"{trainer.participant_id} budget not spent"
print("Privacy budget correctly spent for all participants")

for round_num in range(2, 5):
    global_summary, surprise_scores = round_manager.run_round(
        round_num=round_num, all_local_trainers=trainers, prior_elicitor=mock_elicitor
    )

# Check posterior history accumulated across rounds
for trainer in trainers:
    assert (
        len(trainer.posterior_history) == 4
    ), f"Expected 4 history entries, got {len(trainer.posterior_history)}"
    rounds_logged = [h["round"] for h in trainer.posterior_history]
    assert rounds_logged == [1, 2, 3, 4], f"Wrong rounds logged: {rounds_logged}"
print("Posterior history accumulates correctly across rounds:")

# Check budget decrements correctly
for trainer in trainers:
    rem_eps, _ = tracker.remaining(trainer.participant_id)
    expected_remaining = 10.0 - (0.5 * 4)  # 4 rounds × 0.5 per round
    assert (
        abs(rem_eps - expected_remaining) < 1e-9
    ), f"Expected {expected_remaining} remaining, got {rem_eps}"
    print(f"  {trainer.participant_id}: remaining epsilon = {rem_eps:.1f}")
