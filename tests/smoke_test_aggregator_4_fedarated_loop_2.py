from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from aggregator.federated_loop import run_federated_training
import yaml, tempfile, os

# Write a minimal test config
test_config = {
    "num_participants": 2,
    "num_rounds": 10,
    "channels": ["paid_search", "social", "tv", "ooh"],
    "total_epsilon": 10,
    "total_delta": 1e-3,
    "epsilon_per_round": 0.6,  # will exhaust after round 1 (0.6 > 0.5 remaining after first)
    "delta_per_round": 1e-5,
    "llm_model": "claude-sonnet-4-5",
    "seed": 42,
    "participants": [
        {
            "id": "participant_1",
            "channels": {
                "paid_search": "desc",
                "social": "desc",
                "tv": "desc",
                "ooh": "desc",
            },
        },
        {
            "id": "participant_2",
            "channels": {
                "paid_search": "desc",
                "social": "desc",
                "tv": "desc",
                "ooh": "desc",
            },
        },
    ],
}

# Write to temp yaml
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    yaml.dump(test_config, f)
    config_path = f.name

# Patch LocalTrainer and PriorElicitor to avoid real MCMC and API calls
with (
    patch("aggregator.federated_loop.LocalTrainer") as MockTrainer,
    patch("aggregator.federated_loop.PriorElicitor") as MockElicitor,
):

    # Setup mock trainer instances
    def make_trainer(participant_id, participant_config, channels):
        t = MagicMock()
        t.participant_id = participant_id
        t.participant_config = participant_config
        t.channels = channels
        t.posterior_history = None
        t.train.return_value = {
            ch: {"mean": 0.25, "std": 0.08, "p5": 0.10, "p95": 0.40, "r_hat": 1.01}
            for ch in channels
        }
        return t

    MockTrainer.side_effect = make_trainer
    mock_elicitor_instance = MagicMock()
    mock_elicitor_instance.elicit.return_value = {
        "priors": {
            ch: {"mu": 0.2, "sigma": 0.15}
            for ch in ["paid_search", "social", "tv", "ooh"]
        },
        "confidence": "medium",
        "notes": "",
    }
    MockElicitor.return_value = mock_elicitor_instance

    results = run_federated_training(config_path)

os.unlink(config_path)

print(f"\nTotal rounds completed: {len(results)}")
print(f"Round numbers: {[r['round_num'] for r in results]}")

# Checks
assert len(results) > 0, "No rounds completed"
assert results[0]["round_num"] == 1
assert "global_summary" in results[0]
assert "per_participant_surprise" in results[0]
assert "epsilon_spent_per_participant" in results[0]

for r in results:
    if not r["global_summary"]:
        print(f"  Round {r['round_num']}: empty summary (budget exhausted) — expected")
        continue
    assert set(r["global_summary"].keys()) == {
        "paid_search",
        "social",
        "tv",
        "ooh",
    }, f"Round {r['round_num']} missing channels"
    print(f"  Round {r['round_num']}: channels present")

print("federated_loop runs end-to-end")
print("All round results have correct structure")
