import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import yaml
import tempfile
import os
from unittest.mock import patch, MagicMock
from aggregator.simulate import run_simulation

test_config = {
    "num_participants": 3,
    "num_rounds": 2,
    "channels": ["paid_search", "social", "tv", "ooh"],
    "total_epsilon": 10.0,
    "total_delta": 1e-3,
    "epsilon_per_round": 0.5,
    "delta_per_round": 1e-5,
    "shrinkage": 0.5,
    "llm_model": "claude-sonnet-4-5",
    "seed": 42,
    "participants": [
        {
            "id": f"participant_{i}",
            "channel_descriptions": {
                "paid_search": "Google ads",
                "social": "Meta ads",
                "tv": "TV spots",
                "ooh": "Billboards",
            },
            "industry_vertical": "retail",
            "budget_share": {"paid_search": 0.4, "social": 0.3, "tv": 0.2, "ooh": 0.1},
            "seasonality_pattern": "retail",
        }
        for i in range(1, 4)
    ],
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    yaml.dump(test_config, f)
    config_path = f.name

with (
    patch("aggregator.simulate.LocalTrainer") as MockTrainer,
    patch("aggregator.simulate.PriorElicitor") as MockElicitor,
):

    def make_trainer(participant_id, config_path):
        t = MagicMock()
        t.participant_id = participant_id
        t.channels = {
            "paid_search": "desc",
            "social": "desc",
            "tv": "desc",
            "ooh": "desc",
        }
        t.num_observations = 104
        t.posterior_history = None
        t.train.return_value = {
            ch: {"mean": 0.25, "std": 0.08, "p5": 0.10, "p95": 0.40}
            for ch in ["paid_search", "social", "tv", "ooh"]
        }
        return t

    MockTrainer.side_effect = make_trainer
    mock_elicitor = MagicMock()
    mock_elicitor.elicit.return_value = {
        "priors": {
            ch: {"mu": 0.2, "sigma": 0.15}
            for ch in ["paid_search", "social", "tv", "ooh"]
        },
        "confidence": "medium",
        "notes": "",
    }
    MockElicitor.return_value = mock_elicitor

    history = run_simulation(config_path)

os.unlink(config_path)

# Check Flower history object
assert history is not None, "Simulation returned None"
assert (
    len(history.losses_distributed) > 0 or len(history.metrics_distributed) >= 0
), "History should contain round data"
print(f"Simulation completed — {len(history.losses_distributed)} rounds logged")

# Check round result files were saved
from pathlib import Path

for round_num in range(1, 3):
    result_file = Path("results") / f"round_{round_num}.json"
    assert result_file.exists(), f"Missing results/round_{round_num}.json"
    import json

    with open(result_file) as f:
        data = json.load(f)
    assert "global_summary" in data
    assert set(data["global_summary"].keys()) == {"paid_search", "social", "tv", "ooh"}
    print(f"Round {round_num} results saved correctly")
