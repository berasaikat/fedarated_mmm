import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from unittest.mock import MagicMock
from flwr.common import ndarrays_to_parameters, FitRes, Status, Code
from aggregator.flower_strategy import FederatedMMMStrategy

channels = ["paid_search", "social", "tv", "ooh"]

mock_elicitor = MagicMock()
mock_elicitor.elicit.return_value = {
    "priors": {ch: {"mu": 0.2, "sigma": 0.15} for ch in channels},
    "confidence": "medium",
    "notes": "",
}

strategy = FederatedMMMStrategy(
    prior_elicitor=mock_elicitor, channels=channels, shrinkage=0.5
)


# Simulate 3 clients with different posteriors
def make_fit_res(means, stds):
    params = ndarrays_to_parameters(
        [np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)]
    )
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=104,
        metrics={},
    )


results = [
    (MagicMock(), make_fit_res([0.4, 0.2, 0.3, 0.1], [0.08, 0.10, 0.09, 0.07])),
    (MagicMock(), make_fit_res([0.5, 0.3, 0.2, 0.2], [0.09, 0.08, 0.10, 0.08])),
    (MagicMock(), make_fit_res([0.3, 0.4, 0.4, 0.3], [0.07, 0.09, 0.08, 0.09])),
]

aggregated_params, metrics = strategy.aggregate_fit(
    server_round=1, results=results, failures=[]
)

assert aggregated_params is not None, "aggregate_fit returned None"

from flwr.common import parameters_to_ndarrays

ndarrays = parameters_to_ndarrays(aggregated_params)
assert len(ndarrays) >= 1, "Should return at least means array"

aggregated_means = ndarrays[0]
assert (
    len(aggregated_means) == 4
), f"Expected 4 channel means, got {len(aggregated_means)}"
assert all(np.isfinite(aggregated_means)), "Aggregated means contain NaN or Inf"

print(f"Aggregated means: {dict(zip(channels, aggregated_means.tolist()))}")

# Verify shrinkage pulled means toward grand mean
raw_means = np.array(
    [
        (0.4 + 0.5 + 0.3) / 3,  # paid_search fedavg
        (0.2 + 0.3 + 0.4) / 3,  # social fedavg
        (0.3 + 0.2 + 0.4) / 3,  # tv fedavg
        (0.1 + 0.2 + 0.3) / 3,  # ooh fedavg
    ]
)
grand_mean = np.mean(raw_means)
expected_shrunk = 0.5 * raw_means + 0.5 * grand_mean

for i, ch in enumerate(channels):
    assert (
        abs(aggregated_means[i] - expected_shrunk[i]) < 0.01
    ), f"{ch}: expected shrunk mean {expected_shrunk[i]:.4f}, got {aggregated_means[i]:.4f}"

print("Hierarchical shrinkage applied correctly in aggregate_fit")
