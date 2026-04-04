import sys
import jax.numpy as jnp
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from participants.local_trainer import LocalTrainer

# Use participant_1 as the test subject
trainer = LocalTrainer(
    participant_id="participant_1", config_path="config/participant_1.yaml"
)

# Step 1 — check data loads correctly
spend_matrix, revenue, spend_cols = trainer.load_data()
print(
    f"Data loaded — spend shape: {spend_matrix.shape}, revenue shape: {revenue.shape}"
)
print(f"Channels: {spend_cols}")

# Step 2 — define a simple prior dict (using channel names, not integers)
priors_dict = {ch: {"mu": 0.2, "sigma": 0.5} for ch in spend_cols}
print(f"Priors built: {priors_dict}")

# Step 3 — run training (this will take 2-4 mins)
print("\nRunning MCMC — this will take a few minutes...")
summary = trainer.train(priors_dict=priors_dict)

# Step 4 — inspect output
print("\n--- Posterior Summary ---")
for param, stats in summary.items():
    print(
        f"  {param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, r_hat={stats['r_hat']:.3f}"
    )
