import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm_prior.elicitor import PriorElicitor
from llm_prior.surprise import compute_surprise, aggregate_surprise
from llm_prior.validator import validate_priors

with open("config/participant_1.yaml") as f:
    cfg = yaml.safe_load(f)

channels = cfg["channel_descriptions"]
elicitor = PriorElicitor()

# --- Round 1: elicit priors ---
print("=== Round 1: Eliciting priors ===")
round1_result = elicitor.elicit(participant_config=cfg, channels=channels)
round1_priors = validate_priors(round1_result["priors"], channels)
print(f"Round 1 priors: { {ch: v['mu'] for ch, v in round1_priors.items()} }")

# --- Simulate posterior from MCMC (using Phase 3 key format) ---
simulated_posterior = {
    ch: {"mean": v["mu"] * 1.3, "std": v["sigma"] * 0.8}
    for ch, v in round1_priors.items()
}

# --- Compute surprise ---
surprise = compute_surprise(round1_priors, simulated_posterior)
mean_surprise = aggregate_surprise(surprise)
print(f"\nSurprise scores: { {ch: round(v, 3) for ch, v in surprise.items()} }")
print(f"Mean KL: {mean_surprise:.4f}")

# --- Round 2: refine priors ---
print("\n=== Round 2: Refining priors ===")
round2_result = elicitor.refine(
    participant_config=cfg,
    channels=channels,
    previous_priors=round1_priors,
    posterior_summary=simulated_posterior,
    surprise_scores=surprise,
)
round2_priors = validate_priors(round2_result["priors"], channels)
print(f"Round 2 priors: { {ch: v['mu'] for ch, v in round2_priors.items()} }")

# Check that high-surprise channels were actually updated
print("\n--- Prior shift check ---")
for ch in channels:
    delta = abs(round2_priors[ch]["mu"] - round1_priors[ch]["mu"])
    flag = "⚠️  NO CHANGE" if delta < 0.001 and surprise.get(ch, 0) > 0.5 else "✓"
    print(f"  {ch}: Δmu={delta:.4f} {flag}")
