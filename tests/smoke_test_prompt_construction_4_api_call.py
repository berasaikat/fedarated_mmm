import yaml
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm_prior.elicitor import PriorElicitor
from llm_prior.validator import validate_priors

load_dotenv()

with open("config/participant_1.yaml") as f:
    cfg = yaml.safe_load(f)

channels = cfg["channel_descriptions"]
elicitor = PriorElicitor()

print("Calling Claude API...")
result = elicitor.elicit(
    participant_config=cfg, channels=channels, posterior_history=None
)

print(f"\nRaw result keys: {result.keys()}")
print(f"Confidence: {result.get('confidence')}")
print(f"Notes: {result.get('notes')}")
print("\nPriors returned:")
for ch, vals in result["priors"].items():
    print(f"  {ch}: mu={vals['mu']}, sigma={vals['sigma']}")
    print(f"    reasoning: {vals.get('reasoning', 'MISSING')[:80]}...")

# Validate the output
validated = validate_priors(result["priors"], channels)
print("\nAfter validation:")
for ch, vals in validated.items():
    print(f"  {ch}: mu={vals['mu']}, sigma={vals['sigma']}")
