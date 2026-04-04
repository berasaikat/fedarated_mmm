import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm_prior.prompt_builder import build_elicitation_prompt

with open("config/participant_1.yaml") as f:
    cfg = yaml.safe_load(f)

channels = cfg["channel_descriptions"]
prompt = build_elicitation_prompt(
    participant_config=cfg, channels=channels, posterior_history=None
)

print(prompt)
print("\n--- CHECKS ---")
print(f"✓ industry_vertical present: {cfg['industry_vertical'] in prompt}")
print(f"✓ budget_share present: {str(list(cfg['budget_share'].keys())[0]) in prompt}")
print(f"✓ all channels present: {all(ch in prompt for ch in channels)}")
print(f"✓ JSON schema present: {'priors' in prompt}")
print(f"✓ no markdown instruction present: {'Do not include any markdown' in prompt}")
