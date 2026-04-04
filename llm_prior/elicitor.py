import os
import json
from time import time
import anthropic
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from llm_prior.prompt_builder import build_elicitation_prompt
from llm_prior.refiner import build_refinement_prompt

load_dotenv()


class PriorElicitor:
    def __init__(self, anthropic_client=None, model_name="claude-3-5-sonnet-latest"):
        if anthropic_client is None:
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            self.client = anthropic_client
        self.model_name = model_name

    def _call_llm_and_parse(self, prompt: str, channels: Dict[str, str]) -> dict:
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}],
                )

                response_text = response.content[0].text

                # Attempt to clean potential markdown formatting
                clean_text = response_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                parsed_data = json.loads(clean_text)

                if "priors" not in parsed_data:
                    raise ValueError("Response missing 'priors' key.")

                priors = parsed_data["priors"]

                # Validate that all requested channels have mu and sigma keys
                for ch in channels.keys():
                    if ch not in priors:
                        raise ValueError(
                            f"Channel '{ch}' missing from priors in LLM response."
                        )

                    if "mu" not in priors[ch] or "sigma" not in priors[ch]:
                        raise ValueError(f"Channel '{ch}' is missing 'mu' or 'sigma'.")

                return parsed_data

            except json.JSONDecodeError as e:
                error_msg = (
                    f"Failed to parse JSON: {str(e)}\nResponse text: {response_text}"
                )
                last_error = ValueError(error_msg)
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
            except ValueError as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        # Raise the last encountered error if max_retries is reached
        raise ValueError(
            f"Failed to elicit valid priors after {max_retries} retries. Last error: {str(last_error)}"
        )

    def elicit(
        self,
        participant_config: Dict[str, Any],
        channels: Dict[str, str],
        posterior_history: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        # Normalize channels to dict if a list was passed
        if isinstance(channels, list):
            channels = {ch: ch for ch in channels}

        prompt = build_elicitation_prompt(
            participant_config=participant_config,
            channels=channels,
            posterior_history=posterior_history,
        )
        return self._call_llm_and_parse(prompt, channels)

    def refine(
        self,
        participant_config: Dict[str, Any],
        channels: Dict[str, str],
        previous_priors: Dict[str, Dict[str, float]],
        posterior_summary: Dict[str, Dict[str, float]],
        surprise_scores: Dict[str, float],
    ) -> dict:
        prompt = build_refinement_prompt(
            participant_config=participant_config,
            channels=channels,
            previous_priors=previous_priors,
            posterior_summary=posterior_summary,
            surprise_scores=surprise_scores,
        )
        return self._call_llm_and_parse(prompt, channels)
