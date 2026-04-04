import json
from typing import Dict, Any, List, Optional


def build_elicitation_prompt(
    participant_config: Dict[str, Any],
    channels: Dict[str, str],
    posterior_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Constructs a structured prompt for prior elicitation.

    Args:
        participant_config: Dictionary containing participant context like
                            industry_vertical, budget_share, seasonality, etc.
        channels: Dictionary mapping channel names to their descriptions.
        posterior_history: Optional list of dictionaries containing previous round posteriors.
                           e.g. [{"round": 1, "posteriors": {"paid_search": 0.3, ...}}]

    Returns:
        A formatted prompt string.
    """
    industry_vertical = participant_config.get("industry_vertical", "Unknown")
    budget_shares = participant_config.get("budget_share", {})
    seasonality = participant_config.get("seasonality_pattern", "Unknown")

    prompt_parts = []

    # 1. Context and Objective
    prompt_parts.append(
        "You are an expert Marketing Mix Modeling (MMM) consultant.\n"
        "Your task is to elicit prior distributions for marketing channel effectiveness based on the provided context.\n"
    )

    # 2. Participant Configuration
    prompt_parts.append(f"### Participant Context")
    prompt_parts.append(f"Industry Vertical: {industry_vertical}")
    prompt_parts.append(f"Seasonality: {seasonality}")

    if budget_shares:
        prompt_parts.append(f"\nBudget Shares:")
        for ch, share in budget_shares.items():
            prompt_parts.append(f"- {ch}: {share}")

    # 3. Channel Descriptions
    if channels:
        prompt_parts.append(f"\n### Media Channels")
        for ch, desc in channels.items():
            prompt_parts.append(f"- {ch}: {desc}")

    # 4. Posterior History
    if posterior_history is not None and len(posterior_history) > 0:
        prompt_parts.append("\n### Previous Round Posteriors")
        prompt_parts.append(
            "Below is a table-like summary of the posterior means from previous modeling rounds:"
        )
        for history_entry in posterior_history:
            round_num = history_entry.get("round", "Unknown")
            posteriors = history_entry.get("posteriors", {})
            prompt_parts.append(f"\nRound: {round_num}")
            for ch, value in posteriors.items():
                prompt_parts.append(f"- {ch}: {value}")

    # 5. Output Instructions
    prompt_parts.append(
        "\n### Task Instructions\n"
        "Based on the above context, estimate the prior parameters (mu and sigma) for each media channel's effectiveness.\n"
        "Provide your reasoning for each channel.\n\n"
        "Output ONLY valid JSON matching the exact schema below. Do not include any markdown formatting "
        "(e.g. ```json), no explanatory text before or after the JSON.\n\n"
        "Expected JSON output schema:\n"
        "{\n"
        '  "priors": {\n'
        '    "paid_search": {"mu": 0.35, "sigma": 0.08, "reasoning": "..."},\n'
        '    "social":      {"mu": 0.18, "sigma": 0.10, "reasoning": "..."},\n'
        '    "tv":          {"mu": 0.25, "sigma": 0.12, "reasoning": "..."},\n'
        '    "ooh":         {"mu": 0.10, "sigma": 0.09, "reasoning": "..."}\n'
        "  },\n"
        '  "confidence": "medium",\n'
        '  "notes": "..."\n'
        "}"
    )

    return "\n".join(prompt_parts)
