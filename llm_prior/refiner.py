import json
from typing import Dict, Any


def build_refinement_prompt(
    participant_config: Dict[str, Any],
    channels: Dict[str, str],
    previous_priors: Dict[str, Dict[str, float]],
    posterior_summary: Dict[str, Dict[str, float]],
    surprise_scores: Dict[str, float],
) -> str:
    """
    Constructs a structured prompt for prior refinement based on previous round's results.

    Args:
        participant_config: Dictionary containing participant details like industry_vertical.
        channels: Dictionary mapping channel names to their descriptions.
        previous_priors: Dict mapping channel to {"mu": float, "sigma": float}
        posterior_summary: Dict mapping channel to {"posterior_mean": float, "posterior_std": float}
        surprise_scores: Dict mapping channel to KL divergence score

    Returns:
        A formatted prompt string for refinement elicitation.
    """
    industry_vertical = participant_config.get("industry_vertical", "Unknown")
    budget_shares = participant_config.get("budget_share", {})
    seasonality = participant_config.get("seasonality", "Unknown")

    prompt_parts = []

    # 1. Context and Objective
    prompt_parts.append(
        "You are an expert Marketing Mix Modeling (MMM) consultant.\n"
        "Your task is to refine prior distributions for marketing channel effectiveness for the next round of modeling.\n"
        "You will review the priors you provided previously, the resulting posteriors from the model, and the 'surprise' (KL divergence) metric.\n"
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

    # 4. Results Table
    prompt_parts.append("\n### Previous Round Results & Surprise Scores")
    prompt_parts.append(
        "Below is a comparison of your previous priors vs. the model's posteriors, "
        "along with a Surprise Score (KL Divergence) indicating how much the model deviated from your prior."
    )

    for ch in channels.keys():
        prior = previous_priors.get(ch, {})
        post = posterior_summary.get(ch, {})
        surprise = surprise_scores.get(ch, 0.0)

        pr_mu = prior.get("mu", "N/A")
        pr_sig = prior.get("sigma", "N/A")
        po_mu = post.get("mean", "N/A")
        po_sig = post.get("std", "N/A")

        if isinstance(surprise, float):
            surprise_str = f"{surprise:.4f}"
        else:
            surprise_str = str(surprise)

        prompt_parts.append(
            f"\n**Channel:** {ch}\n"
            f"- Prior: N(mu={pr_mu}, sigma={pr_sig})\n"
            f"- Posterior: N(mean={po_mu}, std={po_sig})\n"
            f"- Surprise Score (KL divergence): {surprise_str}"
        )

    # 5. Instructions
    prompt_parts.append(
        "\n### Task Instructions\n"
        "Based on the results above, please revise the prior parameters (mu and sigma) for the next round.\n"
        "Critically: You MUST revise your priors for any channel where the Surprise Score is > 0.5. "
        "A KL > 0.5 indicates the model found strong evidence disagreeing with your previous prior.\n"
        "For these high-surprise channels, adjust your mu towards the posterior mean, OR increase your sigma "
        "to reflect higher uncertainty. Provide your reasoning for each channel.\n\n"
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
