import logging
from typing import Dict, Any, Iterable

logger = logging.getLogger(__name__)


def validate_priors(
    priors_dict: Dict[str, Any], channels: Iterable[str]
) -> Dict[str, Any]:
    """
    Validates elicited prior distributions for missing channels or out-of-bounds parameters.
    Fills in safe defaults (mu=0.2, sigma=0.15) for invalid or missing entries.

    Args:
        priors_dict: The dictionary of priors parsed from LLM, e.g.,
                     {"paid_search": {"mu": 0.35, "sigma": 0.08}, ...}
        channels: An iterable (e.g., list or dict keys) of expected channel names.

    Returns:
        A validated dictionary with safe defaults applied where necessary.
    """
    validated_priors = {}

    safe_default_mu = 0.2
    safe_default_sigma = 0.15

    for ch in channels:
        if ch not in priors_dict:
            logger.warning(
                f"Channel '{ch}' missing from priors. "
                f"Using safe default (mu={safe_default_mu}, sigma={safe_default_sigma})."
            )
            validated_priors[ch] = {"mu": safe_default_mu, "sigma": safe_default_sigma}
            continue

        ch_data = priors_dict[ch]

        # Validate mu
        mu_raw = ch_data.get("mu")
        is_mu_valid = False
        parsed_mu = None
        try:
            if mu_raw is not None:
                parsed_mu = float(mu_raw)
                if 0 < parsed_mu < 5:
                    is_mu_valid = True
        except (TypeError, ValueError):
            pass

        # Validate sigma
        sigma_raw = ch_data.get("sigma")
        is_sigma_valid = False
        parsed_sigma = None
        try:
            if sigma_raw is not None:
                parsed_sigma = float(sigma_raw)
                if 0.01 < parsed_sigma < 2.0:
                    is_sigma_valid = True
        except (TypeError, ValueError):
            pass

        if not (is_mu_valid and is_sigma_valid):
            if "reasoning" in ch_data:
                logger.warning(
                    f"Discarded reasoning for '{ch}': {ch_data['reasoning']}"
                )

        if is_mu_valid and is_sigma_valid:
            validated_priors[ch] = {"mu": parsed_mu, "sigma": parsed_sigma}
            if "reasoning" in ch_data:
                validated_priors[ch]["reasoning"] = ch_data["reasoning"]
        else:
            logger.warning(
                f"Channel '{ch}' has invalid parameters (mu={mu_raw}, sigma={sigma_raw}). "
                f"Expect bounds 0 < mu < 5 and 0.01 < sigma < 2.0. "
                f"Using safe default (mu={safe_default_mu}, sigma={safe_default_sigma})."
            )
            validated_priors[ch] = {"mu": safe_default_mu, "sigma": safe_default_sigma}

    return validated_priors
