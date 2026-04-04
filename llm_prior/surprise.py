import math
from typing import Dict, Any


def compute_surprise(
    prior_dict: Dict[str, Dict[str, float]],
    posterior_summary: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Computes the surprise (KL divergence) for each channel from the prior to the posterior.
    Computes standard Bayesian surprise KL(Posterior || Prior) for univariate normal distributions.

    Formula for KL(P || Q) where P is N(mu1, sigma1) and Q is N(mu2, sigma2):
    KL = log(sigma2 / sigma1) + (sigma1^2 + (mu1 - mu2)^2) / (2 * sigma2^2) - 0.5

    Args:
        prior_dict: Dict mapping channel to {"mu": float, "sigma": float}
        posterior_summary: Dict mapping channel to {"posterior_mean": float, "posterior_std": float}

    Returns:
        Dict mapping channel to its computed KL divergence score.
    """
    surprise_scores = {}

    for ch, prior_params in prior_dict.items():
        if ch not in posterior_summary:
            continue

        post_params = posterior_summary[ch]

        mu_prior = prior_params.get("mu")
        sigma_prior = prior_params.get("sigma")

        mu_post = post_params.get("mean")
        sigma_post = post_params.get("std")

        if None in (mu_prior, sigma_prior, mu_post, sigma_post):
            logger.warning(
                f"Skipping surprise for '{ch}' — missing parameters. "
                f"prior={prior_params}, posterior={post_params}"
            )
            continue

        var_post = sigma_post**2
        var_prior = sigma_prior**2

        # Computing KL(Posterior || Prior)
        kl = (
            math.log(sigma_prior / sigma_post)
            + (var_post + (mu_post - mu_prior) ** 2) / (2 * var_prior)
            - 0.5
        )
        kl = max(0.0, kl)
        surprise_scores[ch] = kl

    return surprise_scores


def aggregate_surprise(surprise_dict: Dict[str, float]) -> float:
    """
    Aggregates the surprise scores across channels by computing the mean KL divergence.

    Args:
        surprise_dict: Dict mapping channel to KL score.

    Returns:
        Mean KL score across all channels, or 0.0 if the dictionary is empty.
    """
    if not surprise_dict:
        return 0.0

    total_kl = sum(surprise_dict.values())
    return total_kl / len(surprise_dict)
