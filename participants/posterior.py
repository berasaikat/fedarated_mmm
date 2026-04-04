import json
import numpy as np
import numpyro.diagnostics as diag


def extract_posterior_summary(mcmc_samples: dict) -> dict:
    """
    Extracts structured summary statistics for each beta coefficient.
    For accurate effective sample size and r_hat calculation, mcmc_samples must
    retain the chain dimension (e.g., from `mcmc.get_samples(group_by_chain=True)`).
    """
    # Use NumPyro's built-in summary statistics generator.
    # prob=0.90 computes the bounds enclosing 90% of the distribution (p5, p95).
    full_summary = diag.summary(mcmc_samples, prob=0.90)

    extracted = {}
    for param_name, stats in full_summary.items():
        if param_name.startswith("beta_"):
            # Ensure float casting for clean JSON serializability
            extracted[param_name] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "p5": float(stats["5.0%"]),
                "p95": float(stats["95.0%"]),
                "effective_sample_size": float(stats["n_eff"]),
                "r_hat": float(stats["r_hat"]),
            }

    return extracted


def serialize_posterior_summary(summary_dict: dict) -> str:
    """
    Serializes the nested summary dictionary to a JSON string.
    """
    return json.dumps(summary_dict, indent=2)


def deserialize_posterior_summary(json_str: str) -> dict:
    """
    Deserializes a JSON string back into the nested summary dictionary.
    """
    return json.loads(json_str)
