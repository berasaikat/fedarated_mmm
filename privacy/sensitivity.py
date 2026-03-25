import math
import copy
from typing import Dict, Any

def compute_l2_sensitivity(channel_count: int) -> float:
    """
    Computes a conservative L2 sensitivity estimate for the posterior mean vector.
    
    Args:
        channel_count: the number of marketing channels
        
    Returns:
        float: L2 sensitivity = sqrt(channel_count) * 0.5
    """
    return math.sqrt(channel_count) * 0.5


# NOTE: Only channel means are clipped and noised.
# Diagnostic fields (p5, p95, r_hat, eff_sample_size) are passed through
# as they do not directly reveal sensitive spend/revenue values.
def clip_posterior(
    posterior_summary: Dict[str, Dict[str, float]], 
    clip_norm: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """
    L2-clips the beta means vector in the posterior summary to the boundary clip_norm.
    Limits the contribution norm of any single participant's update before noise addition.
    
    Args:
        posterior_summary: dict mapping channel to e.g. {"mean": float, "std": float}
        clip_norm: float representing maximum permitted L2 norm of the means vector.
        
    Returns:
        dict: A deepcopy of posterior_summary with the channel means scaled down if their 
              combined norm exceeds clip_norm.
    """
    means = []
    channels = []
    
    for ch, params in posterior_summary.items():
        if "mean" in params:
            means.append(params["mean"])
            channels.append(ch)
            
    if not means:
        return copy.deepcopy(posterior_summary)
        
    # Calculate the L2 norm of the extracted means vector
    l2_squared = sum(m * m for m in means)
    l2_norm = math.sqrt(l2_squared)
    
    clipped_summary = copy.deepcopy(posterior_summary)
    
    if l2_norm > clip_norm and l2_norm > 0:
        # Scale factor to project back to the bounding hyper-sphere 
        scale_factor = clip_norm / l2_norm
        
        for ch in channels:
            clipped_summary[ch]["mean"] *= scale_factor
            
    return clipped_summary
