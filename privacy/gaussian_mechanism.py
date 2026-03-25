import math
import numpy as np
from typing import Dict, Any
import copy

def add_gaussian_noise(
    posterior_summary: Dict[str, Dict[str, float]], 
    sensitivity: float, 
    epsilon: float, 
    delta: float,
    seed = None
) -> Dict[str, Dict[str, float]]:
    """
    Applies the Gaussian mechanism to posterior summaries to ensure differential privacy.
    
    Computes noise_std = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    Adds Gaussian noise N(0, noise_std) to the 'mean' and 'std' parameters for each channel.
    
    Args:
        posterior_summary: dict mapping channel to e.g. {"mean": float, "std": float}
        sensitivity: the global sensitivity bound
        epsilon: the privacy budget parameter
        delta: the differential privacy failure probability parameter
        
    Returns:
        A new noisy summary dictionary with the same schema as the input.
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError("Epsilon and delta must be strictly positive.")
    
    # Calculate scale of the noise per the Gaussian mechanism definition
    rng = np.random.default_rng(seed)
    noise_std = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    # Deepcopy to avoid mutating the original dictionary during the differential privacy pass
    noisy_summary = copy.deepcopy(posterior_summary)
    
    for ch, params in noisy_summary.items():
        if "mean" in params:
            params["mean"] += rng.normal(0, noise_std)
            
    return noisy_summary
