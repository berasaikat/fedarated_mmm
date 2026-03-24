import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent))
from seasonality import (
    retail_seasonality,
    b2b_seasonality,
    flat_seasonality,
    uniform_seasonality,
    event_driven_seasonality, 
    q4_heavy_seasonality,
    summer_peak_seasonality,
    spring_peak_seasonality,
    fall_peak_seasonality,
    bimodal_seasonality,
)

def apply_adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """Applies simple geometric adstock transformation."""
    adstocked = np.zeros_like(spend)
    if len(spend) > 0:
        adstocked[0] = spend[0]
        for t in range(1, len(spend)):
            adstocked[t] = spend[t] + alpha * adstocked[t - 1]
    return adstocked

def apply_hill_saturation(x: np.ndarray, ec: float, slope: float) -> np.ndarray:
    """Applies Hill function for saturation."""
    return (x ** slope) / (ec ** slope + x ** slope + 1e-9)

def generate_participant_data(
    participant_id: str, 
    channels: List[str], 
    n_weeks: int, 
    seed: int, 
    seasonality_type: str
) -> pd.DataFrame:
    """
    Generates synthetic participant MMM data.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate basis for week
    week = np.arange(1, n_weeks + 1)
    
    # 2. Seasonality — delegate to seasonality.py

    seasonality_map = {
        "retail":       retail_seasonality,
        "b2b_cycle":    b2b_seasonality,
        "flat":         flat_seasonality,
        "uniform":      uniform_seasonality,
        "event_driven": event_driven_seasonality,
        "q4_heavy":     q4_heavy_seasonality,
        "summer_peak":  summer_peak_seasonality,
        "spring_peak":  spring_peak_seasonality,
        "fall_peak":    fall_peak_seasonality,
        "bimodal":      bimodal_seasonality,
    }

    pattern_fn = seasonality_map.get(seasonality_type)
    if pattern_fn is None:
        raise ValueError(
            f"Unknown seasonality_type: '{seasonality_type}'. "
            f"Valid options: {list(seasonality_map.keys())}"
        )

    seasonality = pattern_fn(n_weeks)
    
    df = pd.DataFrame({"week": week})
    
    # Base revenue before ad spend
    revenue_base = 1000.0 * seasonality
    revenue_total = revenue_base.copy()
    
    # Create spend and revenue per channel
    for channel in channels:
        # Autoregressive spend logic to make spend realistic
        spend_mean = rng.uniform(100, 500)
        spend = np.maximum(0, rng.normal(spend_mean, spend_mean * 0.2, size=n_weeks))
        
        # Apply seasonality to spend as well so spend follows peak
        spend = spend * seasonality
        df[channel] = spend
        
        # Ground truth parameters
        alpha = rng.uniform(0.1, 0.7)  # adstock decay
        ec = rng.uniform(spend_mean * 0.5, spend_mean * 1.5)  # half-saturation point
        slope = rng.uniform(1.0, 3.0)  # hill slope
        roi_multiplier = rng.uniform(1.5, 3.5) # simple multiplier to convert saturated spend to revenue
        
        # Apply adstock and saturation
        x_adstocked = apply_adstock(spend, alpha)
        x_saturated = apply_hill_saturation(x_adstocked, ec, slope)
        
        revenue_contrib = x_saturated * (spend_mean * roi_multiplier)
        revenue_total += revenue_contrib
        
    # Add Gaussian noise
    noise_std = np.maximum(1.0, revenue_total.mean() * 0.05)
    noise = rng.normal(0, noise_std, size=n_weeks)
    revenue_total += noise
    
    # Ensure no negative revenue
    df["revenue"] = np.maximum(0, revenue_total)
    
    return df
