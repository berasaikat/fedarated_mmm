import numpy as np


def adstock(spend_array: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Implements geometric adstock decay:
    adstock[t] = spend[t] + decay_rate * adstock[t-1]

    Expected decay_rate range per-channel is typically 0.3 to 0.9.
    """
    adstocked = np.zeros_like(spend_array, dtype=float)
    if len(spend_array) > 0:
        adstocked[0] = spend_array[0]
        for t in range(1, len(spend_array)):
            adstocked[t] = spend_array[t] + decay_rate * adstocked[t - 1]
    return adstocked


def hill_saturation(x: np.ndarray, K: float, n: float) -> np.ndarray:
    """
    Implements the Hill equation: x^n / (K^n + x^n).

    A small epsilon (1e-9) is added to the denominator to prevent division by zero gracefully.
    """
    x_safe = np.maximum(0, x)  # ensures no negative bases
    return (x_safe**n) / (K**n + x_safe**n + 1e-9)
