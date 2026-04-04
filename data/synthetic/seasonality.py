import numpy as np


def retail_seasonality(n_weeks: int) -> np.ndarray:
    """
    Returns a multiplicative seasonal index of length n_weeks.
    Retail pattern: peaks at weeks 47-52 (Holiday) and 13-14 (Spring).
    """
    seasonality = np.ones(n_weeks, dtype=float)

    for i in range(n_weeks):
        woy = (i % 52) + 1  # 1-indexed week of year
        if (47 <= woy <= 52) or (13 <= woy <= 14):
            seasonality[i] = 1.35  # Peak multiplier

    return seasonality


def b2b_seasonality(n_weeks: int) -> np.ndarray:
    """
    Returns a multiplicative seasonal index of length n_weeks.
    B2B pattern: peaks at Q1 start (weeks 1-4) and Q3 start (weeks 27-30).
    """
    seasonality = np.ones(n_weeks, dtype=float)

    for i in range(n_weeks):
        woy = (i % 52) + 1
        if (1 <= woy <= 4) or (27 <= woy <= 30):
            seasonality[i] = 1.25  # Peak multiplier

    return seasonality


def flat_seasonality(n_weeks: int) -> np.ndarray:
    """
    Returns a flat, all-ones seasonal index of length n_weeks.
    """
    return np.ones(n_weeks, dtype=float)


def uniform_seasonality(n_weeks) -> np.ndarray:
    """Truly flat — no seasonal variation, just ones."""
    return np.ones(n_weeks)


def event_driven_seasonality(n_weeks: int) -> np.ndarray:
    """Sharp spikes at specific weeks simulating product launches or events."""
    index = np.ones(n_weeks)
    # Two major event spikes per year
    event_weeks = [8, 12, 60, 64]  # e.g. trade show seasons
    for w in event_weeks:
        if w < n_weeks:
            index[w] *= 2.5
        if w + 1 < n_weeks:
            index[w + 1] *= 1.8
        if w - 1 >= 0:
            index[w - 1] *= 1.4
    return index


def q4_heavy_seasonality(n_weeks: int) -> np.ndarray:
    index = np.ones(n_weeks)
    for i in range(n_weeks):
        woy = (i % 52) + 1
        if 40 <= woy <= 46:
            index[i] = 1.4  # pre-holiday ramp
        elif 47 <= woy <= 52:
            index[i] = 1.9  # peak holiday
        elif 1 <= woy <= 4:
            index[i] = 1.2  # post-holiday tail
    return index


def summer_peak_seasonality(n_weeks: int) -> np.ndarray:
    index = np.ones(n_weeks)
    for i in range(n_weeks):
        woy = (i % 52) + 1
        if 22 <= woy <= 35:
            index[i] = 1.5
        elif 48 <= woy <= 52 or 1 <= woy <= 4:
            index[i] = 0.7  # winter trough
    return index


def spring_peak_seasonality(n_weeks: int) -> np.ndarray:
    index = np.ones(n_weeks)
    for i in range(n_weeks):
        woy = (i % 52) + 1
        if 10 <= woy <= 20:
            index[i] = 1.5
        elif 40 <= woy <= 52:
            index[i] = 0.8
    return index


def fall_peak_seasonality(n_weeks: int) -> np.ndarray:
    index = np.ones(n_weeks)
    for i in range(n_weeks):
        woy = (i % 52) + 1
        if 35 <= woy <= 44:
            index[i] = 1.5
        elif 15 <= woy <= 28:
            index[i] = 0.85
    return index


def bimodal_seasonality(n_weeks: int) -> np.ndarray:
    """Two equal peaks: early summer and year-end."""
    index = np.ones(n_weeks)
    for i in range(n_weeks):
        woy = (i % 52) + 1
        if 24 <= woy <= 28:
            index[i] = 1.4
        elif 48 <= woy <= 52:
            index[i] = 1.4
        elif 10 <= woy <= 15 or 35 <= woy <= 40:
            index[i] = 0.85
    return index
