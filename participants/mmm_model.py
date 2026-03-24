import jax.numpy as jnp
from jax import lax
import numpyro
from numpyro import distributions as dist

def _jax_adstock(spend_channel, alpha):
    """
    JAX-compatible single-channel geometric adstock.
    Iteratively accumulates the decay using `lax.scan` to be auto-diff friendly in NumPyro.
    """
    def body_fn(carry, x):
        y = x + alpha * carry
        return y, y
    _, adstocked = lax.scan(body_fn, init=0.0, xs=spend_channel)
    return adstocked

def _jax_hill(x, ec, slope):
    """
    JAX-compatible single-channel Hill saturation.
    """
    x_safe = jnp.maximum(0.0, x)  # Prevent negative bases
    return (x_safe ** slope) / (ec ** slope + x_safe ** slope + 1e-9)

def mmm_numpyro(spend_matrix, revenue=None, priors_dict=None, channel_names=None):
    """
    Core Bayesian Media Mix Model in NumPyro.
    
    Args:
        spend_matrix: jnp.ndarray shape (T, C). Columns represent channel spend over T weeks.
        revenue: jnp.ndarray shape (T,) for observed ground-truth target.
        priors_dict: dictionary mapping channel indices (0..C-1) 
                     to their hierarchical priors {'mu': float, 'sigma': float}.
    """
    if priors_dict is None:
        priors_dict = {}
        
    T, C = spend_matrix.shape
    
    # We will accumulate expected revenue here across channels
    mu = jnp.zeros(T)
    
    # Hardcoded fixed parameters for Adstock/Saturation as requested
    alpha = 0.5
    slope = 1.5
    
    for c in range(C):
        spend_c = spend_matrix[:, c]
        ec = jnp.maximum(jnp.mean(spend_c), 1.0)
        
        key = channel_names[c] if channel_names else c
        prior = (priors_dict or {}).get(key, {})
        c_mu = prior.get("mu", 0.0)
        c_sigma = prior.get("sigma", 1.0)
        
        # Sample beta coefficient for this specific channel
        beta_c = numpyro.sample(f"beta_{c}", dist.Normal(c_mu, c_sigma))
        
        # Extract 1D array of specific channel's spend
        spend_c = spend_matrix[:, c]
        
        # Apply adstock then Hill saturation (fixed params)
        adstocked_c = _jax_adstock(spend_c, alpha)
        transformed_spend_c = _jax_hill(adstocked_c, ec, slope)
        
        # Compute mu = sum(beta_c * transformed_spend_c)
        mu = mu + beta_c * transformed_spend_c
        
    revenue_scale = jnp.std(revenue) if revenue is not None else 1000.0
    sigma = numpyro.sample("sigma", dist.HalfNormal(revenue_scale * 0.5))
    
    # Likelihood function: sample revenue from Normal(mu, sigma)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=revenue)
