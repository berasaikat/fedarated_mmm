import jax.random as random
import numpy as np
from numpyro.infer import MCMC, NUTS

def run_mcmc(
    model, 
    spend_matrix, 
    revenue, 
    priors_dict, 
    num_warmup=500, 
    num_samples=1000, 
    num_chains=2,
    channel_names=None,
    seed: int = 0
) -> dict:
    """
    Runs MCMC inference using the NUTS sampler on the provided NumPyro model.
    
    Returns:
        A dictionary containing:
        - "samples": The raw posterior samples dict keyed by parameter name.
        - "summary": A dictionary of computed summary statistics (mean, std, 5%, 95%) 
                     for each beta coefficient parameter.
    """
    # Initialize PRNGKey for JAX
    rng_key = random.PRNGKey(seed)
    rng_key, run_key = random.split(rng_key)
    
    # Setup MCMC with NUTS kernel
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )
    
    # Run the graphical model
    mcmc.run(
        run_key, 
        spend_matrix=spend_matrix, 
        revenue=revenue, 
        priors_dict=priors_dict
    )
    
    # Retrieve raw samples dict
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    
    # Compute summary stats specifically for all tracked beta_c variables
    # summary = {}
    # for param_name, samples in posterior_samples.items():
    #     if param_name.startswith("beta_"):
    #         # Ensure numpy is used to drop out of jax tracer arrays if needed
    #         samples_np = np.asarray(samples)
    #         summary[param_name] = {
    #             "mean": float(np.mean(samples_np)),
    #             "std": float(np.std(samples_np)),
    #             "5th_percentile": float(np.percentile(samples_np, 5.0)),
    #             "95th_percentile": float(np.percentile(samples_np, 95.0))
    #         }
            
    return posterior_samples
