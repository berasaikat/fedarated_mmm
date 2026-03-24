import yaml
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
import sys

# Add the local directory to the python path to load the generator properly
sys.path.append(str(Path(__file__).resolve().parent))
from mmm_model import mmm_numpyro
from inference import run_mcmc
from posterior import extract_posterior_summary

class LocalTrainer:
    def __init__(self, participant_id: str, config_path: str):
        self.participant_id = participant_id
        self.config_path = Path(config_path)
        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Identify the precise base folder by resolving config location
        base_dir = self.config_path.resolve().parent.parent
        relative_data_path = self.config.get("data_path", f"data/synthetic/{participant_id}.csv")
        self.csv_path = base_dir / relative_data_path

    def load_data(self):
        """
        Reads CSV and separates spend columns from revenue column.
        Returns: spend_matrix (np.ndarray), revenue (np.ndarray), and the list of spend column names.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Participant {self.participant_id} data not found at: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        revenue = df["revenue"].values
        
        # Exclude reserved tracking columns to isolate dynamic channel features
        drop_cols = ["week", "revenue"]
        spend_cols = [c for c in df.columns if c not in drop_cols]
        spend_matrix = df[spend_cols].values
        
        return spend_matrix, revenue, spend_cols

    def train(self, priors_dict: dict) -> dict:
        """
        Loads local synthetic data, executes NumPyro MCMC Bayesian inference, 
        extracts the chain logic reliably, and returns the summarized posterior dict.
        """
        spend_matrix, revenue, spend_cols = self.load_data()
        
        # Safely convert pure numpy objects/float arrays over to JAX primitives for optimization
        spend_jax = jnp.array(spend_matrix)
        revenue_jax = jnp.array(revenue)
        
        print(f"[{self.participant_id}] Starting Local Training (NUTS MCMC)..")
        
        p_seed = hash(self.participant_id) % (2**31)
        # Run NumPyro MCMC (defaults to 500 warmup, 1000 samples, 2 chains)
        mcmc_results = run_mcmc(
            model=mmm_numpyro,
            spend_matrix=spend_jax,
            revenue=revenue_jax,
            priors_dict=priors_dict,
            channel_names=spend_cols,
            seed=p_seed
        )
        
        print(f"[{self.participant_id}] Extracting Posterior Summary..")
        # Ensure extraction has proper shape dimensions
        mcmc_samples = mcmc_results
        summary = extract_posterior_summary(mcmc_samples)

        # Warn if any chain didn't converge
        for param, stats in summary.items():
            if stats["r_hat"] > 1.05:
                print(f"WARNING: {param} r_hat={stats['r_hat']:.3f} — chain may not have converged")
        
        return summary
