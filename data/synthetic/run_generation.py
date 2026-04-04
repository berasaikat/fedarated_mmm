import os
import yaml
import sys
import pandas as pd
from pathlib import Path

# Add the local directory to the python path to load the generator properly
sys.path.append(str(Path(__file__).resolve().parent))
from generate import generate_participant_data


def load_yaml(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def run(global_config_path: Path | str | None = None):
    # Resolve the base project directory
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_dir = base_dir / "config"
    output_dir = base_dir / "data" / "synthetic"

    # Ensure our output path is ready
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load global configs
    if global_config_path is None:
        global_config_path = config_dir / "global.yaml"
    else:
        global_config_path = Path(global_config_path).resolve()
    if not global_config_path.exists():
        raise FileNotFoundError(
            f"Global configuration mapped to {global_config_path} was not found."
        )

    global_config = load_yaml(global_config_path)
    num_participants = global_config.get("num_participants", 10)
    channels = global_config.get("channels", ["paid_search", "social", "tv", "ooh"])
    seed_base = global_config.get("seed", 42)
    n_weeks = 104  # Assuming 2 years default span of weekly MMM data

    print(
        f"Starting pipeline to generate data for {num_participants} participants...\n"
        + "=" * 50
    )

    for i in range(1, num_participants + 1):
        participant_config_path = config_dir / f"participant_{i}.yaml"
        if not participant_config_path.exists():
            print(f"Warning: Configuration for participant_{i} not found. Skipping.")
            continue

        p_cfg = load_yaml(participant_config_path)
        p_id = p_cfg.get("participant_id", f"participant_{i}")
        seasonality = p_cfg.get("seasonality_pattern", "flat")

        # Vary the random seed to ensure participant data profiles have uniqueness
        p_seed = seed_base * 100 + i

        # Call base synthetic data generator logic
        df = generate_participant_data(
            participant_id=p_id,
            channels=channels,
            n_weeks=n_weeks,
            seed=p_seed,
            seasonality_type=seasonality,
        )

        # Save individual DataFrame as CSV
        out_csv = output_dir / f"{p_id}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Successfully wrote {len(df)} weeks of synthetic data to {out_csv.name}")

        # Print Summary statistics
        mean_rev = df["revenue"].mean()
        mean_channels = {c: df[c].mean() for c in channels if c in df.columns}

        print(f"[Summary Stats for {p_id}]")
        print(f"  > Mean Revenue: {mean_rev:,.2f}")
        for c, m in mean_channels.items():
            print(f"  > Mean Spend ({c}): {m:,.2f}")
        print("-" * 30)


if __name__ == "__main__":
    run()
