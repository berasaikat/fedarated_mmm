import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_geo_data(participant_data_path: str, geo_metadata_path: str) -> pd.DataFrame:
    """
    Loads and merges participant weekly spend/revenue data with geographic market metadata.

    Expected Participant Data Columns: week, geo_id, revenue, <spend_channels...>
    Expected Geo Metadata Columns: geo_id, region, population, median_income, urbanization_level

    Args:
        participant_data_path: Path to the CSV containing weekly geo-level data.
        geo_metadata_path: Path to the CSV containing cross-sectional geographic metadata.

    Returns:
        pd.DataFrame: A mathematically tidy DataFrame merging the series with the metadata on 'geo_id'.
    """
    logger.info(f"Loading participant structured geo data from {participant_data_path}")
    df_data = pd.read_csv(participant_data_path)

    logger.info(f"Loading categorical geo metadata from {geo_metadata_path}")
    df_meta = pd.read_csv(geo_metadata_path)

    if "geo_id" not in df_data.columns:
        logger.warning("No 'geo_id' column found — assigning default 'geo_0'")
        df_data["geo_id"] = "geo_0"

    if "geo_id" not in df_meta.columns:
        raise ValueError(f"Missing 'geo_id' in metadata: {geo_metadata_path}")

    return pd.merge(df_data, df_meta, on="geo_id", how="left")
