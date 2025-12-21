import polars as pl

from mesa_frames import Model


def get_unique_ids(model: Model) -> pl.Series:
    # Collect unique_id across all concrete AgentSets in the registry
    series_list = [aset["unique_id"].cast(pl.UInt64) for aset in model.sets]
    return pl.concat(series_list)
