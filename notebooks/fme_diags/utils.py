import os
from typing import Mapping, Optional

import numpy as np
import xarray as xr
import yaml


def load_config(path: os.PathLike):
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


def open_autoregressive_inference(
    path: os.PathLike,
    start: str,
    step_dim="timestep",
    step_freq="6H",
    calendar="noleap",
    flip_lat: bool = False,
    chunks: Optional[Mapping[str, int]] = None,
) -> xr.Dataset:
    """Open fme output "autoregressive_predictions.nc" file and create a
    CFTimeRange coordinate "time" coordinate.

    Parameters
    ----------
    path: os.PathLike
        Path to an autoregressive_predictions.nc file.
    start: str
        The xr.cftime_range-compatible initial condition time.
    step_dim: str
        The name of the AR step dimension in the dataset at "path".
    step_freq: str
        A CFTime-compatible timestep frequency string.
    calendar: str
        A CFTime-compatible calendar name.
    flip_lat: bool
        If True, reverse the latitude dimension.
    chunks: Mapping[str, int]
        Passed to xr.open_dataset.
    """
    xr.set_options(keep_attrs=True)
    ds = xr.open_dataset(path, chunks=chunks)
    time_coords = xr.cftime_range(
        start=start, freq=step_freq, periods=len(ds[step_dim]), calendar=calendar
    )
    ds = (
        ds.rename({step_dim: "time"})
        .assign_coords(
            {
                "time": ("time", time_coords),
                "lat": (
                    "lat",
                    np.flip(ds["lat"].values) if flip_lat else ds["lat"].values,
                ),
            }
        )
        .isel(lat=slice(None, None, -1))
    )
    return ds
