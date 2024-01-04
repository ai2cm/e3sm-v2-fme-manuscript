import os
from typing import Mapping, Optional

import dask
import numpy as np
import pandas as pd
import xarray as xr
import yaml


def load_config(path: os.PathLike):
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


def get_run_path(config: dict, key: str):
    run_config = config["runs"][key]
    base_dir = run_config.get("dataset_dir", config["dataset_dir"])
    dset_name = run_config["dataset"]
    group = run_config["group"]
    name = run_config["name"]
    return os.path.join(
        base_dir, dset_name, "output", group, name, "autoregressive_predictions.nc"
    )


def get_run_kwargs(config: dict, key: str):
    run_config = config["runs"][key]
    return {
        "path": get_run_path(config, key),
        "start": run_config.get("start", config["start"]),
        "step_dim": run_config.get("step_dim", config["step_dim"]),
        "step_freq": run_config.get(config["step_freq"], config["step_freq"]),
        "calendar": run_config.get("calendar", config["calendar"]),
        "flip_lat": run_config.get("flip_lat", config["flip_lat"]),
        "chunks": run_config.get("chunks", config["chunks"]),
    }


def get_wandb_path(config: dict, key: str, inference=True):
    run_config = config["runs"][key]
    entity = run_config.get("wandb_entity", config["wandb_entity"])
    project = run_config.get("wandb_project", config["wandb_project"])
    if inference:
        run_id = run_config["inference_run_id"]
    else:
        run_id = run_config["training_run_id"]
    return os.path.join(entity, project, run_id)


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
    ds = ds.rename({step_dim: "time"}).assign_coords(
        {
            "time": ("time", time_coords),
            "lat": (
                "lat",
                np.flip(ds["lat"].values) if flip_lat else ds["lat"].values,
            ),
        }
    )
    if flip_lat:
        ds = ds.isel(lat=slice(None, None, -1))

    return ds


def open_reference(
    path: os.PathLike,
    start: str,
    lat_coords: xr.DataArray,
    step_dim="time",
    step_freq="6H",
    calendar="noleap",
):
    xr.set_options(keep_attrs=True)
    ds = xr.open_mfdataset(path)
    time_coords = xr.cftime_range(
        start=start, freq=step_freq, periods=len(ds[step_dim]), calendar=calendar
    )
    ds["time"] = time_coords
    # with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    #     ds = ds.reindex(lat=list(reversed(ds["lat"])))
    ds["lat"] = lat_coords
    return ds


def load_global_time_mean_metrics(
    wandb_run,
    keys_stem: str,
    vars_dict: dict,
    run_label: Optional[str] = None,
    apply_funcs: Optional[dict] = None,
):
    keys = {f"{keys_stem}/{name}": label for name, label in vars_dict.items()}
    metric_names = list(keys.keys())
    metrics = (
        wandb_run.history(keys=metric_names, samples=1)
        .rename(dict(zip(metric_names, vars_dict.keys())), axis=1)
        .drop("_step", axis=1)
    )
    if apply_funcs is not None:
        for name, func in apply_funcs.items():
            metrics[name] = metrics[name].apply(func)
    if run_label is not None:
        metrics.index = [run_label]
    return metrics.T.join(
        pd.DataFrame.from_dict(vars_dict, orient="index", columns=["label"])
    )
