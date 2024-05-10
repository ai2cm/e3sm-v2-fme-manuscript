import os
from typing import List, Mapping, Optional

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
    base_dir = run_config.get("base_dir", config["dataset_dir"])
    if "slurm_inference_run_id" in run_config:
        run_dir = os.path.join(base_dir, run_config["slurm_inference_run_id"])
    else:
        dset_name = run_config["dataset"]
        group = run_config["group"]
        name = run_config["name"]
        run_dir = os.path.join(base_dir, dset_name, "output", group, name)
    return os.path.join(run_dir, "autoregressive_predictions.nc")


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
    run_label: str,
    apply_funcs: Optional[dict] = None,
    run_label_as_column: bool = False,
):
    keys = {f"{keys_stem}/{name}": label for name, label in vars_dict.items()}
    metric_names = list(keys.keys())
    samples = 1 if run_label_as_column else 500
    df = wandb_run.history(keys=metric_names, samples=samples)
    df = df.rename(dict(zip(metric_names, vars_dict.keys())), axis=1)
    if run_label_as_column:
        df = df.drop("_step", axis=1)
    else:
        df = df.set_index("_step")
    if apply_funcs is not None:
        for name, func in apply_funcs.items():
            df[name] = df[name].apply(func)
    if run_label_as_column:
        df.index = [run_label]
        return df.T.join(
            pd.DataFrame.from_dict(vars_dict, orient="index", columns=["label"])
        )
    new_index = pd.MultiIndex.from_product(
        [[run_label], df.index], names=["run", df.index.name]
    )
    return df.set_index(new_index)


def melt_training_steps(
    df: pd.DataFrame,
    id_vars: Optional[List[str]] = None,
):
    """
    df: pd.DataFrame
        A wide format DataFrame, e.g. the output of
        load_global_time_mean_metrics when inference=False.
    id_vars: Optional[List[str]]
        Passed to pd.melt.
    """
    if id_vars is None:
        id_vars = list(df.index.names)
    df_long = pd.melt(df.reset_index(), id_vars=id_vars)
    return df_long.set_index(id_vars + ["variable"])
