from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


def compute_daily_anomalies(
    da: xr.DataArray, time_mean: xr.DataArray = None, smoothed=True
):
    if time_mean is None:
        time_mean = da.mean(dim="time")
    da = da - time_mean
    da_daily = da.resample(time="1D").mean("time").compute()
    if smoothed:
        # TODO: compute daily mean climatologies before smoothing?
        # import pdb

        # pdb.set_trace()
        da_daily_fft = np.fft.rfft(da_daily.values, axis=-3)
        # first 3 harmonics unaltered
        # taper, following NCL smthClmDayTLL
        da_daily_fft[..., 3, :, :] = 0.5 * da_daily_fft[..., 3, :, :]
        da_daily_fft[..., 4:, :, :] = 0.0
        da_daily = xr.DataArray(
            np.fft.irfft(da_daily_fft, n=len(da_daily["time"]), axis=-3),
            coords=da_daily.coords,
            dims=da_daily.dims,
        )
    da_daily = da_daily.groupby("time.dayofyear")
    return da.groupby("time.dayofyear") - da_daily.mean("time")


def remove_rolling_mean(da: xr.DataArray, ndays=120, samples_per_day=4):
    window_len = ndays * samples_per_day
    rollmean = da.rolling(time=window_len, center=False).mean()
    return (da - rollmean).isel(time=slice(window_len, None))


def plot_hovmoller_by_lon(
    da: xr.DataArray,
    var_name: Optional[str] = None,
    figsize=(6, 12),
    time_label="Simulation time",
    labels: Optional[Dict[str, str]] = None,
    **plot_kwargs,
):
    n_col = da.sizes["source"]
    fig, axs = plt.subplots(1, n_col, figsize=figsize, sharey=True)

    if labels is None:
        labels = {"target": "Target", "prediction": "Prediction"}

    assert set(labels.keys()) == set(
        da["source"].values
    ), "labels keys must match da['source']"

    for i, (source, label) in enumerate(labels.items()):
        im = da.sel(source=source).plot(ax=axs[i], add_colorbar=False, **plot_kwargs)
        axs[i].set_title(label)
        axs[i].set_xlabel("Longitude")
        if i > 0:
            axs[i].set_ylabel("")
            axs[i].yaxis.set_tick_params(labelleft=False)

    axs[0].set_ylabel(time_label)

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.13, 0.015, 0.78])

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical",
    )

    if var_name is None:
        var_name = ""
    cbar.set_label(f"Daily mean {var_name}\n[{da.units}]", fontsize="large")

    return fig, axs


def plot_hovmoller_by_lat(
    da: xr.DataArray,
    var_name: Optional[str] = None,
    figsize=(12, 6),
    time_label="Simulation time",
    time_var="time",
):
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axs_ = axs.flatten()
    vmin, vmax = da.min(), da.max()

    da.sel(source="target").plot(
        x=time_var, ax=axs_[0], vmin=vmin, vmax=vmax, add_colorbar=False
    )
    im = da.sel(source="prediction").plot(
        x=time_var, ax=axs_[1], vmin=vmin, vmax=vmax, add_colorbar=False
    )

    axs_[0].set_title("Target")
    axs_[0].set_ylabel("Latitude")
    axs_[0].set_xlabel("")
    axs_[0].xaxis.set_tick_params(labelbottom=False)
    axs_[1].set_title("Generated")
    axs_[1].set_ylabel("Latitude")
    axs_[1].set_xlabel(time_label)

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.13, 0.015, 0.78])

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical",
    )

    if var_name is None:
        var_name = ""
    cbar.set_label(f"Daily mean {var_name}\n[{da.units}]", fontsize="large")

    return fig, axs
