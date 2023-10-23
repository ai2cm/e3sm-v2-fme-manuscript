from typing import Optional

import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt


def compute_time_mean_bias(da_pred: xr.DataArray, da_tar: xr.DataArray):
    bias = da_pred - da_tar
    with ProgressBar():
        time_mean_bias = bias.mean("time").compute()
    return time_mean_bias


def plot_time_mean_bias(
    time_mean_bias: xr.DataArray,
    baseline_time_mean_bias: xr.DataArray,
    var_name: Optional[str] = None,
    figsize=(20, 6),
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs_ = axs.flatten()

    vmin, vmax = time_mean_bias.min().item(), time_mean_bias.max().item()

    time_mean_bias.plot(ax=axs_[0], vmin=vmin, vmax=vmax, add_colorbar=False)
    im = baseline_time_mean_bias.plot(
        ax=axs_[1], vmin=vmin, vmax=vmax, add_colorbar=False
    )

    axs_[0].set_title("Generated")
    axs_[0].set_xlabel("Longitude")
    axs_[0].set_ylabel("Latitude")
    axs_[1].set_title("Baseline")
    axs_[1].set_xlabel("Longitude")
    axs_[1].set_ylabel("")
    axs_[1].yaxis.set_tick_params(labelleft=False)

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.01, 0.8])
    if var_name is None:
        var_name = ""
    else:
        var_name = var_name + " "
    fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical",
        label=f"Time-mean {var_name}bias\n[{time_mean_bias.units}]",
    )

    return fig, axs
