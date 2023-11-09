from typing import List, Optional

import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


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
    vmax = max(abs(vmin), abs(vmax))

    baseline_time_mean_bias.plot(
        ax=axs_[0], norm=TwoSlopeNorm(0.0, -vmax, vmax), cmap="bwr", add_colorbar=False
    )
    im = time_mean_bias.plot(
        ax=axs_[1], norm=TwoSlopeNorm(0.0, -vmax, vmax), cmap="bwr", add_colorbar=False
    )

    axs_[0].set_title("Reference")
    axs_[0].set_xlabel("Longitude")
    axs_[0].set_ylabel("Latitude")
    axs_[1].set_title("Generated")
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


def plot_time_mean_bias_list(
    time_mean_bias: List[xr.DataArray],
    baseline_time_mean_bias: List[xr.DataArray],
    var_names: Optional[List[str]] = None,
    figsize=(20, 11),
):
    # assumes that all DataArrays have the same units and uses a shared colorbar

    fig, axs = plt.subplots(len(time_mean_bias), 2, figsize=figsize)

    vmin = min([bias.min().item() for bias in time_mean_bias])
    vmax = max([bias.max().item() for bias in time_mean_bias])
    vmax = max(abs(vmin), abs(vmax))

    for i, bias in enumerate(time_mean_bias):
        baseline_time_mean_bias[i].plot(
            ax=axs[i][0],
            norm=TwoSlopeNorm(0.0, -vmax, vmax),
            cmap="bwr",
            add_colorbar=False,
        )
        im = bias.plot(
            ax=axs[i][1],
            norm=TwoSlopeNorm(0.0, -vmax, vmax),
            cmap="bwr",
            add_colorbar=False,
        )

        if i == 0:
            axs[i][0].set_title("Reference", fontsize="x-large")
            axs[i][1].set_title("Generated", fontsize="x-large")
        else:
            axs[i][0].set_title("")
            axs[i][1].set_title("")

        if i < len(time_mean_bias) - 1:
            axs[i][0].set_xlabel("")
            axs[i][1].set_xlabel("")
            axs[i][0].xaxis.set_tick_params(length=0, labelbottom=False)
            axs[i][1].xaxis.set_tick_params(length=0, labelbottom=False)

        if i == len(time_mean_bias) - 1:
            axs[i][0].set_xlabel("Longitude", fontsize="large")
            axs[i][1].set_xlabel("Longitude", fontsize="large")

        axs[i][0].set_ylabel("Latitude", fontsize="large")
        axs[i][1].set_ylabel("")
        axs[i][1].yaxis.set_tick_params(length=0, labelleft=False)
        if var_names is not None:
            axs[i][0].text(
                x=-40,
                y=0.5,
                s=var_names[i],
                rotation="vertical",
                verticalalignment="center",
                fontsize="x-large",
            )

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.01, 0.8])
    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.set_label(f"Time-mean bias\n[{time_mean_bias[0].units}]", fontsize="x-large")

    return fig, axs
