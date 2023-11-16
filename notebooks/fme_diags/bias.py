from typing import List, Optional

import cartopy.crs as ccrs
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
    vmax_abs=None,
    verbose=False,
):
    fig, axs = plt.subplots(
        1,
        2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axs_ = axs.flatten()

    if vmax_abs is None:
        vmin, vmax = time_mean_bias.min().item(), time_mean_bias.max().item()
        if verbose:
            print(f"Time-mean bias minimum: {vmin:0.4f}")
            print(f"Time-mean bias maximum: {vmax:0.4f}")
        vmax_abs = max(abs(vmin), abs(vmax))

    im = baseline_time_mean_bias.plot(
        ax=axs_[0],
        norm=TwoSlopeNorm(0.0, -vmax_abs, vmax_abs),
        cmap="RdBu_r",
        add_colorbar=False,
    )
    time_mean_bias.plot(
        ax=axs_[1],
        norm=TwoSlopeNorm(0.0, -vmax_abs, vmax_abs),
        cmap="RdBu_r",
        add_colorbar=False,
    )

    for ax in axs_:
        ax.coastlines(linewidth=0.5, color="grey", alpha=0.5)

    axs_[0].set_title("Reference", fontsize="x-large")
    axs_[0].set_xlabel("Longitude", fontsize="large")
    axs_[0].set_ylabel("Latitude", fontsize="large")
    axs_[1].set_title("Generated", fontsize="x-large")
    axs_[1].set_xlabel("Longitude", fontsize="large")
    axs_[1].set_ylabel("")
    axs_[1].yaxis.set_tick_params(labelleft=False)

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.84])

    if var_name is None:
        var_name = ""
    else:
        var_name = var_name + " "

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.set_label(
        f"Time-mean {var_name}bias\n[{time_mean_bias.units}]", fontsize="large"
    )

    return fig, axs


def plot_time_mean_bias_list(
    time_mean_bias: List[xr.DataArray],
    baseline_time_mean_bias: List[xr.DataArray],
    var_names: Optional[List[str]] = None,
    figsize=(20, 11),
    vmax_abs=None,
    axs=None,
    verbose=False,
):
    # assumes that all DataArrays have the same units and uses a shared colorbar
    if axs is None:
        fig, axs = plt.subplots(
            len(time_mean_bias),
            2,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = None
        assert len(axs) == len(time_mean_bias)
        assert len(axs[0]) == 2

    if vmax_abs is None:
        vmins = [bias.min().item() for bias in time_mean_bias]
        vmaxs = [bias.max().item() for bias in time_mean_bias]
        vmax_abs = max(abs(min(vmins)), abs(max(vmaxs)))
    else:
        verbose = False

    for i, bias in enumerate(time_mean_bias):
        if verbose:
            prefix = (
                f"time_mean_bias[i]"
                if var_names is None
                else f"Time-mean {var_names[i]} bias"
            )
            print(f"{prefix} minimum: {vmins[i]:0.4f}")
            print(f"{prefix} maximum: {vmaxs[i]:0.4f}")
        baseline_time_mean_bias[i].plot(
            ax=axs[i][0],
            norm=TwoSlopeNorm(0.0, -vmax_abs, vmax_abs),
            cmap="RdBu_r",
            add_colorbar=False,
        )
        im = bias.plot(
            ax=axs[i][1],
            norm=TwoSlopeNorm(0.0, -vmax_abs, vmax_abs),
            cmap="RdBu_r",
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
                x=-0.05,
                y=0.5,
                s=var_names[i],
                transform=axs[i][0].transAxes,
                rotation="vertical",
                verticalalignment="center",
                fontsize="x-large",
            )

    for ax in axs.flatten():
        ax.coastlines(linewidth=0.5, color="grey", alpha=0.5)

    if fig is not None:
        plt.tight_layout()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.84])
        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label(
            f"Time-mean bias\n[{time_mean_bias[0].units}]", fontsize="x-large"
        )
        return fig, axs

    return im
