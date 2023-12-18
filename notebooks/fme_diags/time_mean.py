from typing import List, Optional

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def compute_time_mean_bias(da_pred: xr.DataArray, da_tar: xr.DataArray):
    bias = da_pred - da_tar
    with ProgressBar():
        time_mean_bias = bias.mean("time").compute()
    return time_mean_bias


def compute_time_mean_rmse(da_pred: xr.DataArray, da_tar: xr.DataArray):
    rmse = np.sqrt((da_pred - da_tar) ** 2)
    with ProgressBar():
        time_mean_rmse = rmse.mean("time").compute()
    return time_mean_rmse


def plot_time_mean(
    time_mean: xr.DataArray,
    baseline_time_mean: xr.DataArray,
    var_name: Optional[str] = None,
    metric_name: Optional[str] = None,
    figsize=(20, 6),
    axs=None,
    **plot_kwargs,
):
    if axs is None:
        fig, axs = plt.subplots(
            1,
            2,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = None
        assert len(axs) == 2

    axs_ = axs.flatten()

    if metric_name is None:
        metric_name = ""
    else:
        metric_name = metric_name + " "

    if var_name is None:
        var_name = ""
    else:
        var_name = var_name + " "

    im = baseline_time_mean.plot(
        ax=axs_[0],
        add_colorbar=False,
        **plot_kwargs,
    )
    time_mean.plot(
        ax=axs_[1],
        add_colorbar=False,
        **plot_kwargs,
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

    if fig is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.84])

        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.set_label(
            f"Time-mean {var_name}{metric_name}\n[{time_mean.units}]",
            fontsize="x-large",
        )

    return fig, axs


def plot_time_mean_bias(
    time_mean_bias: xr.DataArray,
    baseline_time_mean_bias: xr.DataArray,
    var_name: Optional[str] = None,
    figsize=(20, 6),
    vmax_abs=None,
    verbose=False,
):
    vmin, vmax = time_mean_bias.min().item(), time_mean_bias.max().item()
    vmax_abs = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(0.0, -vmax_abs, vmax_abs)
    cmap = "RdBu_r"
    if verbose:
        print(f"Time-mean bias minimum: {vmin:0.8f}")
        print(f"Time-mean bias maximum: {vmax:0.8f}")
    fig, axs = plot_time_mean(
        time_mean_bias,
        baseline_time_mean_bias,
        var_name=var_name,
        metric_name="bias",
        figsize=figsize,
        verbose=verbose,
        norm=norm,
        cmap=cmap,
    )
    return fig, axs


def plot_time_mean_rmse(
    time_mean_rmse: xr.DataArray,
    baseline_time_mean_rmse: xr.DataArray,
    var_name: Optional[str] = None,
    figsize=(20, 6),
    verbose=False,
):
    vmin, vmax = time_mean_rmse.min().item(), time_mean_rmse.max().item()
    cmap = "inferno"
    if verbose:
        print(f"Time-mean bias minimum: {vmin:0.8f}")
        print(f"Time-mean bias maximum: {vmax:0.8f}")
    fig, axs = plot_time_mean(
        time_mean_rmse,
        baseline_time_mean_rmse,
        var_name=var_name,
        metric_name="rmse",
        figsize=figsize,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    return fig, axs


def plot_time_mean_list(
    time_mean: List[xr.DataArray],
    baseline_time_mean: List[xr.DataArray],
    var_names: Optional[List[str]] = None,
    metric_name: Optional[str] = None,
    figsize=(20, 11),
    vmax_abs=None,
    axs=None,
    verbose=False,
):
    # assumes that all DataArrays have the same units and uses a shared colorbar
    if axs is None:
        fig, axs = plt.subplots(
            len(time_mean),
            2,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = None
        assert len(axs) == len(time_mean)
        assert len(axs[0]) == 2

    if vmax_abs is None:
        vmins = [bias.min().item() for bias in time_mean]
        vmaxs = [bias.max().item() for bias in time_mean]
        vmax_abs = max(abs(min(vmins)), abs(max(vmaxs)))
    else:
        verbose = False

    if metric_name is None:
        metric_name = ""

    for i, bias in enumerate(time_mean):
        if verbose:
            prefix = (
                f"time_mean[i] {metric_name}"
                if var_names is None
                else f"Time-mean {var_names[i]} {metric_name}"
            )
            print(f"{prefix} minimum: {vmins[i]:0.8f}")
            print(f"{prefix} maximum: {vmaxs[i]:0.8f}")
        baseline_time_mean[i].plot(
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

        if i < len(time_mean) - 1:
            axs[i][0].set_xlabel("")
            axs[i][1].set_xlabel("")
            axs[i][0].xaxis.set_tick_params(length=0, labelbottom=False)
            axs[i][1].xaxis.set_tick_params(length=0, labelbottom=False)

        if i == len(time_mean) - 1:
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
            f"Time-mean {metric_name}\n[{time_mean[0].units}]", fontsize="x-large"
        )
        return fig, axs

    return im


def plot_time_mean_bias_list(
    time_mean_bias: List[xr.DataArray],
    baseline_time_mean_bias: List[xr.DataArray],
    var_names: Optional[List[str]] = None,
    figsize=(20, 11),
    vmax_abs=None,
    axs=None,
    verbose=False,
):
    fig, axs = plot_time_mean_list(
        time_mean_bias,
        baseline_time_mean_bias,
        var_names=var_names,
        metric_name="bias",
        figsize=figsize,
        vmax_abs=vmax_abs,
        axs=axs,
        verbose=verbose,
    )
    return fig, axs


def plot_time_mean_rmse_list(
    time_mean_rmse: List[xr.DataArray],
    baseline_time_mean_rmse: List[xr.DataArray],
    var_names: Optional[List[str]] = None,
    figsize=(20, 11),
    vmax_abs=None,
    axs=None,
    verbose=False,
):
    fig, axs = plot_time_mean_list(
        time_mean_rmse,
        baseline_time_mean_rmse,
        var_names=var_names,
        metric_name="rmse",
        figsize=figsize,
        vmax_abs=vmax_abs,
        axs=axs,
        verbose=verbose,
    )
    return fig, axs
