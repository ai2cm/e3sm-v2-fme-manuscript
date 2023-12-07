from typing import Optional

import xarray as xr
from matplotlib import pyplot as plt


def plot_hovmoller_by_lon(
    da: xr.DataArray,
    var_name: Optional[str] = None,
    figsize=(6, 12),
    time_label="Simulation time",
):
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    vmin, vmax = da.min(), da.max()

    da.sel(source="target").plot(ax=axs[0], vmin=vmin, vmax=vmax, add_colorbar=False)
    im = da.sel(source="prediction").plot(
        ax=axs[1], vmin=vmin, vmax=vmax, add_colorbar=False
    )

    axs[0].set_title("Target")
    axs[0].set_ylabel(time_label)
    axs[0].set_xlabel("Longitude")
    axs[1].set_title("Generated")
    axs[1].set_ylabel("")
    axs[1].yaxis.set_tick_params(labelleft=False)
    axs[1].set_xlabel("Longitude")

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
