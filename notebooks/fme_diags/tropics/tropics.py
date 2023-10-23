from typing import Literal, Tuple

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt

from .zwf.zwf_plot import (plot_normalized_asymmetric_spectrum,
                           plot_normalized_symmetric_spectrum, wf_analysis)
from .zwf.zwf_plot_DIFF import plot_diff_spectrum


def compute_wavenumber_frequency_spectra(
    da: xr.DataArray,
    daily_mean: bool = True,
    segment_size: int = 96,
    overlap: int = 60,
    latitude_bounds: Tuple[int, int] = (-15, 15),
) -> xr.Dataset:
    """Compute the symmetric and antisymmetric components of the
    wavenumber-frequency spectra of a target variable and corresponding
    predictions, together with the background power for each.

    Args
    ----
    da: A data variable from an autogressive predictions dataset, opened with
        open_autoregressive_predictions.
    daily_mean: Compute daily means before running wavenumber-frequency
        analysis.
    segment_size: The number of days to decompose.
    overlap: The number of overlapping days in consecutive segments.
    latitude_bounds: A tuple of (southern extent, northern extent), in degrees.

    """
    if daily_mean:
        da = da.resample(time="1D").mean("time")
        samples_per_day = 1
    else:
        day_slice = next(da.resample(time="1D").groups.values().__iter__())
        samples_per_day = day_slice.stop - day_slice.start
    spectra = []
    with ProgressBar():
        for source in da["source"]:
            spectra.append(
                wf_analysis(
                    da.sel(source=source),
                    segsize=segment_size,
                    noverlap=overlap,
                    spd=samples_per_day,
                    latitude_bounds=latitude_bounds,
                )
            )
            spectra[-1]["source"] = source
    spectra = xr.concat(spectra, dim="source")
    return spectra


def plot_wavenumber_frequency_spectra(
    ds: xr.Dataset,
    component: Literal["symmetric", "antisymmetric", "total", "background"],
    normalize: bool = True,
    mjo_zoom: bool = False,
    num_contour_bins=14,
    figsize=(16, 5),
):
    """Plot the target and predicted power spectra on a log10 scale, along with
    the relative error.

    Args
    ----
    ds: Spectra dataset, as output from compute_wavenumber_frequency_spectra.
    component: The spectral component to plot.
    normalize: Divide by background power, unless plotting the background spectra itself.
    mjo_zoom: Zoom the plot in on the MJO region.
    num_countour_bins: The number of discrete color levels.
    figsize: Passed to plt.subplots.

    """
    if component == "antisymmetric":
        plot_fun = plot_normalized_asymmetric_spectrum
    else:
        plot_fun = plot_normalized_symmetric_spectrum

    if component == "background":
        normalize = False

    power = {}
    for source in ds["source"].values:
        power[source] = ds["spectrum"].sel(component=component, source=source)
        if normalize:
            background = ds["spectrum"].sel(component="background", source=source)
            power[source] = power[source] / background
        else:
            # plot raw/background spectra on log scale
            power[source] = np.log10(power[source])

    tar_power = (
        power["target"]
        .sel(wavenumber=slice(-15, 15), frequency=slice(0, None))
        .values.ravel()
    )
    tar_power = tar_power[~np.isnan(tar_power)]
    _, bin_edges = np.histogram(tar_power, bins=num_contour_bins)

    fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    axs_flat = axs.flatten()

    _, img, _ = plot_fun(
        power["target"],
        clevs=bin_edges[1:-1],
        varName="surface_precipitation_rate",
        sourceID="Target",
        do_zoom=mjo_zoom,
        add_colorbar=False,
        ax=axs_flat[0],
    )
    axs_flat[0].set_ylabel("Frequency (CPD)")

    _ = plot_fun(
        power["prediction"],
        clevs=bin_edges[1:-1],
        varName="surface_precipitation_rate",
        sourceID="Generated",
        do_zoom=mjo_zoom,
        add_colorbar=False,
        add_labels=False,
        ax=axs_flat[1],
    )
    fig.colorbar(img, ax=[axs_flat[1]], orientation="vertical", location="right")

    if not normalize:
        # undo log scaling for raw and background spectra
        for source in ds["source"].values:
            power[source] = 10 ** power[source]

    err = 100 * (power["prediction"] - power["target"]) / power["target"]

    # define the colormap for errors
    cmap = plt.cm.seismic
    cmap_list = (
        [cmap(0)] + [cmap(i) for i in range(31, cmap.N, cmap.N // 12)] + [cmap(255)]
    )
    # set 0 to white
    cmap_list[6] = cmap(128)
    bin_limit = (
        np.abs(err.sel(wavenumber=slice(-15, 15), frequency=slice(0, None)))
        .max()
        .item()
    )
    bin_edges = np.linspace(-bin_limit, bin_limit, num=12)
    _, img, _ = plot_diff_spectrum(
        err,
        clevs=bin_edges,
        do_zoom=mjo_zoom,
        rightStr=component,
        add_colorbar=False,
        ax=axs_flat[2],
    )
    fig.colorbar(
        img, ax=[axs_flat[2]], label="%", orientation="vertical", location="right"
    )
    fig.suptitle(
        "Precipitation rate wavenumber-frequency spectrum"
        f" ({'normalized ' if normalize else ''}{component})"
    )

    return fig, axs
