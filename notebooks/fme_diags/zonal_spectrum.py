from typing import Optional, Tuple

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS


def weighted_mean(da, weights, dim=None):
    return da.weighted(weights).mean(dim)


def compute_zonal_mean_spectra(da: xr.DataArray, lat_weights=None) -> xr.DataArray:
    power = xr.apply_ufunc(
        lambda x: np.abs(np.fft.rfft(x, axis=-1)) ** 2,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "zonal_wavenumber"]],
        dask="parallelized",
        output_dtypes=[da.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {"zonal_wavenumber": da.shape[-1] // 2 + 1}
        },
        keep_attrs=True,
    ).assign_coords({"zonal_wavenumber": np.arange(da.shape[-1] // 2 + 1)})
    if lat_weights is None:
        lat_weights = xr.ones_like(da["lat"])
    return weighted_mean(power, lat_weights, dim="lat")


def compute_zonal_time_mean_spectrum(
    da: xr.DataArray, lat_weights=None
) -> Tuple[xr.DataArray, xr.DataArray]:
    spectra = compute_zonal_mean_spectra(da, lat_weights=lat_weights)
    with ProgressBar():
        spectra_mean = spectra.mean("time").compute()
        spectra_std = spectra.std("time").compute()
    spectra_mean.attrs["n_time"] = len(da["time"])
    spectra_std.attrs["n_time"] = len(da["time"])
    return spectra_mean, spectra_std


def plot_zonal_time_mean_spectra(
    spectra_mean: xr.DataArray,
    spectra_std: xr.DataArray = None,
    stderr: bool = False,
    wavenumber_slice: Optional[slice] = None,
    source_to_label=None,
    ax=None,
    figsize=(10, 5),
):
    if source_to_label is None:
        source_to_label = {
            "target": "Target",
            "prediction": "Generated",
        }
    sources = source_to_label.keys()
    colors = list(TABLEAU_COLORS)
    if wavenumber_slice is not None:
        spectra_mean = spectra_mean.sel(zonal_wavenumber=wavenumber_slice)
    x = spectra_mean["zonal_wavenumber"].values
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    for i, source in enumerate(sources):
        linestyle = "-"
        if source == "target":
            linestyle = "--"
            colors.insert(i, "k")
        elif source == "prediction":
            colors.insert(i, "g")
        y = spectra_mean.sel(source=source).values
        ax.semilogy(
            x, y, label=source_to_label[source], color=colors[i], linestyle=linestyle
        )
        if spectra_std is not None:
            if wavenumber_slice is not None:
                spectra_std = spectra_std.sel(zonal_wavenumber=wavenumber_slice)
            std = spectra_std.sel(source=source).values
            if stderr:
                std = std / np.sqrt(spectra_std.attrs["n_time"])
            ax.fill_between(
                x,
                y - std,
                y + std,
                color=colors[i],
                alpha=0.2,
            )
    ax.set_xlabel("Zonal wavenumber")
    plt.legend()
    return fig, ax
