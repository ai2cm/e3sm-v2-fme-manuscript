from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt


def compute_histograms(
    da: xr.DataArray,
    n_bins=300,
    hist_range: Optional[Tuple[float, float]] = None,
    lat_weighted=False,
) -> xr.Dataset:
    """Compute the histogram of a precipitation DataArray.

    Args
    ----

    da: xr.DataArray The surface precipitation rate variable from FME inference
        outputs. The dataset where da comes from was should have been opened
        using utils.open_autoregressive_inference.

    """
    assert "time" in da.dims
    assert "source" in da.dims
    assert "sample" in da.dims

    if hist_range is None:
        with ProgressBar():
            hist_range = da.min().compute().item(), da.max().compute().item()

    bin_edges = np.linspace(*hist_range, n_bins + 1)

    if lat_weighted:
        weights = np.resize(np.cos(np.deg2rad(da["lat"].values)), da.shape[-1:-3:-1]).T
    else:
        weights = np.ones(da.shape[-2:])

    # coordinates for this histogram DataArray
    coords = {
        # ignore the initial condition which is the same in "target" and "prediction"
        "source": da["source"],
        "sample": da["sample"],
        "time": da["time"][1:],
        "bin_id": np.arange(n_bins),
    }

    samples = []
    for i_sample in range(len(da["sample"])):
        tar_hists = []
        gen_hists = []
        # TODO: parallelize
        for i_time in range(1, len(da["time"])):
            da_ = da.isel(sample=i_sample, time=i_time)
            tar_hist, _ = np.histogram(
                da_.sel(source="target").values,
                bins=bin_edges,
                weights=weights,
            )
            gen_hist, _ = np.histogram(
                da_.sel(source="prediction").values,
                bins=bin_edges,
                weights=weights,
            )
            tar_hists.append(tar_hist)
            gen_hists.append(gen_hist)
        # stack along time dimension and create a new dimension for sample: sample, time, bins
        tar_hists = np.expand_dims(np.stack(tar_hists), 0)
        gen_hists = np.expand_dims(np.stack(gen_hists), 0)
        # stack along new dimension for source
        samples.append(np.stack([tar_hists, gen_hists]))
    # concat along the sample dimension
    samples = xr.DataArray(np.concatenate(samples, axis=1), coords=coords)

    return xr.Dataset({"hist": samples, "bin_edges": bin_edges})


def plot_time_mean_histogram(
    ds: xr.Dataset,
    sample=0,
    figsize=(10, 5),
    labels: Optional[Dict[str, str]] = None,
    **hist_kwargs
):
    if labels is None:
        labels = {"target": "Target", "prediction": "Prediction"}
    else:
        assert set(labels.keys()) == {
            "target",
            "prediction",
        }, "labels must be a dict with keys 'target' and 'prediction'"
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    ds = ds.sel(sample=sample)
    bin_edges = ds["bin_edges"].values
    tar_hist_mean = ds["hist"].sel(source="target").mean(dim="time").values
    gen_hist_mean = ds["hist"].sel(source="prediction").mean(dim="time").values
    _ = ax.hist(
        bin_edges[:-1],
        bin_edges,
        weights=tar_hist_mean,
        color="k",
        linestyle="--",
        label=labels["target"],
        **hist_kwargs
    )
    _ = ax.hist(
        bin_edges[:-1],
        bin_edges,
        weights=gen_hist_mean,
        color="g",
        linestyle="-",
        label=labels["prediction"],
        **hist_kwargs
    )
    return fig, ax
