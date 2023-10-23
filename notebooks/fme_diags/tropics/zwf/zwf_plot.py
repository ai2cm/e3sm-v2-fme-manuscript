# BSD 3-Clause License

# Copyright (c) 2018, E3SM:  Energy Exascale Earth System Model
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# James Benedict - Los Alamos National Lab


# Script to compute and plot spectral powers of a subseasonal tropical field in
#   zonal wavenumber-frequency space.  Both the plot files and files containing the
#   associated numerical data shown in the plots are created.
#
# User-defined inputs can be entered near the beginning of the main script (the line
#   if __name__ == "__main__":)
#
# To invoke on NERSC-Cori or LCRC:
#   First, activate E3SM unified environment.  E.g, for NERSC-Cori:
#   > source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_cori-knl.sh
#   Then, simply run:
#   > python zwf_plot.py

import sys
import os
import math
import glob
import numpy as np
import xarray as xr

# our local module:
from . import zwf_functions as wf

# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


CMAP_LIST_RAW = [
    "white",
    "paleturquoise",
    "lightblue",
    "skyblue",
    "lightgreen",
    "limegreen",
    "green",
    "darkgreen",
    "yellow",
    "orange",
    "orangered",
    "red",
    "maroon",
    "magenta",
    "orchid",
    "pink",
    "lavenderblush",
]

CMAP_LIST_NORM = [
    "white",
    "gainsboro",
    "lightgray",
    "silver",
    "paleturquoise",
    "skyblue",
    "lightgreen",
    "mediumseagreen",
    "seagreen",
    "yellow",
    "orange",
    "red",
    "maroon",
    "pink",
]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
    """Return index of [array] closest in value to [value]
    Example:
    array = [ 0.21069679  0.61290182  0.63425412  0.84635244  0.91599191  0.00213826
              0.17104965  0.56874386  0.57319379  0.28719469]
    print(find_nearest(array, value=0.5))
    # 0.568743859261

    """


def wf_analysis(x, segsize=96, noverlap=60, spd=1, latitude_bounds=(-15, 15)):
    """Return zonal wavenumber-frequency power spectra of x.  The returned spectra are:
    spec_sym:    Raw (non-normalized) power spectrum of the component of x that is symmetric about the equator.
    spec_asym:   Raw (non-normalized) power spectrum of the component of x that is antisymmetric about the equator.
    nspec_sym:   Normalized (by a smoothed red-noise background spectrum) power spectrum of the component of x that is symmetric about the equator.
    nspec_asym:  Normalized (by a smoothed red-noise background spectrum) power spectrum of the component of x that is antisymmetric about the equator.

    The NCL version of 'wkSpaceTime' smooths the symmetric and antisymmetric components
    along the frequency dimension using a 1-2-1 filter once.

    """
    # Get the "raw" spectral power
    # OPTIONAL kwargs:
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z_all = wf.spacetime_power(
        x,
        segsize=segsize,
        noverlap=noverlap,
        spd=spd,
        latitude_bounds=latitude_bounds,
        rmvLowFrq=True,
    )
    z2 = z_all.sel(component=["symmetric", "antisymmetric"])
    z2avg = z2.mean(dim="component")
    z_all.loc[{"frequency": 0}] = np.nan  # get rid of spurious power at \nu = 0 (mean)

    # Following NCL's wkSpaceTime, apply one pass of a 1-2-1 filter along the frequency
    #   domain to the raw (non-normalized) spectra/um.
    #   Do not use 0 frequency when smoothing here.
    #   Use weights that sum to 1 to ensure that smoothing is conservative.
    z2s = wf.smoothFrq121(z2, 1)  # data[0][:, data["time"] > 0]

    # The background is supposed to be derived from both symmetric & antisymmetric
    # Inputs to the background spectrum calculation should be z2avg
    background = wf.smoothBackground_wavefreq(z2avg)
    background = background.expand_dims("component", 2).assign_coords(
        component=["background"]
    )
    # separate components
    spec = z_all.sel(component=["total"])
    spec_sym = z2s[0, ...]
    spec_asy = z2s[1, ...]
    data = {
        "spectrum": xr.concat([spec, spec_sym, spec_asy, background], dim="component"),
    }
    return xr.Dataset(data)


def plot_raw_symmetric_spectrum(
    s,
    ofil=None,
    dataDesc=None,
    clevs=None,
    cmapSpec=CMAP_LIST_RAW,
    varName=None,
    sourceID=None,
    do_zoom=False,
    disp_col="black",
    disp_thk=1.5,
    disp_alpha=0.60,
    perfrq_col="dimgray",
    perfrq_thk=1.0,
    perfrq_alpha=0.80,
    equivDepths=[50, 25, 12],
    log=True,
    add_colorbar=True,
    ax=None,
):
    """Basic plot of non-normalized (raw) symmetric power spectrum with shallow water curves."""

    assert varName is not None
    text_offset = 0.005
    fb = [0, 0.8]  # frequency bounds for plot
    wnb = [-15, 15]  # zonal wavenumber bounds for plot
    if max(s["frequency"].values) == 0.5:
        fb = [0, 0.5]
    if do_zoom:
        fb = [0, 0.18]
        wnb = [-7, 7]
    # get data for dispersion curves:
    swfreq, swwn = wf.genDispersionCurves(Ahe=equivDepths)
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    cmapSpecUse = ListedColormap(
        cmapSpec[1:-1]
    )  # recall: range is NOT inclusive for final index
    cmapSpecUse.set_under(cmapSpec[0])
    cmapSpecUse.set_over(cmapSpec[-1])
    normSpecUse = BoundaryNorm(clevs, cmapSpecUse.N)

    # Final data refinement:  transpose and trim, set 0 freq to NaN, take log10, refine metadata
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(*wnb))
    z.loc[{"frequency": 0}] = np.nan
    if log:
        east_power = z.sel(
            frequency=slice((1.0 / 96.0), (1.0 / 24.0)), wavenumber=slice(1, 3)
        ).sum()
        west_power = z.sel(
            frequency=slice((1.0 / 96.0), (1.0 / 24.0)), wavenumber=slice(-3, -1)
        ).sum()
        ew_ratio = east_power / west_power
        print("\neast_power: %12.5f" % east_power)
        print("west_power: %12.5f" % west_power)
        print("ew_ratio: %12.5f\n" % ew_ratio)

        z = np.log10(z)
        z.attrs["long_name"] = (
            varName
            + ": log-base10 of lightly smoothed spectral power of component symmetric about equator"
        )
        z.attrs[
            "method"
        ] = "Follows Figure 1 methods of Wheeler and Kiladis (1999; https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2)"
        z.attrs[
            "ew_ratio_method"
        ] = "Sum of raw (not log10) symmetric spectral power for ZWNs +/- 1-3, periods 24-96 days"
        z.attrs["east_power"] = east_power.values
        z.attrs["west_power"] = west_power.values
        z.attrs["ew_ratio"] = ew_ratio.values

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])
    # img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
    img = ax.contourf(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        cmap=cmapSpecUse,
        norm=normSpecUse,
        extend="both",
    )
    img2 = ax.contour(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        linewidths=1.0,
        linestyles="solid",
        colors="gray",
        alpha=0.7,
    )
    ax.axvline(
        0, linestyle="dashed", color=perfrq_col, linewidth=perfrq_thk, alpha=disp_alpha
    )
    if (1.0 / 30.0) < fb[1]:
        ax.axhline(
            (1.0 / 30.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 30.0) + text_offset,
            "30 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 6.0) < fb[1]:
        ax.axhline(
            (1.0 / 6.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 6.0) + text_offset,
            "6 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 3.0) < fb[1]:
        ax.axhline(
            (1.0 / 3.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 3.0) + text_offset,
            "3 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    for ii in range(3, 6):
        ax.plot(
            swk[ii, 0, :],
            swf[ii, 0, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 1, :],
            swf[ii, 1, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 2, :],
            swf[ii, 2, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
    ax.set_xlim(wnb)
    ax.set_ylim(fb)
    # ax.set_title(varName + ": Log ${\sum_{15^{\circ}S}^{15^{\circ}N} Power_{SYM}}$")   # Version w/ LaTeX
    ax.set_title(f"{varName}: Log{{Sum(Power) from 15°S-15°N}}\n")  # Version w/o LaTeX
    ax.set_title(sourceID, loc="left")
    ax.set_title("Symmetric", loc="right")
    plt.ylabel("Frequency (CPD)")
    plt.xlabel("Zonal wavenumber")
    plt.gcf().text(0.12, 0.03, "Westward", fontsize=11)
    plt.gcf().text(0.64, 0.03, "Eastward", fontsize=11)
    if add_colorbar and fig is not None:
        fig.colorbar(img)
    if ofil is not None and fig is not None:
        fig.savefig(ofil, bbox_inches="tight", dpi=300)
        print("Plot file created: %s\n" % ofil)
    return z, img, img2


def plot_normalized_symmetric_spectrum(
    s,
    ofil=None,
    dataDesc=None,
    clevs=None,
    cmapSpec=CMAP_LIST_NORM,
    varName=None,
    sourceID=None,
    do_zoom=False,
    disp_col="black",
    disp_thk=1.5,
    disp_alpha=0.60,
    perfrq_col="dimgray",
    perfrq_thk=1.0,
    perfrq_alpha=0.80,
    equivDepths=[50, 25, 12],
    add_colorbar=True,
    add_labels=True,
    ax=None,
):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""

    text_offset = 0.005
    fb = [0, 0.8]  # frequency bounds for plot
    wnb = [-15, 15]  # zonal wavenumber bounds for plot
    if max(s["frequency"].values) == 0.5:
        fb = [0, 0.5]
    if do_zoom:
        fb = [0, 0.18]
        wnb = [-7, 7]
    # get data for dispersion curves:
    swfreq, swwn = wf.genDispersionCurves(Ahe=equivDepths)
    # swfreq.shape # -->(6, 3, 50)
    # For n=1 ER waves, allow dispersion curves to touch 0 -- this is for plot aesthetics only
    for i in range(
        0, 3
    ):  # loop 0-->2 for the assumed 3 shallow water dispersion curves for ER waves
        indMinPosFrqER = np.where(
            swwn[3, i, :] >= 0.0, swwn[3, i, :], 1e20
        ).argmin()  # index of swwn for least positive wn
        swwn[3, i, indMinPosFrqER], swfreq[3, i, indMinPosFrqER] = (
            0.0,
            0.0,
        )  # this sets ER's frequencies to 0. at wavenumber 0.
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    #    for i in range(6):
    #      print(f'\nwaveType={i}')
    #      print(f"{'waveNum':12}  {'frq (ih=0)':12}  {'frq (ih=1)':12}  {'frq (ih=2)':12}")
    #      for j in range(50):
    #        print(f'{swk[i,0,j]:12.4f}  {swf[i,0,j]:12.4f}  {swf[i,1,j]:12.4f}  {swf[i,2,j]:12.4f}')
    #    sys.exit()

    cmapSpecUse = ListedColormap(
        cmapSpec[1:-1]
    )  # recall: range is NOT inclusive for final index
    cmapSpecUse.set_under(cmapSpec[0])
    cmapSpecUse.set_over(cmapSpec[-1])
    normSpecUse = BoundaryNorm(clevs, cmapSpecUse.N)

    # Final data refinement:  transpose and trim, set 0 freq to NaN (no log10 for
    #   normalized results), refine metadata
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15, 15))
    z.loc[{"frequency": 0}] = np.nan
    z.attrs["long_name"] = (
        f"{sourceID} {varName}"
        ": lightly smoothed spectral power of component symmetric about equator, "
        "normalized by heavily smoothed background spectrum"
    )
    z.attrs[
        "method"
    ] = "Follows Figure 3 methods of Wheeler and Kiladis (1999; https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])
    # img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
    img = ax.contourf(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        cmap=cmapSpecUse,
        norm=normSpecUse,
        extend="both",
    )
    img2 = ax.contour(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        linewidths=1.0,
        linestyles="solid",
        colors="gray",
        alpha=0.7,
    )
    ax.axvline(
        0, linestyle="dashed", color=perfrq_col, linewidth=perfrq_thk, alpha=disp_alpha
    )
    if (1.0 / 30.0) < fb[1]:
        ax.axhline(
            (1.0 / 30.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 30.0) + text_offset,
            "30 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 6.0) < fb[1]:
        ax.axhline(
            (1.0 / 6.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 6.0) + text_offset,
            "6 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 3.0) < fb[1]:
        ax.axhline(
            (1.0 / 3.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 3.0) + text_offset,
            "3 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    for ii in range(3, 6):
        ax.plot(
            swk[ii, 0, :],
            swf[ii, 0, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 1, :],
            swf[ii, 1, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 2, :],
            swf[ii, 2, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
    ax.set_xlim(wnb)
    ax.set_ylim(fb)
    # ax.set_title(varName + ": $\sum_{15^{\circ}S}^{15^{\circ}N} Power_{SYM}$ / Background")       # Version w/ LaTeX
    # ax.set_title(
    #     f"{varName}: {{Sum(Power) from 15°S-15°N}}/Background\n"
    # )  # Version w/o LaTeX
    ax.set_title(sourceID, loc="left")
    # ax.set_title("Symmetric", loc="right")

    # For now, only add equivalent depth and shallow water curve labels -NOT- to zoomed-in plots
    if add_labels and not do_zoom:
        # Shallow water dispersion curve line labels:  See https://matplotlib.org/stable/tutorials/text/text_intro.html
        # n=1 ER dispersion curve labels
        iwave, ih = 3, 0
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], -11.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 3, 1
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], -9.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 3, 2
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], -8.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            -7.0,
            0.10,
            "n=1 ER",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

        # Kelvin dispersion curve labels
        iwave, ih = 4, 0
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 8.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 4, 1
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 10.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 4, 2
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 14.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            6.0,
            0.13,
            "Kelvin",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

        # IG dispersion curve labels
        iwave, ih = 5, 0
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 0.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 5, 1
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 0.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 5, 2
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 0.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            -10.0,
            0.48,
            "n=1 WIG",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            5.0,
            0.48,
            "n=1 EIG",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

        # MJO label
        ax.text(
            6.0,
            0.0333,
            "MJO",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

    if fig is not None:
        ax.set_ylabel("Frequency (CPD)")

    ax.set_xlabel("Zonal wavenumber")
    ax.text(0.0, -0.1, "Westward", fontsize=11, transform=ax.transAxes)
    ax.text(1.0, -0.1, "Eastward", fontsize=11, transform=ax.transAxes, ha="right")

    if add_colorbar and fig is not None:
        fig.colorbar(img)

    if ofil is not None and fig is not None:
        fig.savefig(ofil, bbox_inches="tight", dpi=300)
        print("Plot file created: %s\n" % ofil)

    # Save plotted data z to file as xArray data array
    # if not do_zoom:
    #     z.to_netcdf(outDataDir + "/zwfData_norm_sym_" + dataDesc + ".nc")
    return z, img, img2


def plot_raw_asymmetric_spectrum(
    s,
    ofil=None,
    dataDesc=None,
    clevs=None,
    cmapSpec="viridis",
    varName=None,
    sourceID=None,
    do_zoom=False,
    disp_col="black",
    disp_thk=1.5,
    disp_alpha=0.60,
    perfrq_col="dimgray",
    perfrq_thk=1.0,
    perfrq_alpha=0.80,
    equivDepths=[50, 25, 12],
):
    """Basic plot of non-normalized (raw) antisymmetric power spectrum with shallow water curves."""

    assert varName is not None
    text_offset = 0.005
    fb = [0, 0.8]  # frequency bounds for plot
    wnb = [-15, 15]  # zonal wavenumber bounds for plot
    if max(s["frequency"].values) == 0.5:
        fb = [0, 0.5]
    if do_zoom:
        fb = [0, 0.18]
        wnb = [-7, 7]
    # get data for dispersion curves:
    swfreq, swwn = wf.genDispersionCurves(Ahe=equivDepths)
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    cmapSpecUse = ListedColormap(
        cmapSpec[1:-1]
    )  # recall: range is NOT inclusive for final index
    cmapSpecUse.set_under(cmapSpec[0])
    cmapSpecUse.set_over(cmapSpec[-1])
    normSpecUse = BoundaryNorm(clevs, cmapSpecUse.N)

    # Final data refinement:  transpose and trim, set 0 freq to NaN, take log10, refine metadata
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15, 15))
    z.loc[{"frequency": 0}] = np.nan
    z = np.log10(z)
    z.attrs["long_name"] = (
        varName
        + ": log-base10 of lightly smoothed spectral power of component antisymmetric about equator"
    )
    z.attrs[
        "method"
    ] = "Follows Figure 1 methods of Wheeler and Kiladis (1999; https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2)"

    fig, ax = plt.subplots()
    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])
    img = ax.contourf(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        cmap=cmapSpecUse,
        norm=normSpecUse,
        extend="both",
    )
    img2 = ax.contour(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        linewidths=1.0,
        linestyles="solid",
        colors="gray",
        alpha=0.7,
    )
    ax.axvline(
        0, linestyle="dashed", color=perfrq_col, linewidth=perfrq_thk, alpha=disp_alpha
    )
    if (1.0 / 30.0) < fb[1]:
        ax.axhline(
            (1.0 / 30.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 30.0) + text_offset,
            "30 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 6.0) < fb[1]:
        ax.axhline(
            (1.0 / 6.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 6.0) + text_offset,
            "6 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 3.0) < fb[1]:
        ax.axhline(
            (1.0 / 3.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 3.0) + text_offset,
            "3 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    for ii in range(0, 3):
        ax.plot(
            swk[ii, 0, :],
            swf[ii, 0, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 1, :],
            swf[ii, 1, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 2, :],
            swf[ii, 2, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
    ax.set_xlim(wnb)
    ax.set_ylim(fb)
    # ax.set_title(varName + ": Log10 $\sum_{15^{\circ}S}^{15^{\circ}N} Power_{ASYM}$")       # Version w/ LaTeX
    ax.set_title(f"{varName}: Log{{Sum(Power) from 15°S-15°N}}\n")  # Version w/o LaTeX
    ax.set_title(sourceID, loc="left")
    ax.set_title("Antisymmetric", loc="right")
    plt.ylabel("Frequency (CPD)")
    plt.xlabel("Zonal wavenumber")
    plt.gcf().text(0.12, 0.03, "Westward", fontsize=11)
    plt.gcf().text(0.64, 0.03, "Eastward", fontsize=11)
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches="tight", dpi=300)
        print("Plot file created: %s\n" % ofil)

    # Save plotted data z to file as xArray data array
    # if not do_zoom:
    #     z.to_netcdf(outDataDir + "/zwfData_raw_asym_" + dataDesc + ".nc")


def plot_normalized_asymmetric_spectrum(
    s,
    ofil=None,
    dataDesc=None,
    clevs=None,
    cmapSpec=CMAP_LIST_NORM,
    varName=None,
    sourceID=None,
    do_zoom=False,
    disp_col="black",
    disp_thk=1.5,
    disp_alpha=0.60,
    perfrq_col="dimgray",
    perfrq_thk=1.0,
    perfrq_alpha=0.80,
    equivDepths=[50, 25, 12],
    add_colorbar=True,
    add_labels=True,
    ax=None,
):
    """Basic plot of normalized antisymmetric power spectrum with shallow water curves."""

    assert varName is not None
    text_offset = 0.005
    fb = [0, 0.8]  # frequency bounds for plot
    wnb = [-15, 15]  # zonal wavenumber bounds for plot
    if max(s["frequency"].values) == 0.5:
        fb = [0, 0.5]
    if do_zoom:
        fb = [0, 0.18]
        wnb = [-7, 7]
    # get data for dispersion curves:
    swfreq, swwn = wf.genDispersionCurves(Ahe=equivDepths)
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    cmapSpecUse = ListedColormap(
        cmapSpec[1:-1]
    )  # recall: range is NOT inclusive for final index
    cmapSpecUse.set_under(cmapSpec[0])
    cmapSpecUse.set_over(cmapSpec[-1])
    normSpecUse = BoundaryNorm(clevs, cmapSpecUse.N)

    # Final data refinement:  transpose and trim, set 0 freq to NaN (no log10 for
    #   normalized results), refine metadata
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15, 15))
    z.loc[{"frequency": 0}] = np.nan
    z.attrs["long_name"] = (
        varName
        + ": lightly smoothed spectral power of component antisymmetric about equator, normalized by heavily smoothed background spectrum"
    )
    z.attrs[
        "method"
    ] = "Follows Figure 3 methods of Wheeler and Kiladis (1999; https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])
    img = ax.contourf(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        cmap=cmapSpecUse,
        norm=normSpecUse,
        extend="both",
    )
    img2 = ax.contour(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        linewidths=1.0,
        linestyles="solid",
        colors="gray",
        alpha=0.7,
    )
    ax.axvline(
        0, linestyle="dashed", color=perfrq_col, linewidth=perfrq_thk, alpha=disp_alpha
    )
    if (1.0 / 30.0) < fb[1]:
        ax.axhline(
            (1.0 / 30.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 30.0) + text_offset,
            "30 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 6.0) < fb[1]:
        ax.axhline(
            (1.0 / 6.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 6.0) + text_offset,
            "6 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 3.0) < fb[1]:
        ax.axhline(
            (1.0 / 3.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 3.0) + text_offset,
            "3 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    for ii in range(0, 3):
        ax.plot(
            swk[ii, 0, :],
            swf[ii, 0, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 1, :],
            swf[ii, 1, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
        ax.plot(
            swk[ii, 2, :],
            swf[ii, 2, :],
            color=disp_col,
            linewidth=disp_thk,
            alpha=perfrq_alpha,
        )
    ax.set_xlim(wnb)
    ax.set_ylim(fb)
    # ax.set_title(varName + ": $\sum_{15^{\circ}S}^{15^{\circ}N} Power_{SYM}$ / Background")       # Version w/ LaTeX
    # ax.set_title(
    #     f"{varName}: {{Sum(Power) from 15°S-15°N}}/Background\n"
    # )  # Version w/o LaTeX
    ax.set_title(sourceID, loc="left")
    # ax.set_title("Antisymmetric", loc="right")

    if add_labels and not do_zoom:
        # Shallow water dispersion curve line labels:  See https://matplotlib.org/stable/tutorials/text/text_intro.html
        # MRG dispersion curve labels -- SKIP LABELING EQUIVALENT DEPTHS FOR MRG WAVES AND ONLY LABEL FOR N=2 EIG WAVES, WHICH ARE POSTIVE-WAVENUMBER EXTENSIONS OF THE MRG CURVES
        #        iwave, ih = 0, 0
        #        idxClose,valClose = find_nearest(swk[iwave,ih,:], 4.)    # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        #        ax.text(valClose,swf[iwave,ih,idxClose],f'{equivDepths[ih]}',fontsize=9,verticalalignment='center',horizontalalignment='center',clip_on=True,bbox={'facecolor': 'white', 'edgecolor':'none', 'alpha': 0.7, 'pad': 0.0})
        #        iwave, ih = 0, 1
        #        idxClose,valClose = find_nearest(swk[iwave,ih,:], 6.)    # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        #        ax.text(valClose,swf[iwave,ih,idxClose],f'{equivDepths[ih]}',fontsize=9,verticalalignment='center',horizontalalignment='center',clip_on=True,bbox={'facecolor': 'white', 'edgecolor':'none', 'alpha': 0.7, 'pad': 0.0})
        #        iwave, ih = 0, 2
        #        idxClose,valClose = find_nearest(swk[iwave,ih,:], 8.)    # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        #        ax.text(valClose,swf[iwave,ih,idxClose],f'{equivDepths[ih]}',fontsize=9,verticalalignment='center',horizontalalignment='center',clip_on=True,bbox={'facecolor': 'white', 'edgecolor':'none', 'alpha': 0.7, 'pad': 0.0})
        ax.text(
            -6.0,
            0.18,
            "MRG",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

        # n=0 EIG dispersion curve labels
        iwave, ih = 1, 0
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 5.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 1, 1
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 8.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 1, 2
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], 8.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            9.0,
            0.48,
            "n=0 EIG",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

        # n=2 IG dispersion curve labels
        iwave, ih = 2, 0
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], -2.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 2, 1
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], -2.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        iwave, ih = 2, 2
        idxClose, valClose = find_nearest(
            swk[iwave, ih, :], -2.0
        )  # Locate index of wavenumber closest to input value [and the actual (float) wavenumber value]
        ax.text(
            valClose,
            swf[iwave, ih, idxClose],
            f"{equivDepths[ih]}",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            -10.0,
            0.65,
            "n=2 WIG",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )
        ax.text(
            8.0,
            0.65,
            "n=2 EIG",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

        # MJO label
        ax.text(
            3.0,
            0.0333,
            "MJO",
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.0},
        )

    if fig is not None:
        plt.ylabel("Frequency (CPD)")

    ax.set_xlabel("Zonal wavenumber")
    ax.text(0.0, -0.1, "Westward", fontsize=11, transform=ax.transAxes)
    ax.text(1.0, -0.1, "Eastward", fontsize=11, transform=ax.transAxes, ha="right")

    if add_colorbar and fig is not None:
        fig.colorbar(img)

    if ofil is not None and fig is not None:
        fig.savefig(ofil, bbox_inches="tight", dpi=300)
        print("Plot file created: %s\n" % ofil)

    # Save plotted data z to file as xArray data array
    # if not do_zoom:
    #     z.to_netcdf(outDataDir + "/zwfData_norm_asym_" + dataDesc + ".nc")
    return z, img, img2


def plot_background_spectrum(
    s,
    ofil=None,
    dataDesc=None,
    clevs=None,
    cmapSpec="viridis",
    varName=None,
    sourceID=None,
    do_zoom=False,
    disp_col="black",
    disp_thk=1.5,
    disp_alpha=0.60,
    perfrq_col="dimgray",
    perfrq_thk=1.0,
    perfrq_alpha=0.80,
    equivDepths=[50, 25, 12],
):
    """Basic plot of background power spectrum with shallow water curves."""

    assert varName is not None
    text_offset = 0.005
    fb = [0, 0.8]  # frequency bounds for plot
    wnb = [-15, 15]  # zonal wavenumber bounds for plot
    if max(s["frequency"].values) == 0.5:
        fb = [0, 0.5]
    if do_zoom:
        fb = [0, 0.18]
        wnb = [-7, 7]
    # get data for dispersion curves:
    swfreq, swwn = wf.genDispersionCurves(Ahe=equivDepths)
    # swfreq.shape # -->(6, 3, 50)
    # For n=1 ER waves, allow dispersion curves to touch 0 -- this is for plot aesthetics only
    for i in range(
        0, 3
    ):  # loop 0-->2 for the assumed 3 shallow water dispersion curves for ER waves
        indMinPosFrqER = np.where(
            swwn[3, i, :] >= 0.0, swwn[3, i, :], 1e20
        ).argmin()  # index of swwn for least positive wn
        swwn[3, i, indMinPosFrqER], swfreq[3, i, indMinPosFrqER] = (
            0.0,
            0.0,
        )  # this sets ER's frequencies to 0. at wavenumber 0.
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    cmapSpecUse = ListedColormap(
        cmapSpec[1:-1]
    )  # recall: range is NOT inclusive for final index
    cmapSpecUse.set_under(cmapSpec[0])
    cmapSpecUse.set_over(cmapSpec[-1])
    normSpecUse = BoundaryNorm(clevs, cmapSpecUse.N)

    # Final data refinement:  transpose and trim, set 0 freq to NaN, take log10, refine metadata
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15, 15))
    z.loc[{"frequency": 0}] = np.nan
    z = np.log10(z)
    z.attrs["long_name"] = (
        varName
        + ": heavily smoothed version of the mean of spectral powers associated with the components symmetric and antisymmetric about equator"
    )
    z.attrs[
        "method"
    ] = "Follows Figure 2 methods of Wheeler and Kiladis (1999; https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2)"

    fig, ax = plt.subplots()
    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])
    img = ax.contourf(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        cmap=cmapSpecUse,
        norm=normSpecUse,
        extend="both",
    )
    img2 = ax.contour(
        kmesh0,
        vmesh0,
        z,
        levels=clevs,
        linewidths=1.0,
        linestyles="solid",
        colors="gray",
        alpha=0.7,
    )
    ax.axvline(
        0, linestyle="dashed", color=perfrq_col, linewidth=perfrq_thk, alpha=disp_alpha
    )
    if (1.0 / 30.0) < fb[1]:
        ax.axhline(
            (1.0 / 30.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 30.0) + text_offset,
            "30 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 6.0) < fb[1]:
        ax.axhline(
            (1.0 / 6.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 6.0) + text_offset,
            "6 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    if (1.0 / 3.0) < fb[1]:
        ax.axhline(
            (1.0 / 3.0), linestyle="dashed", color=perfrq_col, alpha=perfrq_alpha
        )
        ax.text(
            wnb[0] + 1,
            (1.0 / 3.0) + text_offset,
            "3 days",
            color=perfrq_col,
            alpha=perfrq_alpha,
        )
    ax.set_xlim(wnb)
    ax.set_ylim(fb)
    ax.set_title(f"{varName}: Log{{Smoothed Background Power}}\n")
    ax.set_title(sourceID, loc="left")
    #    ax.set_title("", loc='right')
    plt.ylabel("Frequency (CPD)")
    plt.xlabel("Zonal wavenumber")
    plt.gcf().text(0.12, 0.03, "Westward", fontsize=11)
    plt.gcf().text(0.64, 0.03, "Eastward", fontsize=11)
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches="tight", dpi=300)
        print("Plot file created: %s\n" % ofil)

    # Save plotted data z to file as xArray data array
    # if not do_zoom:
    #     z.to_netcdf(outDataDir + "/zwfData_background_" + dataDesc + ".nc")
