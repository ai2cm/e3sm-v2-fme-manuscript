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

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# our local module:
from . import zwf_functions as wf


def plot_diff_spectrum(
    s,
    ofil=None,
    clevs=None,
    do_zoom=False,
    titleStr="Main title",
    leftStr="Left title",
    rightStr="Right title",
    cbar_label="cbar_label",
    disp_col="black",
    disp_thk=1.5,
    disp_alpha=0.60,
    perfrq_col="darkgray",
    perfrq_thk=1.0,
    perfrq_alpha=0.80,
    add_colorbar=True,
    ax=None,
):
    """Basic plot of difference of either non-normalized (raw) or normalized antisymmetric power
    spectra with shallow water curves."""

    text_offset = 0.005
    fb = [0, min(0.8, max(s["frequency"].values))]  # frequency bounds for plot
    wnb = [-15, 15]  # zonal wavenumber bounds for plot
    if do_zoom:
        fb = [0, 0.18]
        wnb = [-7, 7]
    # get data for dispersion curves:
    swfreq, swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15, 15))
    z.loc[{"frequency": 0}] = np.nan

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    kmesh0, vmesh0 = np.meshgrid(z["wavenumber"], z["frequency"])
    # img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
    img = ax.contourf(kmesh0, vmesh0, z, levels=clevs, cmap="seismic", extend="both")
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
    if rightStr == "symmetric":
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
    if rightStr == "antisymmetric":
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
    # ax.set_title(titleStr)
    # ax.set_title(leftStr, loc="left")
    ax.set_title("Error", loc="right")
    plt.ylabel("Frequency (CPD)")
    plt.xlabel("Zonal wavenumber")
    ax.text(0.0, -0.1, "Westward", fontsize=11, transform=ax.transAxes)
    ax.text(1.0, -0.1, "Eastward", fontsize=11, transform=ax.transAxes, ha="right")

    if add_colorbar and fig is not None:
        fig.colorbar(img, ticks=clevs, label=cbar_label)

    if ofil is not None and fig is not None:
        fig.savefig(ofil, bbox_inches="tight", dpi=300)
        print("Plot file created: %s\n" % ofil)

    return z, img, img2


#
# Load spectral data [dims=(frequency,wavenumber)]
#
def get_power_data(fName):
    print("\nRetrieving power data from file: " + fName)
    try:
        ds = xr.open_dataset(fName)
    except:
        sys.exit("Exiting: Error reading data set: %s" % fName)
    # xPow = ds["power"].compute()
    # return xPow
    return ds["power"]


if __name__ == "__main__":
    #
    # ==================   User parameters (below)   ============================
    #
    #   * vari:  Variable label in input filename, will also appear on plot
    #   * zwf_inData_dir:  Path to directory containing pre-processed spectral data
    #   * srcID_[A,B]:  Source ID descriptor for pre-processed spectral data
    #   * srcID_[A,B]_short:  Abbreviated version of srcID_[A,B] for plot title and filename
    #   * yrSpanFile_[A,B]:  Year span label in input file names, will also appear in
    #                        plot filename
    #   * srcID_short_diffFile:  Descriptor of spectral difference to be used in the plot filename
    #   * srcID_short_diffFile:  Descriptor of spectral difference to be used in the plot title
    #   * outPlotDir:  Directory in which plot file will be saved
    #
    vari = "PRECT"

    zwf_inData_dir = "/home/ac.benedict/analysis/results/dataTrop"

    # "A" and "B" data sources:  Difference will be defined as A-B
    #    srcID_A = "v2.LR.historical_0101"
    #    srcID_A_short = "v2hist"
    #    yrSpanFile_A = "2005-2014"
    #
    #    srcID_B = "DECK.v1b.LR.CMIP6"
    #    srcID_B_short = "v1hist"
    #    yrSpanFile_B = "2005-2014"
    #
    ##    srcID_B = "TRMM"
    ##    srcID_B_short = "TRMM"
    ##    yrSpanFile_B = "2001-2010"

    #    srcID_A = "DECKv1b_A1"
    #    srcID_A_short = "v1amip"
    #    yrSpanFile_A = "2001-2010"

    srcID_A = "v2.LR.piControl"
    "v2.LR.amip_0201"
    srcID_A_short = "v2piControl"
    "v2amip"
    yrSpanFile_A = "0460-0489"
    "1985-2014"

    srcID_B = "DECKv1b_piControl"
    "DECKv1b_A1"
    srcID_B_short = "v1piControl"
    "v1amip"
    yrSpanFile_B = "0250-0279"
    "1985-2014"

    #    srcID_B = "TRMM"
    #    srcID_B_short = "TRMM"
    #    yrSpanFile_B = "2001-2010"

    srcID_short_diffFile = "%s-%s" % (srcID_A_short, srcID_B_short)
    srcID_short_diffPlot = "%s$-$%s" % (srcID_A_short, srcID_B_short)

    # outputs
    #    outDataDir = "/home/ac.benedict/analysis/diag_e3sm_py_20210709/testing/test_plots"
    outPlotDir = "/home/ac.benedict/analysis/results/plotTrop"

    do_zooming = True  # Set to True to also make plots to zoom into MJO spectral region

    # Generic settings for spectra plots -- do not need to ba adjusted by user, but can be
    #   customized.  The user can add additional key-value pairs as necessary to expand
    #   plotting customization.
    #    contour_levs_raw_diff  = (-0.2,-0.15,-0.1,-0.05,-0.02,-0.01,0.01,0.02,0.05,0.1,0.15,0.2)
    #    contour_levs_norm_diff = (-0.2,-0.15,-0.1,-0.05,-0.02,-0.01,0.01,0.02,0.05,0.1,0.15,0.2)
    contour_levs_raw_diffRatio = (
        -80.0,
        -60.0,
        -40.0,
        -20.0,
        -10.0,
        -5.0,
        5.0,
        10.0,
        20.0,
        40.0,
        60.0,
        80.0,
    )
    contour_levs_norm_diffRatio = (
        -60.0,
        -30.0,
        -20.0,
        -15.0,
        -10.0,
        -5.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        60.0,
    )

    # Dictionary with additional plot resources -- NOTE: These are default settings,
    #   some of which are changed just below the call the the plotting routine.
    optPlot = {
        "do_zoom": False,
        "titleStr": "",
        "leftStr": "",
        "rightStr": "",
        "cbar_label": "",
        "disp_col": "black",
        "disp_thk": 1.5,
        "disp_alpha": 0.60,
        "perfrq_col": "darkgray",
        "perfrq_thk": 1.0,
        "perfrq_alpha": 0.80,
    }
    # disp_col      # Color for dispersion lines/labels
    # disp_thk      # Thickness for dispersion lines
    # disp_alpha    # Transparency for dispersion lines/labels (alpha=0 for opaque)
    # perfrq_col    # Color for period and frequency ref lines
    # perfrq_thk    # Thickness for period and frequency ref lines
    # perfrq_alpha  # Transparency for period and frequency ref lines (alpha=0 for opaque)
    # ==================   User parameters (above)   ============================

    # Error trapping and warnings

    # =========================================
    # Load data

    # ---  Source "A"  ---
    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_raw_sym_"
        + srcID_A
        + "_"
        + yrSpanFile_A
        + "_"
        + vari
        + ".nc"
    )
    xA_raw_sym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_norm_sym_"
        + srcID_A
        + "_"
        + yrSpanFile_A
        + "_"
        + vari
        + ".nc"
    )
    xA_norm_sym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_raw_asym_"
        + srcID_A
        + "_"
        + yrSpanFile_A
        + "_"
        + vari
        + ".nc"
    )
    xA_raw_asym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_norm_asym_"
        + srcID_A
        + "_"
        + yrSpanFile_A
        + "_"
        + vari
        + ".nc"
    )
    xA_norm_asym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_background_"
        + srcID_A
        + "_"
        + yrSpanFile_A
        + "_"
        + vari
        + ".nc"
    )
    xA_background = get_power_data(f)

    # ---  Source "B"  ---
    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_raw_sym_"
        + srcID_B
        + "_"
        + yrSpanFile_B
        + "_"
        + vari
        + ".nc"
    )
    xB_raw_sym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_norm_sym_"
        + srcID_B
        + "_"
        + yrSpanFile_B
        + "_"
        + vari
        + ".nc"
    )
    xB_norm_sym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_raw_asym_"
        + srcID_B
        + "_"
        + yrSpanFile_B
        + "_"
        + vari
        + ".nc"
    )
    xB_raw_asym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_norm_asym_"
        + srcID_B
        + "_"
        + yrSpanFile_B
        + "_"
        + vari
        + ".nc"
    )
    xB_norm_asym = get_power_data(f)

    f = (
        zwf_inData_dir
        + "/"
        + "zwfData_background_"
        + srcID_B
        + "_"
        + yrSpanFile_B
        + "_"
        + vari
        + ".nc"
    )
    xB_background = get_power_data(f)

    # =========================================
    # Unit conversion:  "undo" log-10 for raw and background spectra
    xA_raw_sym = 10.0**xA_raw_sym  # this will clobber attributes, but this is okay
    xA_raw_asym = 10.0**xA_raw_asym
    xA_background = 10.0**xA_background
    xB_raw_sym = 10.0**xB_raw_sym  # this will clobber attributes, but this is okay
    xB_raw_asym = 10.0**xB_raw_asym
    xB_background = 10.0**xB_background

    # =========================================
    # Compute difference spectra
    rawDiff_sym = xA_raw_sym - xB_raw_sym
    rawDiffRatio_sym = 100.0 * (xA_raw_sym - xB_raw_sym) / xB_raw_sym

    normDiff_sym = xA_norm_sym - xB_norm_sym
    normDiffRatio_sym = 100.0 * (xA_norm_sym - xB_norm_sym) / xB_norm_sym

    rawDiff_asym = xA_raw_asym - xB_raw_asym
    rawDiffRatio_asym = 100.0 * (xA_raw_asym - xB_raw_asym) / xB_raw_asym

    normDiff_asym = xA_norm_asym - xB_norm_asym
    normDiffRatio_asym = 100.0 * (xA_norm_asym - xB_norm_asym) / xB_norm_asym

    backgroundDiff = xA_background - xB_background
    backgroundDiffRatio = 100.0 * (xA_background - xB_background) / xB_background

    # =========================================
    # Plot spectra

    # The 'plotFileDescriptor' string goes into the output file name encapsulating the
    #   data that is plotted:  'yrBegData_A' and 'yrEndData_A' are used for simplicity,
    #   assuming that the time windows from the A and B data source mostly overlap
    plotFileDescriptor = srcID_short_diffFile + "_" + yrSpanFile_A + "_" + vari

    # ---  Ratio of differences of raw spectra  ---
    outPlotName = (
        outPlotDir + "/" + "zwfPlot_diffRatio_raw_sym_" + plotFileDescriptor + ".png"
    )
    print("\nPlotting: %s" % outPlotName)
    optPlot.update({"do_zoom": False})
    optPlot.update({"titleStr": "%s: $\Delta$(Raw power)\n" % vari})
    optPlot.update(
        {"leftStr": "(%s$-$%s)/%s" % (srcID_A_short, srcID_B_short, srcID_B_short)}
    )
    optPlot.update({"rightStr": "Symmetric"})
    optPlot.update({"cbar_label": "%"})
    plot_diff_spectrum(
        rawDiffRatio_sym, outPlotName, contour_levs_raw_diffRatio, **optPlot
    )
    if do_zooming:
        outPlotName = (
            outPlotDir
            + "/"
            + "zwfPlot_diffRatio_raw_sym_ZOOM_"
            + plotFileDescriptor
            + ".png"
        )
        optPlot.update({"do_zoom": True})
        plot_diff_spectrum(
            rawDiffRatio_sym, outPlotName, contour_levs_raw_diffRatio, **optPlot
        )

    outPlotName = (
        outPlotDir + "/" + "zwfPlot_diffRatio_raw_asym_" + plotFileDescriptor + ".png"
    )
    print("Plotting: %s" % outPlotName)
    optPlot.update({"do_zoom": False})
    optPlot.update({"titleStr": "%s: $\Delta$(Raw power)\n" % vari})
    optPlot.update(
        {"leftStr": "(%s$-$%s)/%s" % (srcID_A_short, srcID_B_short, srcID_B_short)}
    )
    optPlot.update({"rightStr": "Antisymmetric"})
    optPlot.update({"cbar_label": "%"})
    plot_diff_spectrum(
        rawDiffRatio_asym, outPlotName, contour_levs_raw_diffRatio, **optPlot
    )
    if do_zooming:
        outPlotName = (
            outPlotDir
            + "/"
            + "zwfPlot_diffRatio_raw_asym_ZOOM_"
            + plotFileDescriptor
            + ".png"
        )
        optPlot.update({"do_zoom": True})
        plot_diff_spectrum(
            rawDiffRatio_asym, outPlotName, contour_levs_raw_diffRatio, **optPlot
        )

    outPlotName = (
        outPlotDir + "/" + "zwfPlot_diffRatio_background_" + plotFileDescriptor + ".png"
    )
    print("Plotting: %s" % outPlotName)
    optPlot.update({"do_zoom": False})
    optPlot.update({"titleStr": "%s: $\Delta$(Raw power)\n" % vari})
    optPlot.update(
        {"leftStr": "(%s$-$%s)/%s" % (srcID_A_short, srcID_B_short, srcID_B_short)}
    )
    optPlot.update({"rightStr": "Background"})
    optPlot.update({"cbar_label": "%"})
    plot_diff_spectrum(
        backgroundDiffRatio, outPlotName, contour_levs_raw_diffRatio, **optPlot
    )
    if do_zooming:
        outPlotName = (
            outPlotDir
            + "/"
            + "zwfPlot_diffRatio_background_ZOOM_"
            + plotFileDescriptor
            + ".png"
        )
        optPlot.update({"do_zoom": True})
        plot_diff_spectrum(
            backgroundDiffRatio, outPlotName, contour_levs_raw_diffRatio, **optPlot
        )

    # ---  Ratio of differences of normalized spectra  ---
    outPlotName = (
        outPlotDir + "/" + "zwfPlot_diffRatio_norm_sym_" + plotFileDescriptor + ".png"
    )
    print("Plotting: %s" % outPlotName)
    optPlot.update({"do_zoom": False})
    optPlot.update({"titleStr": "%s: $\Delta$(Normalized power)\n" % vari})
    optPlot.update(
        {"leftStr": "(%s$-$%s)/%s" % (srcID_A_short, srcID_B_short, srcID_B_short)}
    )
    optPlot.update({"rightStr": "Symmetric"})
    optPlot.update({"cbar_label": "%"})
    plot_diff_spectrum(
        normDiffRatio_sym, outPlotName, contour_levs_norm_diffRatio, **optPlot
    )
    if do_zooming:
        outPlotName = (
            outPlotDir
            + "/"
            + "zwfPlot_diffRatio_norm_sym_ZOOM_"
            + plotFileDescriptor
            + ".png"
        )
        optPlot.update({"do_zoom": True})
        plot_diff_spectrum(
            normDiffRatio_sym, outPlotName, contour_levs_norm_diffRatio, **optPlot
        )

    outPlotName = (
        outPlotDir + "/" + "zwfPlot_diffRatio_norm_asym_" + plotFileDescriptor + ".png"
    )
    print("Plotting: %s" % outPlotName)
    optPlot.update({"do_zoom": False})
    optPlot.update({"titleStr": "%s: $\Delta$(Normalized power)\n" % vari})
    optPlot.update(
        {"leftStr": "(%s$-$%s)/%s" % (srcID_A_short, srcID_B_short, srcID_B_short)}
    )
    optPlot.update({"rightStr": "Antisymmetric"})
    optPlot.update({"cbar_label": "%"})
    plot_diff_spectrum(
        normDiffRatio_asym, outPlotName, contour_levs_norm_diffRatio, **optPlot
    )
    if do_zooming:
        outPlotName = (
            outPlotDir
            + "/"
            + "zwfPlot_diffRatio_norm_asym_ZOOM_"
            + plotFileDescriptor
            + ".png"
        )
        optPlot.update({"do_zoom": True})
        plot_diff_spectrum(
            normDiffRatio_asym, outPlotName, contour_levs_norm_diffRatio, **optPlot
        )
