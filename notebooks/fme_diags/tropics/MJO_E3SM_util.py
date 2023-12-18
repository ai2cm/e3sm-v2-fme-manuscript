# 2023-12-18: received from Lin Yao (linyao@uchicago.edu)

# This file contains functions for the MJO-E3SM project

import numpy as np
import xarray as xr
from scipy.signal import detrend


def split_hann_taper_pnt(seg_size, pnt=100):
    """
    seg_size: the size of the taper;
    pnt: the number of the total points used to do hanning window
    """
    npts = int(pnt)  # the total length of hanning window
    hann_taper = np.hanning(npts)  # generate Hanning taper
    taper = np.ones(seg_size)  # create the split cosine bell taper

    # copy the first half of hanner taper to target taper
    taper[: npts // 2] = hann_taper[: npts // 2]
    # copy the second half of hanner taper to the target taper
    # taper[-npts//2-1:] = hann_taper[-npts//2-1:]
    taper[-(npts // 2) :] = hann_taper[-(npts // 2) :]
    return taper


# get MJO-related signals by filtering the Wheeler-Kiladis spectra
def detrend_func(x):
    return detrend(x, axis=0)


def get_MJO_signal(
    u, d=1, kmin=1, kmax=3, flow=1 / 100.0, fhig=1 / 20.0, detrendflg=True
):
    """
    0. input u=u[time, lon]
    1. optional: detrend the data in time. Defult is True.
    2. apply taper in time
    3. Fourier transform in time and space.
    4. remove coefficients outside k=1-3, T=20-100 day
    5. reconstruct u
    """

    # detrend
    if detrendflg:
        u_detrended = xr.apply_ufunc(
            detrend_func,
            u,
            # input_core_dims=[['time']],
            # output_core_dims=[['time']],
            dask="parallelized",
            output_dtypes=[u.dtype],
        )
    else:
        u_detrended = u.copy()

    # taper
    # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
    taper = split_hann_taper_pnt(seg_size=len(u["time"]))

    u_detrend_tap = u_detrended * xr.DataArray(taper, dims=["time"])

    # Fourier transform
    lon_index = u.dims.index("lon")
    u_lon = np.fft.fft(u_detrend_tap.values, axis=lon_index)  # [time, k]

    u_lon[:, :kmin] = 0.0
    u_lon[:, kmax + 1 : -kmax] = 0.0

    time_index = u.dims.index("time")
    u_lon_time = np.fft.fft(u_lon, axis=time_index)

    freq = np.fft.fftfreq(len(u["time"]), d)
    tlow = np.argmin(np.abs(freq - flow))
    thig = np.argmin(np.abs(freq - fhig))

    if tlow == 1:
        u_lon_time[:tlow, :] = 0.0
        u_lon_time[thig + 1 : -thig, :] = 0.0
        # u_lon_time[-tlow+1:,:] = 0.0

        u_lon_time[tlow : thig + 1, kmin : kmax + 1] = 0.0
        u_lon_time[-thig:, -kmax:] = 0.0

    else:
        u_lon_time[:tlow, :] = 0.0
        u_lon_time[thig + 1 : -thig, :] = 0.0
        u_lon_time[-tlow + 1 :, :] = 0.0

        u_lon_time[tlow : thig + 1, kmin : kmax + 1] = 0.0
        u_lon_time[-thig : -tlow + 1, -kmax:] = 0.0

    # reconstruct u
    u_retime = np.fft.ifft(u_lon_time, axis=time_index)
    u_re_values = np.fft.ifft(u_retime, axis=lon_index)

    u_re = xr.DataArray(
        data=u_re_values.real,
        dims=u.dims,
        coords=u.coords,
    )

    return u_re
