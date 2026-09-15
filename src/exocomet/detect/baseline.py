"""Local baseline and robust noise estimation.

Both quantities are computed with running medians rather than running means so
that the dips we are trying to find do not pull the reference level down with
them: a median is insensitive to a minority of strongly deviant points, which is
exactly what an in-transit cadence is.

Windows are **truncated** at the ends of the series rather than padded. Padding
with a repeated edge value is the usual default, and it is actively harmful
here: repeating one noisy sample across half a window both biases the baseline
and collapses the scatter estimate, because most of the window then holds
identical values. In testing, that inflated the apparent significance of pure
noise at the array edges to 30-70 sigma — far above a genuine injected dip — and
would have manufactured false candidates at every sector boundary. A truncated
window is noisier near the edges but remains unbiased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exocomet.core.types import FloatArray

__all__ = [
    "MAD_TO_SIGMA",
    "days_to_cadences",
    "local_noise",
    "odd_window",
    "rolling_baseline",
    "rolling_median",
]

#: Scale factor converting a median absolute deviation into a Gaussian-equivalent
#: standard deviation.
MAD_TO_SIGMA = 1.4826


def days_to_cadences(days: float, cadence_days: float) -> int:
    """Convert a duration in days to a whole number of cadences.

    Windows are specified in physical time (days) so that the same
    configuration means the same thing on Kepler's 29.4-minute cadence and on
    TESS's 20 s to 1800 s products; this converts to the cadence count each
    windowing function actually needs, at runtime, using the light curve's own
    measured spacing.

    Parameters
    ----------
    days
        Requested window duration, in days.
    cadence_days
        Median spacing between consecutive cadences, in days (see
        :attr:`~exocomet.core.types.LightCurveData.cadence_days`).

    Returns
    -------
    int
        At least 1 cadence; :func:`odd_window` handles oddness and clamping to
        the series length.
    """
    if cadence_days <= 0.0:
        raise ValueError(f"cadence_days must be > 0, got {cadence_days}")
    return max(round(days / cadence_days), 1)


def odd_window(window: int, n_points: int) -> int:
    """Clamp ``window`` to an odd value no larger than the series length.

    Parameters
    ----------
    window
        Requested window size in cadences.
    n_points
        Length of the series the window will be applied to.

    Returns
    -------
    int
        An odd window size in ``[1, n_points]``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    clamped = min(window, n_points)
    if clamped % 2 == 0:
        clamped -= 1
    return max(clamped, 1)


def rolling_median(values: FloatArray, window: int) -> FloatArray:
    """Centred running median over real samples only, with truncated edges.

    Unlike :func:`scipy.ndimage.median_filter`, no synthetic padding is
    introduced: near the ends of the series the window simply contains fewer
    points. See the module docstring for why that distinction matters.
    """
    size = odd_window(window, values.size)
    series = pd.Series(values, dtype="float64")
    rolled = series.rolling(window=size, center=True, min_periods=1).median()
    return np.asarray(rolled.to_numpy(), dtype=np.float64)


def rolling_baseline(flux: FloatArray, window: int) -> FloatArray:
    """Estimate the local out-of-dip flux level with a running median.

    Parameters
    ----------
    flux
        Normalised flux series.
    window
        Window length in cadences; coerced to odd and clamped to the series
        length.

    Returns
    -------
    numpy.ndarray
        Baseline flux, same shape as ``flux``.
    """
    return rolling_median(flux, window)


def local_noise(flux: FloatArray, baseline: FloatArray, window: int) -> FloatArray:
    """Estimate per-cadence scatter from the running median absolute deviation.

    The residual ``flux - baseline`` is locally zero-centred by construction, so
    a running median of its absolute value is a local MAD; scaling by
    :data:`MAD_TO_SIGMA` puts it on the same footing as a standard deviation for
    Gaussian noise.

    A floor equal to the global scatter times ``1e-3`` is applied so that
    perfectly flat stretches (synthetic data, or a masked region) cannot produce
    a zero denominator downstream.

    Parameters
    ----------
    flux
        Normalised flux series.
    baseline
        Local baseline from :func:`rolling_baseline`.
    window
        Window length in cadences.

    Returns
    -------
    numpy.ndarray
        Robust per-cadence sigma, same shape as ``flux``, strictly positive.
    """
    residual = np.abs(flux - baseline)
    mad = rolling_median(residual, window)
    sigma = MAD_TO_SIGMA * mad

    global_sigma = MAD_TO_SIGMA * float(np.median(residual))
    floor = max(global_sigma * 1.0e-3, np.finfo(np.float64).tiny)
    return np.maximum(sigma, floor)
