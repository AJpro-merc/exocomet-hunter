"""Removal of instrumental and stellar trends ahead of detection.

Spacecraft photometry drifts: thermal settling after a safe-mode recovery, focus
changes, pointing jitter and intrinsic stellar variability all impose slow trends
that would otherwise be mistaken for broad, lopsided dips. Detrending divides
those out.

The central tension is that the filter which removes trends can also remove the
signal. A window that is too short follows a real dip down and erases it; one
that is too long leaves trends behind and manufactures false positives. The
default window is therefore set to several times the longest dip duration we
expect to find — for Kepler long cadence, ``window_length=401`` is roughly eight
days against the day-scale events reported in the literature.

Upward outliers are clipped before fitting, downward ones never are: a bright
spike is a flare or a cosmic ray, while a faint excursion is the thing we are
looking for.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter

from exocomet.core.config import DetrendConfig
from exocomet.core.types import FloatArray, LightCurveData
from exocomet.detect.baseline import MAD_TO_SIGMA, days_to_cadences, odd_window

__all__ = ["SavGolDetrender", "SplineDetrender", "detrend_savgol", "detrend_spline"]


def _clip_upper_outliers(flux: FloatArray, sigma: float) -> tuple[FloatArray, int]:
    """Replace strong upward excursions with the median, leaving dips untouched.

    Returns the cleaned series and the number of points replaced, so that the
    count can be logged rather than silently discarded.
    """
    median = float(np.median(flux))
    scatter = MAD_TO_SIGMA * float(np.median(np.abs(flux - median)))
    if scatter <= 0.0:
        return flux.copy(), 0

    mask = flux > median + sigma * scatter
    cleaned = flux.copy()
    cleaned[mask] = median
    return cleaned, int(np.count_nonzero(mask))


def detrend_savgol(
    lc: LightCurveData, config: DetrendConfig | None = None
) -> LightCurveData:
    """Detrend by dividing out a Savitzky-Golay fit to the light curve.

    Parameters
    ----------
    lc
        Normalised light curve.
    config
        Detrending parameters; defaults are used when omitted.

    Returns
    -------
    LightCurveData
        A copy whose flux has been divided by the fitted trend, so the
        out-of-dip level sits at approximately 1.0. Metadata records the method,
        the window used, and how many upward outliers were clipped.
    """
    cfg = config or DetrendConfig()
    window = odd_window(days_to_cadences(cfg.window_days, lc.cadence_days), lc.n_points)
    if window <= cfg.polyorder:
        # Too few cadences to fit the requested polynomial; nothing to remove.
        return lc.with_flux(lc.flux.copy(), detrend_method="none", detrend_window=window)

    cleaned, n_clipped = _clip_upper_outliers(lc.flux, cfg.upper_sigma_clip)
    trend = savgol_filter(cleaned, window_length=window, polyorder=cfg.polyorder)
    trend = np.where(np.abs(trend) < 1.0e-12, 1.0, trend)

    return lc.with_flux(
        np.asarray(lc.flux / trend, dtype=np.float64),
        detrend_method="savgol",
        detrend_window=window,
        detrend_polyorder=cfg.polyorder,
        detrend_clipped_points=n_clipped,
    )


def detrend_spline(
    lc: LightCurveData, config: DetrendConfig | None = None
) -> LightCurveData:
    """Detrend by dividing out a least-squares spline with regularly spaced knots.

    Provided as the documented alternative to :func:`detrend_savgol`: it yields
    smoother residuals but is more sensitive to where knots happen to fall
    relative to a real event. Comparing the two is part of the methodology
    write-up rather than a runtime choice.
    """
    cfg = config or DetrendConfig()
    cleaned, n_clipped = _clip_upper_outliers(lc.flux, cfg.upper_sigma_clip)

    span = lc.baseline_days
    n_knots = int(span / cfg.spline_knot_spacing_days) - 1
    if n_knots < 1 or lc.n_points < 8:
        return lc.with_flux(lc.flux.copy(), detrend_method="none")

    interior = np.linspace(lc.time[0], lc.time[-1], n_knots + 2)[1:-1]
    spline = LSQUnivariateSpline(lc.time, cleaned, interior, k=3)
    trend = np.asarray(spline(lc.time), dtype=np.float64)
    trend = np.where(np.abs(trend) < 1.0e-12, 1.0, trend)

    return lc.with_flux(
        np.asarray(lc.flux / trend, dtype=np.float64),
        detrend_method="spline",
        detrend_knot_spacing_days=cfg.spline_knot_spacing_days,
        detrend_clipped_points=n_clipped,
    )


class SavGolDetrender:
    """Savitzky-Golay detrender satisfying the :class:`~exocomet.core.interfaces.Detrender` protocol."""

    def __init__(self, config: DetrendConfig | None = None) -> None:
        self.config = config or DetrendConfig()

    def detrend(self, lc: LightCurveData) -> LightCurveData:
        """Divide out a Savitzky-Golay trend."""
        return detrend_savgol(lc, self.config)


class SplineDetrender:
    """Spline detrender satisfying the :class:`~exocomet.core.interfaces.Detrender` protocol."""

    def __init__(self, config: DetrendConfig | None = None) -> None:
        self.config = config or DetrendConfig()

    def detrend(self, lc: LightCurveData) -> LightCurveData:
        """Divide out a least-squares spline trend."""
        return detrend_spline(lc, self.config)
