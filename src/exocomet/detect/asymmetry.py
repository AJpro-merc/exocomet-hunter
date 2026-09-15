# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Measurement of dip asymmetry — the discriminating step of the pipeline.

A planet is an opaque disc: it covers the star and uncovers it at the same rate,
so its transit is symmetric about mid-transit. A comet drags a tail of dust
behind it, so the obscuration builds quickly and clears slowly (or, for a
leading-tail geometry, the reverse). Quantifying that lopsidedness is what
separates a comet candidate from the far more numerous planets, eclipsing
binaries and noise excursions.

Each side of a dip is measured independently between the half-depth crossing and
the dip minimum, giving a duration and a slope per side. Two asymmetry metrics
are formed from those four numbers; both are constructed to be **positive for a
steep-ingress, slow-egress (comet-like) event**, so requiring them to agree in
sign is a statement about physical consistency rather than an arithmetic
coincidence.
"""

from __future__ import annotations

import numpy as np

from exocomet.core.config import AsymmetryConfig
from exocomet.core.types import (
    AsymmetryResult,
    DipEvent,
    EventWindow,
    FloatArray,
    SideFit,
)

__all__ = [
    "compute_asymmetry",
    "locate_minimum",
    "measure_event",
    "measure_side",
    "ols_slope",
]


def ols_slope(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Fit ``y = a + b x`` by ordinary least squares.

    Parameters
    ----------
    x, y
        Matching one-dimensional arrays with at least three points.

    Returns
    -------
    slope, slope_err
        The gradient and its standard error. ``slope_err`` is ``inf`` when the
        residual variance cannot be estimated (fewer than three points, or no
        spread in ``x``).
    """
    n = x.size
    if n < 3:
        return float("nan"), float("inf")

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    s_xx = float(np.sum(dx * dx))
    if s_xx <= 0.0:
        return float("nan"), float("inf")

    slope = float(np.sum(dx * (y - y_mean)) / s_xx)
    residuals = y - (y_mean + slope * dx)
    dof = n - 2
    sigma_sq = float(np.sum(residuals * residuals) / dof)
    slope_err = float(np.sqrt(sigma_sq / s_xx))
    return slope, slope_err


def locate_minimum(
    time: FloatArray, flux: FloatArray, window: EventWindow
) -> tuple[float, float, int]:
    """Locate the deepest point of an event to sub-cadence precision.

    A parabola is fitted through the deepest cadence and its two neighbours; its
    vertex gives the interpolated minimum. If the three points are collinear, or
    the vertex falls outside the central cadence, the discrete minimum is
    returned unchanged.

    Returns
    -------
    t_min, f_min, index
        Interpolated time and flux of the minimum, plus the index of the deepest
        cadence.
    """
    start, end = window.start_index, window.end_index
    segment = flux[start : end + 1]
    idx = int(start + np.argmin(segment))

    if idx <= 0 or idx >= time.size - 1:
        return float(time[idx]), float(flux[idx]), idx

    f_prev, f_here, f_next = float(flux[idx - 1]), float(flux[idx]), float(flux[idx + 1])
    denominator = f_prev - 2.0 * f_here + f_next
    if denominator <= 0.0:
        return float(time[idx]), float(flux[idx]), idx

    delta = 0.5 * (f_prev - f_next) / denominator
    if not -1.0 < delta < 1.0:
        return float(time[idx]), float(flux[idx]), idx

    dt = 0.5 * float(time[idx + 1] - time[idx - 1])
    t_min = float(time[idx]) + delta * dt
    f_min = f_here - 0.25 * (f_prev - f_next) * delta
    return t_min, f_min, idx


def _half_depth_crossing(
    time: FloatArray,
    flux: FloatArray,
    idx_min: int,
    bound: int,
    half_level: float,
    direction: int,
) -> tuple[float, int] | None:
    """Find where the flux crosses ``half_level`` on one side of the minimum.

    Walks outward from the dip minimum until the flux rises back above
    ``half_level``, then linearly interpolates between the bracketing cadences.

    Parameters
    ----------
    direction
        ``-1`` to walk towards earlier times (ingress), ``+1`` for later
        (egress).

    Returns
    -------
    tuple or None
        ``(crossing_time, index_of_first_point_inside)``, or ``None`` if the
        flux never recovers to the half-depth level within the window.
    """
    i = idx_min
    while (direction < 0 and i - 1 >= bound) or (direction > 0 and i + 1 <= bound):
        nxt = i + direction
        if flux[nxt] >= half_level:
            f_in, f_out = float(flux[i]), float(flux[nxt])
            t_in, t_out = float(time[i]), float(time[nxt])
            if f_out == f_in:
                return t_out, i
            frac = (half_level - f_in) / (f_out - f_in)
            return t_in + frac * (t_out - t_in), i
        i = nxt
    return None


def measure_side(
    time: FloatArray,
    flux: FloatArray,
    window: EventWindow,
    idx_min: int,
    t_min: float,
    baseline_level: float,
    depth: float,
    side: str,
    config: AsymmetryConfig | None = None,
    search_bound: int | None = None,
) -> SideFit | None:
    """Measure the duration and slope of one side of a dip.

    Parameters
    ----------
    side
        ``"ingress"`` for the falling side, ``"egress"`` for the rising side.
    baseline_level
        Local out-of-dip flux level for this event.
    depth
        ``baseline_level - f_min``, a positive number.
    search_bound
        Index beyond the detected window out to which the half-depth crossing
        may be sought. Defaults to the window edge. Supplying a wider bound is
        what allows dips whose wings extend past the significance threshold to
        be measured at all.

    Returns
    -------
    SideFit or None
        ``None`` when the half-depth level is never crossed within the search
        region, or when too few cadences can be gathered to fit a slope — in
        either case the event is reported as unmeasurable rather than scored on
        noise.
    """
    cfg = config or AsymmetryConfig()
    if depth <= 0.0:
        return None

    half_level = baseline_level - cfg.half_depth_fraction * depth
    direction = -1 if side == "ingress" else 1
    if search_bound is None:
        search_bound = window.start_index if direction < 0 else window.end_index

    crossing = _half_depth_crossing(time, flux, idx_min, search_bound, half_level, direction)
    if crossing is None:
        return None
    t_half, idx_inside = crossing

    # A steep side can cross the half-depth level within a cadence or two,
    # leaving too few points to constrain a gradient. Reach a little further
    # out -- still within the search region -- rather than discarding the event.
    if direction < 0:
        lo, hi = idx_inside, idx_min
        while hi - lo + 1 < cfg.min_points_per_side and lo > search_bound:
            lo -= 1
    else:
        lo, hi = idx_min, idx_inside
        while hi - lo + 1 < cfg.min_points_per_side and hi < search_bound:
            hi += 1

    x = time[lo : hi + 1]
    y = flux[lo : hi + 1]
    if x.size < cfg.min_points_per_side:
        return None

    slope, slope_err = ols_slope(x, y)
    if not np.isfinite(slope):
        return None

    duration = abs(t_min - t_half)
    if duration <= 0.0:
        return None

    return SideFit(
        duration_days=float(duration),
        slope=float(slope),
        slope_err=float(slope_err),
        n_points=int(x.size),
        half_depth_time=float(t_half),
    )


def compute_asymmetry(ingress: SideFit, egress: SideFit) -> AsymmetryResult:
    """Combine two side measurements into normalised asymmetry metrics.

    Both metrics live in ``[-1, 1]`` and are positive for the comet-like case of
    a fast fade followed by a slow recovery. See
    :class:`exocomet.core.types.AsymmetryResult` for the exact definitions.
    """
    dur_sum = egress.duration_days + ingress.duration_days
    a_dur = (egress.duration_days - ingress.duration_days) / dur_sum if dur_sum > 0 else 0.0

    ingress_mag = abs(ingress.slope)
    egress_mag = abs(egress.slope)
    slope_sum = ingress_mag + egress_mag
    a_slope = (ingress_mag - egress_mag) / slope_sum if slope_sum > 0 else 0.0

    return AsymmetryResult(
        a_dur=float(a_dur), a_slope=float(a_slope), ingress=ingress, egress=egress
    )


def measure_event(
    time: FloatArray,
    flux: FloatArray,
    baseline: FloatArray,
    window: EventWindow,
    target_id: str,
    config: AsymmetryConfig | None = None,
) -> DipEvent:
    """Turn a candidate window into a fully measured :class:`DipEvent`.

    The local baseline level is taken as the median of the baseline series
    across the window, which is more stable than reading a single cadence.
    """
    cfg = config or AsymmetryConfig()

    t_min, f_min, idx_min = locate_minimum(time, flux, window)
    baseline_level = float(np.median(baseline[window.start_index : window.end_index + 1]))
    depth = baseline_level - f_min

    # Search beyond the detection window: it was drawn where the dip exceeds the
    # significance threshold, which is by construction narrower than the event.
    pad = int(round(window.n_points * cfg.search_padding_factor))
    lo_bound = max(window.start_index - pad, 0)
    hi_bound = min(window.end_index + pad, time.size - 1)

    ingress = measure_side(
        time, flux, window, idx_min, t_min, baseline_level, depth, "ingress", cfg, lo_bound
    )
    egress = measure_side(
        time, flux, window, idx_min, t_min, baseline_level, depth, "egress", cfg, hi_bound
    )

    if ingress is None or egress is None:
        missing = "ingress" if ingress is None else "egress"
        return DipEvent(
            target_id=target_id,
            t_min=t_min,
            f_min=f_min,
            depth=depth,
            window=window,
            asymmetry=None,
            unmeasurable_reason=f"{missing} side could not be fitted within the event window",
        )

    return DipEvent(
        target_id=target_id,
        t_min=t_min,
        f_min=f_min,
        depth=depth,
        window=window,
        asymmetry=compute_asymmetry(ingress, egress),
    )
