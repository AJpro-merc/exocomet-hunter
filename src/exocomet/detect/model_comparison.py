# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Parametric model comparison: is this dip better described as a comet?

The asymmetry ratios in :mod:`exocomet.detect.asymmetry` are interpretable but
non-parametric — they compress a whole event into two numbers and say nothing
about how well any physical shape actually describes the data. The established
approach in the exocomet literature (Kennedy et al. 2019) is instead to fit two
competing profiles to each event and compare them:

**Symmetric** — a Gaussian dip, the null hypothesis. A planet, an eclipsing
binary, or a noise excursion is described well by it.

**Comet** — a Gaussian ingress joined at the minimum to an exponential egress.
This is the standard optically-thin dust-tail parameterisation: the obscuration
builds as the head approaches and then decays as the tail sweeps out.

Because the comet profile has one extra free parameter it can always fit at
least as well, so the models are ranked by BIC, which charges a penalty for that
extra freedom. ``delta_bic = BIC(symmetric) - BIC(comet)`` is positive when the
comet profile earns its complexity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from exocomet.core.types import (
    DipEvent,
    FloatArray,
    ModelComparison,
    ModelFit,
)

__all__ = ["comet_profile", "compare_models", "symmetric_profile"]

#: Lower bound on the fitted widths, in days. Prevents the optimiser from
#: collapsing a profile onto a single cadence.
_MIN_WIDTH_DAYS = 1.0e-4


def symmetric_profile(
    t: FloatArray, t0: float, depth: float, width: float, baseline: float
) -> FloatArray:
    """Gaussian dip centred on ``t0``.

    Parameters
    ----------
    t
        Times at which to evaluate the model.
    t0
        Time of minimum.
    depth
        Fractional depth at minimum.
    width
        Gaussian standard deviation in days.
    baseline
        Out-of-dip flux level.
    """
    width = max(width, _MIN_WIDTH_DAYS)
    return np.asarray(baseline - depth * np.exp(-0.5 * ((t - t0) / width) ** 2), dtype=np.float64)


def comet_profile(
    t: FloatArray,
    t0: float,
    depth: float,
    ingress_width: float,
    egress_tau: float,
    baseline: float,
) -> FloatArray:
    """Gaussian ingress joined to an exponential egress at ``t0``.

    The two halves meet continuously at the minimum, both reaching
    ``baseline - depth`` there.

    Parameters
    ----------
    ingress_width
        Gaussian standard deviation of the falling side, in days.
    egress_tau
        E-folding timescale of the recovering side, in days. A tail that clears
        slowly has a large ``egress_tau`` relative to ``ingress_width`` — the
        quantitative statement of "comet-shaped".
    """
    ingress_width = max(ingress_width, _MIN_WIDTH_DAYS)
    egress_tau = max(egress_tau, _MIN_WIDTH_DAYS)

    out = np.empty_like(t, dtype=np.float64)
    before = t <= t0
    out[before] = baseline - depth * np.exp(-0.5 * ((t[before] - t0) / ingress_width) ** 2)
    out[~before] = baseline - depth * np.exp(-(t[~before] - t0) / egress_tau)
    return out


def _chi2(residuals: FloatArray) -> float:
    return float(np.sum(residuals**2))


def _fit(
    name: str,
    model: object,
    x0: list[float],
    bounds: tuple[list[float], list[float]],
    t: FloatArray,
    y: FloatArray,
    yerr: FloatArray,
) -> ModelFit:
    """Run a bounded least-squares fit and package the result."""

    def residuals(params: np.ndarray) -> np.ndarray:
        return (model(t, *params) - y) / yerr  # type: ignore[operator]

    try:
        result = least_squares(residuals, x0=x0, bounds=bounds, method="trf", max_nfev=2000)
        converged = bool(result.success)
        params = [float(v) for v in result.x]
        chi2 = _chi2(np.asarray(result.fun, dtype=np.float64))
    except (ValueError, RuntimeError):  # pragma: no cover - pathological windows
        converged = False
        params = list(x0)
        chi2 = float("inf")

    names = (
        ["t0", "depth", "width", "baseline"]
        if name == "symmetric"
        else ["t0", "depth", "ingress_width", "egress_tau", "baseline"]
    )
    return ModelFit(
        name=name,
        params=dict(zip(names, params, strict=True)),
        chi2=chi2,
        n_params=len(params),
        n_points=int(t.size),
        converged=converged,
    )


def compare_models(
    time: FloatArray,
    flux: FloatArray,
    flux_err: FloatArray,
    event: DipEvent,
    baseline_level: float,
    padding_factor: float = 1.5,
    delta_bic_threshold: float = 10.0,
) -> ModelComparison | None:
    """Fit both profiles to one event and compare them by BIC.

    The fitting region is the event window widened by ``padding_factor`` so that
    the recovering tail — which by construction extends past the point where the
    flux came back within the detection threshold — is included rather than
    truncated.

    Parameters
    ----------
    baseline_level
        Local out-of-dip level, used to seed the fit and to bound the depth.
    padding_factor
        How much wider than the detected window to fit, as a multiple of the
        window duration.
    delta_bic_threshold
        Evidence threshold carried onto the returned comparison, so that
        :attr:`ModelComparison.favors_comet` and the flagging decision in
        :mod:`exocomet.detect.scoring` share one configured value.

    Returns
    -------
    ModelComparison or None
        ``None`` when the padded window holds too few cadences to constrain five
        free parameters.
    """
    window = event.window
    span = max(window.duration_days, 1.0e-6)
    pad = 0.5 * span * (padding_factor - 1.0)
    lo_time, hi_time = window.start_time - pad, window.end_time + pad

    mask = (time >= lo_time) & (time <= hi_time)
    n = int(np.count_nonzero(mask))
    if n < 8:
        return None

    t = time[mask]
    y = flux[mask]
    yerr = np.where(flux_err[mask] > 0, flux_err[mask], 1.0)

    depth0 = max(event.depth, 1.0e-9)
    width0 = max(span / 4.0, _MIN_WIDTH_DAYS * 10.0)

    symmetric = _fit(
        "symmetric",
        symmetric_profile,
        x0=[event.t_min, depth0, width0, baseline_level],
        bounds=(
            [lo_time, 0.0, _MIN_WIDTH_DAYS, baseline_level - 10.0 * depth0],
            [hi_time, 10.0 * depth0, span, baseline_level + 10.0 * depth0],
        ),
        t=t,
        y=y,
        yerr=yerr,
    )

    comet = _fit(
        "comet",
        comet_profile,
        x0=[event.t_min, depth0, width0, width0, baseline_level],
        bounds=(
            [lo_time, 0.0, _MIN_WIDTH_DAYS, _MIN_WIDTH_DAYS, baseline_level - 10.0 * depth0],
            [hi_time, 10.0 * depth0, span, 10.0 * span, baseline_level + 10.0 * depth0],
        ),
        t=t,
        y=y,
        yerr=yerr,
    )

    return ModelComparison(
        symmetric=symmetric, comet=comet, delta_bic_threshold=delta_bic_threshold
    )
