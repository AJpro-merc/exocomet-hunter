# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic light curves with known ground truth.

Every correctness claim about the detector is checked here first, on data whose
answer we already know, before any real photometry is trusted. The injected
profile is piecewise linear on purpose: for a linear ramp the half-depth
crossing sits exactly halfway along the ramp in time, so the measured
half-durations are exactly half the injected durations, and the two asymmetry
metrics both reduce to ``(egress - ingress) / (egress + ingress)``. That gives
the tests an analytic target instead of a fudge factor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from exocomet.core.types import FloatArray, LightCurveData

#: Kepler long-cadence sampling, in days (~29.4 minutes).
KEPLER_LONG_CADENCE_DAYS = 0.02043


def make_flat_light_curve(
    n_points: int = 2000,
    cadence_days: float = KEPLER_LONG_CADENCE_DAYS,
    noise_sigma: float = 1.0e-4,
    seed: int = 0,
    target_id: str = "SYNTHETIC-1",
) -> LightCurveData:
    """Create a featureless, normalised light curve with Gaussian noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=np.float64) * cadence_days
    flux = np.ones(n_points, dtype=np.float64)
    if noise_sigma > 0:
        flux = flux + rng.normal(0.0, noise_sigma, size=n_points)
    flux_err = np.full(n_points, max(noise_sigma, 1.0e-12), dtype=np.float64)
    return LightCurveData(
        target_id=target_id,
        mission="SYNTHETIC",
        time=time,
        flux=flux,
        flux_err=flux_err,
        meta={"synthetic": True, "noise_sigma": noise_sigma},
    )


def inject_asymmetric_dip(
    time: FloatArray,
    flux: FloatArray,
    t0: float,
    depth: float,
    ingress_duration: float,
    egress_duration: float,
) -> FloatArray:
    """Inject a piecewise-linear dip with independently set ingress and egress.

    Parameters
    ----------
    t0
        Time of the minimum.
    depth
        Fractional depth at the minimum.
    ingress_duration, egress_duration
        Full durations of the falling and rising ramps, in days. Equal values
        give a symmetric event; ``egress > ingress`` is the comet-like case.
    """
    out = flux.copy()

    falling = (time >= t0 - ingress_duration) & (time <= t0)
    out[falling] -= depth * (1.0 - (t0 - time[falling]) / ingress_duration)

    rising = (time > t0) & (time <= t0 + egress_duration)
    out[rising] -= depth * (1.0 - (time[rising] - t0) / egress_duration)

    return out


def inject_symmetric_dip(
    time: FloatArray, flux: FloatArray, t0: float, depth: float, duration: float
) -> FloatArray:
    """Inject a symmetric triangular dip of total width ``2 * duration``."""
    return inject_asymmetric_dip(time, flux, t0, depth, duration, duration)


def expected_a_dur(ingress_duration: float, egress_duration: float) -> float:
    """Analytic duration asymmetry for a piecewise-linear injected dip."""
    return (egress_duration - ingress_duration) / (egress_duration + ingress_duration)


@pytest.fixture
def flat_lc() -> LightCurveData:
    """A pure-noise light curve with no injected events."""
    return make_flat_light_curve()


@pytest.fixture
def make_lc() -> Callable[..., LightCurveData]:
    """Factory fixture for building custom flat light curves."""
    return make_flat_light_curve


@pytest.fixture
def symmetric_lc() -> LightCurveData:
    """A light curve containing one symmetric dip."""
    lc = make_flat_light_curve(seed=1)
    t0 = float(lc.time[lc.n_points // 2])
    flux = inject_symmetric_dip(lc.time, lc.flux, t0=t0, depth=2.0e-3, duration=0.2)
    return lc.with_flux(flux, injected_t0=t0, injected_ingress=0.2, injected_egress=0.2)


@pytest.fixture
def comet_lc() -> LightCurveData:
    """A light curve containing one comet-like dip: steep ingress, slow egress."""
    lc = make_flat_light_curve(seed=2)
    t0 = float(lc.time[lc.n_points // 2])
    flux = inject_asymmetric_dip(
        lc.time, lc.flux, t0=t0, depth=2.0e-3, ingress_duration=0.1, egress_duration=0.5
    )
    return lc.with_flux(flux, injected_t0=t0, injected_ingress=0.1, injected_egress=0.5)


@pytest.fixture
def reversed_comet_lc() -> LightCurveData:
    """A light curve with the opposite asymmetry: slow ingress, steep egress."""
    lc = make_flat_light_curve(seed=3)
    t0 = float(lc.time[lc.n_points // 2])
    flux = inject_asymmetric_dip(
        lc.time, lc.flux, t0=t0, depth=2.0e-3, ingress_duration=0.5, egress_duration=0.1
    )
    return lc.with_flux(flux, injected_t0=t0, injected_ingress=0.5, injected_egress=0.1)
