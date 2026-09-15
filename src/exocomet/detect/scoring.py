"""Significance estimation for measured events.

An asymmetry value on its own means little: a shallow dip in noisy photometry can
look lopsided by chance. The uncertainty on the asymmetry is therefore estimated
by bootstrap — perturbing the in-event flux within its own error bars and
re-measuring — and an event is only flagged when its asymmetry is large compared
with that uncertainty *and* the two independent asymmetry metrics agree about
which side is the long one.
"""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle

from exocomet.core.config import AsymmetryConfig, ScoringConfig
from exocomet.core.types import DipEvent, EventScore, FloatArray, LightCurveData

from exocomet.detect.asymmetry import measure_event
from exocomet.detect.model_comparison import compare_models

__all__ = ["bootstrap_a_dur_sigma", "periodicity_flag", "score_event"]


def bootstrap_a_dur_sigma(
    time: FloatArray,
    flux: FloatArray,
    flux_err: FloatArray,
    baseline: FloatArray,
    event: DipEvent,
    draws: int,
    rng: np.random.Generator,
    config: AsymmetryConfig | None = None,
) -> tuple[float, int]:
    """Estimate the uncertainty on an event's duration asymmetry.

    Each draw adds Gaussian noise, scaled by the per-cadence error bars, to the
    cadences in and immediately around the event, then repeats the full
    ingress/egress measurement. The spread of the resulting ``a_dur`` values is
    the quantity a single measurement cannot provide.

    Returns
    -------
    sigma, n_valid
        Standard deviation of the bootstrap ``a_dur`` distribution and the
        number of draws that produced a measurable event.
    """
    cfg = config or AsymmetryConfig()
    window = event.window

    lo = max(window.start_index - 1, 0)
    hi = min(window.end_index + 1, flux.size - 1)
    original = flux[lo : hi + 1].copy()
    errors = flux_err[lo : hi + 1]

    work = flux.copy()
    samples: list[float] = []
    for _ in range(draws):
        work[lo : hi + 1] = original + rng.normal(0.0, errors)
        perturbed = measure_event(time, work, baseline, window, event.target_id, cfg)
        if perturbed.asymmetry is not None:
            samples.append(perturbed.asymmetry.a_dur)

    if len(samples) < 2:
        return float("nan"), len(samples)
    return float(np.std(samples, ddof=1)), len(samples)


def periodicity_flag(
    lc: LightCurveData, config: ScoringConfig | None = None
) -> tuple[bool, float | None]:
    """Test the whole light curve for a dominant periodic signal.

    A strong period is not proof that an event is an eclipsing binary, but it is
    a reason to rank a candidate lower: genuine exocomet transits are aperiodic,
    whereas binaries and planets repeat. The result is used as a ranking
    penalty, never as a hard rejection.

    Returns
    -------
    periodic, period_days
        Whether a peak exceeded the configured false-alarm threshold, and the
        period of the strongest peak (``None`` when the test cannot be run).
    """
    cfg = config or ScoringConfig()
    if lc.n_points < 10 or lc.baseline_days <= 0:
        return False, None

    model = LombScargle(lc.time, lc.flux, lc.flux_err)
    frequency, power = model.autopower(
        minimum_frequency=1.0 / cfg.periodicity_max_period_days,
        maximum_frequency=1.0 / cfg.periodicity_min_period_days,
    )
    if power.size == 0:
        return False, None

    best = int(np.argmax(power))
    best_period = float(1.0 / frequency[best])
    try:
        fap = float(model.false_alarm_probability(power[best]))
    except (ValueError, ZeroDivisionError):  # pragma: no cover - degenerate photometry
        return False, best_period

    return fap < cfg.periodicity_fap_threshold, best_period


def score_event(
    event: DipEvent,
    lc: LightCurveData,
    baseline: FloatArray,
    config: ScoringConfig | None = None,
    asymmetry_config: AsymmetryConfig | None = None,
    rng: np.random.Generator | None = None,
    periodic: bool = False,
    period_days: float | None = None,
) -> EventScore:
    """Score a measured event for asymmetry significance.

    Parameters
    ----------
    periodic, period_days
        Result of :func:`periodicity_flag` for the parent light curve. Passed in
        rather than recomputed because the periodogram is a property of the
        whole light curve, not of one event.

    Returns
    -------
    EventScore
        ``flagged`` is ``True`` only when the asymmetry exceeds the significance
        threshold and (when configured) the two metrics agree in sign.
    """
    cfg = config or ScoringConfig()
    generator = rng if rng is not None else np.random.default_rng()

    baseline_level = float(
        np.median(baseline[event.window.start_index : event.window.end_index + 1])
    )
    model = compare_models(lc.time, lc.flux, lc.flux_err, event, baseline_level)

    if event.asymmetry is None:
        return EventScore(
            a_dur_sigma=float("nan"),
            significance=0.0,
            signs_agree=False,
            flagged=False,
            periodic=periodic,
            period_days=period_days,
            model=model,
        )

    sigma, _ = bootstrap_a_dur_sigma(
        lc.time,
        lc.flux,
        lc.flux_err,
        baseline,
        event,
        cfg.bootstrap_draws,
        generator,
        asymmetry_config,
    )

    a_dur = event.asymmetry.a_dur
    if not np.isfinite(sigma):
        significance = 0.0
    elif sigma <= 0.0:
        # Noiseless photometry: any non-zero asymmetry is infinitely significant.
        significance = float("inf") if a_dur != 0.0 else 0.0
    else:
        significance = abs(a_dur) / sigma

    signs_agree = event.asymmetry.signs_agree
    flagged = significance >= cfg.significance_threshold
    if cfg.require_sign_agreement:
        flagged = flagged and signs_agree

    return EventScore(
        a_dur_sigma=sigma,
        significance=significance,
        signs_agree=signs_agree,
        flagged=bool(flagged),
        periodic=periodic,
        period_days=period_days,
        model=model,
    )
