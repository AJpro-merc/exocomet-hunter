"""Injection and recovery: measuring what the pipeline can actually find.

A detection count on its own is close to meaningless. "I found three events" only
becomes a scientific statement once paired with "and I would have found 90% of
events like these, but only 20% of shallower ones" — that is the difference
between a list of candidates and a result someone else can build an occurrence
rate on.

The method is to inject synthetic comet-shaped transits of known depth,
duration and asymmetry into a light curve, run the unmodified detector, and
record whether each injection came back. Sweeping a grid of injected parameters
produces a completeness map: the recovery fraction as a function of signal
strength, and with it the pipeline's detection floor.

Injections go into **real** photometry wherever possible, so that genuine
systematics, gaps and stellar variability are part of the test rather than
idealised away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exocomet.core.config import PipelineConfig
from exocomet.core.types import FloatArray, LightCurveData

if TYPE_CHECKING:  # pragma: no cover
    from exocomet.core.interfaces import Detector

__all__ = [
    "InjectionResult",
    "RecoveryGrid",
    "inject_comet_transit",
    "inject_flare",
    "inject_starspot_modulation",
    "run_injection_recovery",
]


def inject_comet_transit(
    time: FloatArray,
    flux: FloatArray,
    t0: float,
    depth: float,
    ingress_duration: float,
    egress_duration: float,
) -> FloatArray:
    """Inject one piecewise-linear comet-shaped transit.

    A linear ramp is used rather than the Gaussian/exponential profile that the
    model-comparison step fits. That mismatch is deliberate: injecting exactly
    the model being fitted would flatter the pipeline, whereas a different shape
    tests whether it recovers events it was not tuned for.

    Parameters
    ----------
    t0
        Epoch of minimum light.
    depth
        Fractional depth at minimum.
    ingress_duration, egress_duration
        Durations of the falling and rising ramps in days. ``egress > ingress``
        is the comet-like configuration.
    """
    out = flux.copy()

    falling = (time >= t0 - ingress_duration) & (time <= t0)
    out[falling] -= depth * (1.0 - (t0 - time[falling]) / ingress_duration)

    rising = (time > t0) & (time <= t0 + egress_duration)
    out[rising] -= depth * (1.0 - (time[rising] - t0) / egress_duration)

    return out


def inject_flare(
    time: FloatArray,
    flux: FloatArray,
    t0: float,
    amplitude: float,
    rise_duration: float,
    decay_duration: float,
) -> FloatArray:
    """Inject one stellar-flare-shaped *brightening* (Next Steps D1 false-positive lookalike).

    A flare is the sign-flipped opposite of a comet transit: a fast linear
    rise followed by a slower exponential decay, adding flux rather than
    removing it. Real flares are the most common non-comet source of a sharp,
    asymmetric excursion in real photometry, so a classifier that has never
    seen one has a large blind spot.

    Parameters
    ----------
    t0
        Time of peak brightness.
    amplitude
        Fractional flux added at the peak (positive).
    rise_duration
        Duration of the linear rise, in days.
    decay_duration
        e-folding time of the exponential decay, in days.
    """
    out = flux.copy()

    rising = (time >= t0 - rise_duration) & (time <= t0)
    out[rising] += amplitude * (1.0 - (t0 - time[rising]) / rise_duration)

    decaying = time > t0
    out[decaying] += amplitude * np.exp(-(time[decaying] - t0) / decay_duration)

    return out


def inject_starspot_modulation(
    time: FloatArray,
    flux: FloatArray,
    period_days: float,
    amplitude: float,
    phase: float = 0.0,
) -> FloatArray:
    """Inject sinusoidal starspot-rotation modulation (Next Steps D1 false-positive lookalike).

    Real stellar rotation with an uneven spot distribution produces a smooth,
    periodic brightness variation -- easily confused with a real signal by
    any detector that does not already know what a quiet star looks like.
    Unlike the other injectors here, this one modulates the *whole* light
    curve rather than a single localised event.

    Parameters
    ----------
    period_days
        Rotation period, in days (Next Steps D1 range: 0.5-30 d).
    amplitude
        Fractional peak-to-peak modulation depth (Next Steps D1 range:
        100-5000 ppm, i.e. 1e-4 to 5e-3).
    phase
        Phase offset in radians.
    """
    return flux + amplitude * np.sin(2.0 * np.pi * time / period_days + phase)


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of a single injection."""

    t0: float
    depth: float
    ingress_duration: float
    egress_duration: float
    detected: bool
    flagged: bool
    recovered_t0: float | None
    recovered_a_dur: float | None
    delta_bic: float

    @property
    def depth_ppm(self) -> float:
        """Injected depth in parts per million."""
        return self.depth * 1.0e6


@dataclass(frozen=True)
class RecoveryGrid:
    """Aggregated results of an injection-recovery experiment."""

    results: tuple[InjectionResult, ...]

    @property
    def n_injected(self) -> int:
        """Total number of injections performed."""
        return len(self.results)

    def completeness(self, flagged_only: bool = True) -> float:
        """Overall recovery fraction.

        Parameters
        ----------
        flagged_only
            When ``True`` (the default) an injection counts as recovered only if
            it was both detected *and* flagged as significantly asymmetric — the
            criterion an actual search result would have to meet. When ``False``,
            mere detection is enough.
        """
        if not self.results:
            return 0.0
        hits = sum(1 for r in self.results if (r.flagged if flagged_only else r.detected))
        return hits / len(self.results)

    def completeness_by_depth(self, flagged_only: bool = True) -> dict[float, float]:
        """Recovery fraction grouped by injected depth."""
        by_depth: dict[float, list[bool]] = {}
        for r in self.results:
            outcome = r.flagged if flagged_only else r.detected
            by_depth.setdefault(r.depth, []).append(outcome)
        return {d: float(np.mean(v)) for d, v in sorted(by_depth.items())}

    def detection_floor(self, target_completeness: float = 0.5) -> float | None:
        """Shallowest injected depth reaching ``target_completeness``.

        Returns
        -------
        float or None
            The depth in fractional units, or ``None`` if no tested depth
            reached the requested recovery fraction.
        """
        for depth, frac in self.completeness_by_depth().items():
            if frac >= target_completeness:
                return depth
        return None


def run_injection_recovery(
    lc: LightCurveData,
    detector: Detector,
    depths: list[float],
    ingress_duration: float = 0.1,
    egress_duration: float = 0.5,
    n_per_depth: int = 20,
    match_tolerance_days: float = 0.5,
    config: PipelineConfig | None = None,
    rng: np.random.Generator | None = None,
) -> RecoveryGrid:
    """Inject transits across a grid of depths and measure the recovery fraction.

    Each injection is placed at a random epoch, clear of the edges of the series
    where detection is deliberately suppressed. The detector is then run on the
    injected light curve and an injection counts as recovered when a returned
    event falls within ``match_tolerance_days`` of the injected epoch.

    Parameters
    ----------
    lc
        Host light curve; ideally real photometry, so that real systematics are
        part of the test.
    detector
        The detector to characterise — any :class:`~exocomet.core.interfaces.Detector`.
    depths
        Fractional depths to sweep.
    n_per_depth
        Injections per depth. More gives a smoother completeness curve at linear
        cost; the binomial uncertainty on a recovery fraction goes as
        ``1/sqrt(n)``.

    Returns
    -------
    RecoveryGrid
        Every individual outcome, plus completeness summaries.
    """
    cfg = config or PipelineConfig()
    generator = rng if rng is not None else np.random.default_rng(cfg.runtime.rng_seed)

    # Keep injections away from the region where candidate detection is
    # suppressed by the edge guard, plus room for the full transit.
    margin_days = cfg.candidates.edge_margin_days
    pad = margin_days + egress_duration + ingress_duration
    lo, hi = lc.time[0] + pad, lc.time[-1] - pad
    if hi <= lo:
        raise ValueError("light curve is too short to host an injection clear of its edges")

    results: list[InjectionResult] = []
    for depth in depths:
        for _ in range(n_per_depth):
            t0 = float(generator.uniform(lo, hi))
            injected = lc.with_flux(
                inject_comet_transit(
                    lc.time, lc.flux, t0, depth, ingress_duration, egress_duration
                ),
                injected_t0=t0,
                injected_depth=depth,
            )

            matches = [
                record
                for record in detector.run(injected)
                if abs(record.event.t_min - t0) <= match_tolerance_days
            ]
            best = max(matches, key=lambda r: r.score.significance) if matches else None

            results.append(
                InjectionResult(
                    t0=t0,
                    depth=depth,
                    ingress_duration=ingress_duration,
                    egress_duration=egress_duration,
                    detected=best is not None,
                    flagged=bool(best is not None and best.score.flagged),
                    recovered_t0=best.event.t_min if best else None,
                    recovered_a_dur=(
                        best.event.asymmetry.a_dur
                        if best and best.event.asymmetry is not None
                        else None
                    ),
                    delta_bic=best.score.delta_bic if best else float("nan"),
                )
            )

    return RecoveryGrid(results=tuple(results))
