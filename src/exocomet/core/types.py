"""Core data structures shared by every stage of the pipeline.

These types deliberately contain no dependency on ``lightkurve`` or any other
archive client. Detection operates on plain NumPy arrays wrapped in
:class:`LightCurveData`, which keeps the science code testable on synthetic
data and independent of how the photometry was obtained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "AsymmetryResult",
    "CandidateRecord",
    "DipEvent",
    "EventScore",
    "EventWindow",
    "FloatArray",
    "LightCurveData",
    "ModelComparison",
    "ModelFit",
    "SideFit",
    "VettingResult",
    "VettingStatus",
]


@dataclass(frozen=True)
class LightCurveData:
    """A normalised, uniformly-described light curve.

    Parameters
    ----------
    target_id
        Catalogue identifier, e.g. ``"KIC 3542116"``.
    mission
        Mission the photometry came from, e.g. ``"Kepler"`` or ``"TESS"``.
    time
        Observation times in days (BKJD for Kepler, BTJD for TESS).
    flux
        Flux normalised so that the out-of-transit level is approximately 1.0.
    flux_err
        Per-cadence flux uncertainty, in the same normalised units as ``flux``.
    meta
        Free-form provenance (quarters/sectors stitched, detrending applied, ...).
    """

    target_id: str
    mission: str
    time: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes and monotonic time ordering."""
        n = self.time.size
        if self.flux.size != n or self.flux_err.size != n:
            raise ValueError(
                f"time/flux/flux_err length mismatch: "
                f"{n}/{self.flux.size}/{self.flux_err.size}"
            )
        if n == 0:
            raise ValueError(f"light curve for {self.target_id} is empty")
        if not np.all(np.diff(self.time) > 0):
            raise ValueError(f"time array for {self.target_id} must be strictly increasing")

    @property
    def n_points(self) -> int:
        """Number of cadences."""
        return int(self.time.size)

    @property
    def cadence_days(self) -> float:
        """Median spacing between consecutive cadences, in days."""
        return float(np.median(np.diff(self.time)))

    @property
    def baseline_days(self) -> float:
        """Total time span covered by the light curve, in days."""
        return float(self.time[-1] - self.time[0])

    def with_flux(self, flux: FloatArray, **meta: Any) -> LightCurveData:
        """Return a copy carrying new flux values and merged metadata."""
        return LightCurveData(
            target_id=self.target_id,
            mission=self.mission,
            time=self.time,
            flux=flux,
            flux_err=self.flux_err,
            meta={**self.meta, **meta},
        )


@dataclass(frozen=True)
class EventWindow:
    """Index and time bounds of a candidate dip, inclusive on both ends."""

    start_index: int
    end_index: int
    start_time: float
    end_time: float

    @property
    def duration_days(self) -> float:
        """Total span of the window in days."""
        return self.end_time - self.start_time

    @property
    def n_points(self) -> int:
        """Number of cadences inside the window."""
        return self.end_index - self.start_index + 1


@dataclass(frozen=True)
class SideFit:
    """Measurement of one side (ingress or egress) of a dip.

    ``duration_days`` runs from the half-depth crossing to the dip minimum, and
    ``slope`` is the ordinary-least-squares gradient of flux with respect to
    time over that same interval (negative on ingress, positive on egress).
    """

    duration_days: float
    slope: float
    slope_err: float
    n_points: int
    half_depth_time: float


@dataclass(frozen=True)
class AsymmetryResult:
    """Asymmetry metrics for a dip.

    Sign convention (both metrics positive == comet-like):

    ``a_dur``
        ``(egress_duration - ingress_duration) / (egress + ingress)``. Positive
        when the egress is the longer side, i.e. the star fades quickly and
        recovers slowly — the signature of a trailing dust tail.
    ``a_slope``
        ``(|ingress_slope| - |egress_slope|) / (|ingress| + |egress|)``. Written
        with the ingress term first precisely so that it is *also* positive for
        a steep-in/slow-out event, making "the two metrics agree in sign" a
        direct statement about physical consistency rather than an arithmetic
        accident.
    """

    a_dur: float
    a_slope: float
    ingress: SideFit
    egress: SideFit

    @property
    def signs_agree(self) -> bool:
        """Whether both metrics describe the same physical asymmetry direction."""
        return bool(np.sign(self.a_dur) == np.sign(self.a_slope) and self.a_dur != 0.0)


@dataclass(frozen=True)
class DipEvent:
    """A single detected dimming event with its geometry measured."""

    target_id: str
    t_min: float
    f_min: float
    depth: float
    window: EventWindow
    asymmetry: AsymmetryResult | None = None
    unmeasurable_reason: str | None = None

    @property
    def depth_ppm(self) -> float:
        """Event depth in parts per million of the local baseline."""
        return self.depth * 1.0e6

    @property
    def is_measurable(self) -> bool:
        """Whether both sides of the dip could be fitted."""
        return self.asymmetry is not None


@dataclass(frozen=True)
class ModelFit:
    """Result of fitting one parametric profile to an event."""

    name: str
    params: dict[str, float]
    chi2: float
    n_params: int
    n_points: int
    converged: bool

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``chi2 + k ln n``.

        Lower is better. The penalty term is what stops the extra free
        parameter of the comet profile from winning by flexibility alone.
        """
        if not self.converged or self.n_points <= 0:
            return float("inf")
        return self.chi2 + self.n_params * float(np.log(self.n_points))


@dataclass(frozen=True)
class ModelComparison:
    """Symmetric versus comet-shaped profile fits for one event."""

    symmetric: ModelFit
    comet: ModelFit

    @property
    def delta_bic(self) -> float:
        """``BIC(symmetric) - BIC(comet)``; positive favours the comet profile.

        By the usual convention a difference above 10 is strong evidence for the
        preferred model.
        """
        delta = self.symmetric.bic - self.comet.bic
        return float(delta) if np.isfinite(delta) else float("-inf")

    @property
    def favors_comet(self) -> bool:
        """Whether the comet profile is strongly preferred (``delta_bic > 10``)."""
        return self.delta_bic > 10.0

    @property
    def tau_over_sigma(self) -> float:
        """Ratio of fitted egress timescale to ingress width.

        This is the physically meaningful asymmetry descriptor. Unlike a
        contour-based duration ratio, it is derived from the whole fitted
        profile, so it measures the extended dust tail rather than the nearly
        symmetric core.

        Values above 1 indicate a trailing tail — obscuration that clears more
        slowly than it accumulated.

        Notes
        -----
        A contour-based ratio measured at depth fraction :math:`f` compares
        :math:`\\sigma\\sqrt{2\\ln(1/f)}` against :math:`\\tau\\ln(1/f)`. At the
        half-depth contour these are :math:`1.177\\sigma` and :math:`0.693\\tau`,
        so the contour ratio changes sign at :math:`\\tau/\\sigma \\approx 1.70`
        even while the tail is unambiguously the longer side. That mismatch is
        why this ratio, not the contour ratio, is the primary shape statistic.
        """
        sigma = self.comet.params.get("ingress_width", float("nan"))
        tau = self.comet.params.get("egress_tau", float("nan"))
        if not (np.isfinite(sigma) and np.isfinite(tau)) or sigma <= 0:
            return float("nan")
        return float(tau / sigma)


@dataclass(frozen=True)
class EventScore:
    """Statistical assessment of a dip's asymmetry."""

    a_dur_sigma: float
    significance: float
    signs_agree: bool
    flagged: bool
    periodic: bool = False
    period_days: float | None = None
    model: ModelComparison | None = None

    @property
    def delta_bic(self) -> float:
        """Model-comparison evidence for the comet profile, or ``nan`` if unfitted."""
        return self.model.delta_bic if self.model is not None else float("nan")

    @property
    def tau_over_sigma(self) -> float:
        """Fitted tail-to-core asymmetry ratio, or ``nan`` if unfitted."""
        return self.model.tau_over_sigma if self.model is not None else float("nan")


class VettingStatus(str, Enum):
    """Outcome of the mundane-explanation checks applied to a flagged event."""

    CANDIDATE = "candidate"
    ECLIPSING_BINARY = "eclipsing_binary"
    FLARE = "flare"
    ARTIFACT = "artifact"
    KNOWN_VARIABLE = "known_variable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class VettingResult:
    """Why an event did or did not survive vetting."""

    status: VettingStatus
    reasons: tuple[str, ...] = ()
    catalog_matches: dict[str, str] = field(default_factory=dict)

    @property
    def is_candidate(self) -> bool:
        """Whether the event survived every check."""
        return self.status is VettingStatus.CANDIDATE


@dataclass(frozen=True)
class CandidateRecord:
    """One row of the pipeline's output table: event, score and vetting."""

    event: DipEvent
    score: EventScore
    vetting: VettingResult | None = None

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dictionary suitable for a :class:`pandas.DataFrame`."""
        asym = self.event.asymmetry
        return {
            "target_id": self.event.target_id,
            "t_min": self.event.t_min,
            "depth_ppm": self.event.depth_ppm,
            "duration_days": self.event.window.duration_days,
            "a_dur": asym.a_dur if asym else np.nan,
            "a_slope": asym.a_slope if asym else np.nan,
            "ingress_duration_days": asym.ingress.duration_days if asym else np.nan,
            "egress_duration_days": asym.egress.duration_days if asym else np.nan,
            "a_dur_sigma": self.score.a_dur_sigma,
            "significance": self.score.significance,
            "delta_bic": self.score.delta_bic,
            "tau_over_sigma": self.score.tau_over_sigma,
            "signs_agree": self.score.signs_agree,
            "flagged": self.score.flagged,
            "periodic": self.score.periodic,
            "period_days": self.score.period_days,
            "vetting_status": self.vetting.status.value if self.vetting else None,
            "vetting_reasons": "; ".join(self.vetting.reasons) if self.vetting else None,
        }
