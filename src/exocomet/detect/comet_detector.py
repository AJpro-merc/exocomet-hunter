"""The asymmetric-dip detector: the first concrete :class:`Detector`.

This class is deliberately thin. It owns no science of its own; it sequences the
baseline estimation, candidate finding, geometric measurement and statistical
scoring implemented in the sibling modules. Keeping the algorithm decomposed
that way is what lets each piece be unit-tested against synthetic ground truth
before any real photometry is involved.
"""

from __future__ import annotations

import numpy as np

from exocomet.core.config import PipelineConfig
from exocomet.core.interfaces import Detector
from exocomet.core.types import CandidateRecord, DipEvent, EventScore, FloatArray, LightCurveData
from exocomet.detect.asymmetry import measure_event
from exocomet.detect.baseline import days_to_cadences, local_noise, rolling_baseline
from exocomet.detect.candidates import find_candidate_events
from exocomet.detect.scoring import periodicity_flag, score_event

__all__ = ["AsymmetricDipDetector"]


class AsymmetricDipDetector(Detector):
    """Detect comet-like, asymmetric dimming events.

    Parameters
    ----------
    config
        Full pipeline configuration; defaults are used when omitted.

    Examples
    --------
    >>> detector = AsymmetricDipDetector()
    >>> records = detector.run(detrended_light_curve)  # doctest: +SKIP
    """

    name = "asymmetric-dip"

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._rng = np.random.default_rng(self.config.runtime.rng_seed)

    def baseline_and_noise(self, lc: LightCurveData) -> tuple[FloatArray, FloatArray]:
        """Return the local baseline and robust noise for a light curve."""
        median_window = days_to_cadences(self.config.baseline.median_window_days, lc.cadence_days)
        noise_window = days_to_cadences(self.config.baseline.noise_window_days, lc.cadence_days)
        baseline = rolling_baseline(lc.flux, median_window)
        sigma = local_noise(lc.flux, baseline, noise_window)
        return baseline, sigma

    def detect(self, lc: LightCurveData) -> list[DipEvent]:
        """Find and geometrically measure every candidate dip.

        Events whose ingress or egress could not be fitted are still returned,
        carrying ``unmeasurable_reason``, so that they appear in run diagnostics
        instead of vanishing silently.
        """
        baseline, sigma = self.baseline_and_noise(lc)
        windows = find_candidate_events(lc.time, lc.flux, baseline, sigma, self.config.candidates)
        return [
            measure_event(
                lc.time, lc.flux, baseline, window, lc.target_id, self.config.asymmetry
            )
            for window in windows
        ]

    def score(self, event: DipEvent, lc: LightCurveData) -> EventScore:
        """Score one event, recomputing the baseline for the parent light curve."""
        baseline, _ = self.baseline_and_noise(lc)
        periodic, period_days = periodicity_flag(lc, self.config.scoring)
        return score_event(
            event,
            lc,
            baseline,
            self.config.scoring,
            self.config.asymmetry,
            self._rng,
            periodic=periodic,
            period_days=period_days,
        )

    def run(self, lc: LightCurveData) -> list[CandidateRecord]:
        """Detect and score every event in one pass.

        Overrides the base implementation so that the baseline and the
        periodogram — both properties of the whole light curve — are computed
        once rather than once per event.
        """
        baseline, sigma = self.baseline_and_noise(lc)
        windows = find_candidate_events(lc.time, lc.flux, baseline, sigma, self.config.candidates)
        if not windows:
            return []

        periodic, period_days = periodicity_flag(lc, self.config.scoring)

        records: list[CandidateRecord] = []
        for window in windows:
            event = measure_event(
                lc.time, lc.flux, baseline, window, lc.target_id, self.config.asymmetry
            )
            score = score_event(
                event,
                lc,
                baseline,
                self.config.scoring,
                self.config.asymmetry,
                self._rng,
                periodic=periodic,
                period_days=period_days,
            )
            records.append(CandidateRecord(event=event, score=score))
        return records
