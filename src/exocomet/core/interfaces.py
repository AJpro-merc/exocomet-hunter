"""Plugin interfaces that define the shape of the pipeline.

The pipeline is deliberately written against these abstractions rather than
against the exocomet detector specifically. A different search — for example the
planned anomalous-dimming (technosignature) detector — is added by implementing
:class:`Detector` and passing it to
:func:`exocomet.pipeline.run_target`; no orchestration code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from exocomet.core.types import (
    CandidateRecord,
    DipEvent,
    EventScore,
    LightCurveData,
    VettingResult,
)

__all__ = ["Detector", "Detrender", "LightCurveSource", "Vetter"]


@runtime_checkable
class LightCurveSource(Protocol):
    """Anything that can produce a light curve for a target identifier."""

    def fetch(self, target_id: str, mission: str) -> LightCurveData:
        """Return the stitched, normalised light curve for ``target_id``."""
        ...


@runtime_checkable
class Detrender(Protocol):
    """Anything that removes instrumental trends from a light curve."""

    def detrend(self, lc: LightCurveData) -> LightCurveData:
        """Return a copy of ``lc`` with long-timescale trends removed."""
        ...


@runtime_checkable
class Vetter(Protocol):
    """Anything that decides whether a flagged event has a mundane explanation."""

    def vet(self, event: DipEvent, lc: LightCurveData) -> VettingResult:
        """Classify ``event`` as a genuine candidate or a known false positive."""
        ...


class Detector(ABC):
    """Base class for detection algorithms.

    A detector answers two questions: which parts of a light curve look like
    events (:meth:`detect`), and how confident we are that a given event is real
    (:meth:`score`). Splitting the two keeps the expensive statistical work off
    the path of the cheap first pass.
    """

    #: Short identifier recorded in run manifests and output tables.
    name: str = "detector"

    @abstractmethod
    def detect(self, lc: LightCurveData) -> list[DipEvent]:
        """Find candidate events in a (detrended) light curve."""

    @abstractmethod
    def score(self, event: DipEvent, lc: LightCurveData) -> EventScore:
        """Assess the statistical significance of a single event."""

    def run(self, lc: LightCurveData) -> list[CandidateRecord]:
        """Detect and score in one pass.

        Parameters
        ----------
        lc
            A detrended light curve.

        Returns
        -------
        list of CandidateRecord
            One record per detected event, scored but not yet vetted.
        """
        return [CandidateRecord(event=event, score=self.score(event, lc)) for event in self.detect(lc)]
