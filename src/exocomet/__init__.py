# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Exocomet Hunter: detection of asymmetric, comet-like transits in stellar photometry.

The package is organised as a small framework rather than a single script:

``exocomet.core``
    Data structures, typed configuration, and the plugin interfaces.
``exocomet.io``
    Light-curve retrieval and on-disk caching (MAST via ``lightkurve``).
``exocomet.detrend``
    Removal of instrumental trends ahead of detection.
``exocomet.detect``
    The asymmetric-dip detector: baseline estimation, candidate finding,
    ingress/egress measurement, and significance scoring.
``exocomet.vetting``
    Checks against catalogues and known artefact signatures.
``exocomet.viz``
    Diagnostic and publication figures.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from exocomet.core.config import PipelineConfig
from exocomet.core.interfaces import Detector, Detrender, LightCurveSource, Vetter
from exocomet.core.types import (
    AsymmetryResult,
    CandidateRecord,
    DipEvent,
    EventScore,
    EventWindow,
    LightCurveData,
    SideFit,
    VettingResult,
    VettingStatus,
)

try:
    __version__ = version("exocomet-hunter")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "AsymmetryResult",
    "CandidateRecord",
    "Detector",
    "Detrender",
    "DipEvent",
    "EventScore",
    "EventWindow",
    "LightCurveData",
    "LightCurveSource",
    "PipelineConfig",
    "SideFit",
    "VettingResult",
    "VettingStatus",
    "Vetter",
    "__version__",
]
