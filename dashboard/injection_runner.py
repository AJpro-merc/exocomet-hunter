# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Injected-scenario runs: real photometry, a synthetic transit, unmodified detector.

No type in the real pipeline marks a result as coming from injected data
(confirmed against ``calibration/injection.py`` and ``cli.py`` before writing
this -- ``inject_comet_transit`` et al. return plain flux arrays with no
provenance field). Every caller in this dashboard must therefore carry the
``injected=True`` flag itself, all the way through to storage and display --
see ``storage.add_run_target``/``add_candidates``'s ``injected`` columns and
the frontend's mandatory "INJECTED" badge. This is a scientific-integrity
requirement, not a UI nicety: an injected result must never be visually
indistinguishable from a real detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exocomet.calibration.injection import inject_comet_transit
from exocomet.core.config import PipelineConfig
from exocomet.core.types import CandidateRecord
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol
from exocomet.io.download import fetch_light_curve


@dataclass(frozen=True)
class InjectionParams:
    depth: float = 0.001
    ingress_duration_days: float = 0.3
    egress_duration_days: float = 1.0
    #: Fraction of the light curve's time span at which to centre the
    #: injected transit; 0.5 = the middle of the observed baseline.
    epoch_fraction: float = 0.5


def run_injected_target(
    target_id: str,
    mission: str,
    config: PipelineConfig,
    raw_cache_dir: str,
    params: InjectionParams,
) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """Fetch real photometry, inject a synthetic transit, run the unmodified detector.

    Returns the detected records plus a small dict describing exactly what
    was injected (for the report/UI to state plainly, alongside "INJECTED").
    """
    lc = fetch_light_curve(target_id, mission=mission, cache_dir=raw_cache_dir, discard_after_read=True)

    t0 = float(lc.time[0] + params.epoch_fraction * (lc.time[-1] - lc.time[0]))
    injected_flux = inject_comet_transit(
        lc.time,
        lc.flux,
        t0=t0,
        depth=params.depth,
        ingress_duration=params.ingress_duration_days,
        egress_duration=params.egress_duration_days,
    )
    lc_injected = lc.with_flux(
        injected_flux,
        injected=True,
        injection_t0=t0,
        injection_depth=params.depth,
        injection_ingress_days=params.ingress_duration_days,
        injection_egress_days=params.egress_duration_days,
    )

    detrended = detrend_savgol(lc_injected, config.detrend)
    records = AsymmetricDipDetector(config).run(detrended)

    injection_meta = {
        "t0": t0,
        "depth": params.depth,
        "ingress_duration_days": params.ingress_duration_days,
        "egress_duration_days": params.egress_duration_days,
    }
    return records, injection_meta
