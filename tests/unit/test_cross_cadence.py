# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-cadence consistency test.

The same physical event, detected the same way regardless of whether it was
measured at Kepler long cadence or TESS 120s/1800s cadence. This is the test
A1's fix promised (Next Steps A1: "inject the same comet into
a 120 s light curve and a 1800 s light curve covering the same time span;
detection, epoch, and tau/sigma must agree within tolerance") -- windows are
specified in days and converted to cadences at runtime, so the same physical
window should apply no matter the sampling rate. No network access: entirely
synthetic light curves at TESS-realistic cadences.
"""

from __future__ import annotations

import numpy as np

from exocomet.core.config import PipelineConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from tests.conftest import inject_asymmetric_dip, make_flat_light_curve

#: Common TESS product cadences, in days.
TESS_SPOC_120S_DAYS = 120.0 / 86400.0
TESS_FFI_1800S_DAYS = 1800.0 / 86400.0

FAST = PipelineConfig()
FAST = PipelineConfig(scoring=type(FAST.scoring)(bootstrap_draws=60))


#: Both cadences simulate the same physical time span, so the injected event
#: lands at the same absolute epoch regardless of how many cadences that takes.
SPAN_DAYS = 20.0


def _detect_one(cadence_days: float, seed: int) -> tuple[float, float] | None:
    """Inject one comet-shaped dip at the given cadence; return (epoch, tau_over_sigma)."""
    n_points = int(SPAN_DAYS / cadence_days)
    lc = make_flat_light_curve(
        n_points=n_points, cadence_days=cadence_days, noise_sigma=1.5e-4, seed=seed
    )
    t0 = SPAN_DAYS / 2.0
    flux = inject_asymmetric_dip(
        lc.time, lc.flux, t0=t0, depth=4.0e-3, ingress_duration=0.15, egress_duration=0.55
    )
    injected = lc.with_flux(flux)
    records = AsymmetricDipDetector(FAST).run(injected)
    if not records:
        return None
    record = min(records, key=lambda r: abs(r.event.window.start_time - t0))
    return record.event.window.start_time, record.score.tau_over_sigma


class TestCrossCadenceConsistency:
    def test_epoch_and_tau_over_sigma_agree_across_cadences(self) -> None:
        """A day-scale comet dip is found consistently at 120s and 1800s cadence."""
        result_120s = _detect_one(TESS_SPOC_120S_DAYS, seed=1)
        result_1800s = _detect_one(TESS_FFI_1800S_DAYS, seed=1)

        assert result_120s is not None, "event not found at 120s cadence"
        assert result_1800s is not None, "event not found at 1800s cadence"

        epoch_120s, tau_120s = result_120s
        epoch_1800s, tau_1800s = result_1800s

        assert abs(epoch_120s - epoch_1800s) < 0.5, (
            f"epoch disagreement across cadences: {epoch_120s} vs {epoch_1800s}"
        )
        assert np.isfinite(tau_120s) and np.isfinite(tau_1800s)
        assert abs(tau_120s - tau_1800s) < 1.0, (
            f"tau/sigma disagreement across cadences: {tau_120s} vs {tau_1800s}"
        )
