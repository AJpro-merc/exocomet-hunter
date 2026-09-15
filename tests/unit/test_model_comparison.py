# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Model comparison: does the BIC actually prefer the right profile?

This is the metric the exocomet literature ranks on, so these tests matter more
than the non-parametric asymmetry ratios: they check that a comet-shaped event
prefers the comet profile, a symmetric one does not, and that the extra free
parameter cannot win on flexibility alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from exocomet.core.config import PipelineConfig, ScoringConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detect.model_comparison import comet_profile, symmetric_profile

from tests.conftest import inject_asymmetric_dip, make_flat_light_curve

FAST = PipelineConfig(scoring=ScoringConfig(bootstrap_draws=30))


class TestProfiles:
    def test_symmetric_profile_is_symmetric_about_t0(self) -> None:
        t = np.linspace(-1.0, 1.0, 201)
        y = symmetric_profile(t, t0=0.0, depth=0.01, width=0.2, baseline=1.0)
        assert np.allclose(y, y[::-1], atol=1.0e-12)

    def test_comet_profile_is_continuous_at_the_minimum(self) -> None:
        """The Gaussian ingress and exponential egress must meet at the minimum."""
        eps = 1.0e-9
        t = np.array([-eps, 0.0, eps])
        y = comet_profile(t, t0=0.0, depth=0.01, ingress_width=0.1, egress_tau=0.4, baseline=1.0)
        assert y[0] == pytest.approx(y[2], abs=1.0e-7)
        assert y[1] == pytest.approx(0.99, abs=1.0e-9)

    def test_comet_profile_recovers_more_slowly_than_it_falls(self) -> None:
        t = np.array([-0.3, 0.3])
        y = comet_profile(t, t0=0.0, depth=0.01, ingress_width=0.1, egress_tau=0.5, baseline=1.0)
        assert y[1] < y[0]  # still depressed on egress while ingress has recovered


class TestModelSelection:
    def test_comet_event_prefers_the_comet_profile(self, comet_lc) -> None:
        record = AsymmetricDipDetector(FAST).run(comet_lc)[0]
        assert record.score.model is not None
        assert record.score.delta_bic > 10.0
        assert record.score.model.favors_comet

    def test_symmetric_event_does_not_prefer_the_comet_profile(self, symmetric_lc) -> None:
        """The BIC penalty must stop the extra parameter winning by flexibility."""
        record = AsymmetricDipDetector(FAST).run(symmetric_lc)[0]
        assert record.score.model is not None
        assert not record.score.model.favors_comet

    def test_stronger_asymmetry_gives_stronger_evidence(self) -> None:
        evidence = []
        for egress in (0.12, 0.8):
            lc = make_flat_light_curve(seed=51)
            flux = inject_asymmetric_dip(
                lc.time, lc.flux, t0=20.0, depth=3.0e-3, ingress_duration=0.1, egress_duration=egress
            )
            record = AsymmetricDipDetector(FAST).run(lc.with_flux(flux))[0]
            evidence.append(record.score.delta_bic)

        assert evidence[1] > evidence[0]

    def test_fits_converge_on_a_clean_event(self, comet_lc) -> None:
        model = AsymmetricDipDetector(FAST).run(comet_lc)[0].score.model
        assert model is not None
        assert model.symmetric.converged
        assert model.comet.converged
        assert np.isfinite(model.symmetric.bic)
        assert np.isfinite(model.comet.bic)

    def test_comet_fit_recovers_a_longer_egress_timescale(self, comet_lc) -> None:
        """The fitted parameters should describe the injected geometry."""
        model = AsymmetricDipDetector(FAST).run(comet_lc)[0].score.model
        assert model is not None
        assert model.comet.params["egress_tau"] > model.comet.params["ingress_width"]
