# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end detector behaviour: what gets flagged, and what must not."""

from __future__ import annotations

import numpy as np
import pytest

from exocomet.core.config import PipelineConfig, ScoringConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector

from tests.conftest import inject_asymmetric_dip, make_flat_light_curve

# The bootstrap dominates runtime and its precision is not what these tests
# probe, so they run with far fewer draws than a production search.
FAST = PipelineConfig(scoring=ScoringConfig(bootstrap_draws=60))


class TestDetectorFlagging:
    def test_pure_noise_produces_no_events(self, flat_lc) -> None:
        """The headline false-positive test: noise alone must yield nothing."""
        assert AsymmetricDipDetector(FAST).run(flat_lc) == []

    def test_comet_like_dip_is_flagged(self, comet_lc) -> None:
        records = AsymmetricDipDetector(FAST).run(comet_lc)
        assert len(records) == 1

        record = records[0]
        assert record.score.flagged
        assert record.event.asymmetry is not None
        assert record.event.asymmetry.a_dur > 0.0

    def test_symmetric_dip_is_not_flagged(self, symmetric_lc) -> None:
        """A symmetric dip is detected as an event but rejected as comet-like."""
        records = AsymmetricDipDetector(FAST).run(symmetric_lc)
        assert len(records) == 1
        assert not records[0].score.flagged

    def test_reversed_asymmetry_is_not_flagged(self, reversed_comet_lc) -> None:
        """Slow ingress, fast egress is backwards for a comet: never a candidate.

        The sign of ``A_dur`` still records which way round the event is, and is
        still reported; it is the flagging decision that must not be sign-blind.
        """
        records = AsymmetricDipDetector(FAST).run(reversed_comet_lc)
        assert len(records) == 1
        assert not records[0].score.flagged
        assert records[0].event.asymmetry is not None
        assert records[0].event.asymmetry.a_dur < 0.0

    def test_reversed_asymmetry_still_reports_significance(
        self, reversed_comet_lc
    ) -> None:
        """The contour statistics survive as output columns, not as the decision."""
        records = AsymmetricDipDetector(FAST).run(reversed_comet_lc)
        assert len(records) == 1
        score = records[0].score
        assert not score.flagged
        assert score.significance > 0.0
        assert score.tau_over_sigma < FAST.scoring.tau_over_sigma_threshold


class TestSignificance:
    def test_deeper_events_are_more_significant(self) -> None:
        """Signal-to-noise must move the significance in the obvious direction."""
        significances = []
        for depth in (1.0e-3, 8.0e-3):
            lc = make_flat_light_curve(seed=41)
            flux = inject_asymmetric_dip(
                lc.time, lc.flux, t0=20.0, depth=depth, ingress_duration=0.1, egress_duration=0.5
            )
            records = AsymmetricDipDetector(FAST).run(lc.with_flux(flux))
            assert len(records) == 1
            significances.append(records[0].score.significance)

        assert significances[1] > significances[0]

    def test_bootstrap_reports_a_finite_uncertainty(self, comet_lc) -> None:
        record = AsymmetricDipDetector(FAST).run(comet_lc)[0]
        assert np.isfinite(record.score.a_dur_sigma)
        assert record.score.a_dur_sigma > 0.0


class TestReproducibility:
    def test_same_seed_gives_identical_scores(self, comet_lc) -> None:
        """Reproducibility is a claim the README makes; this is the check."""
        first = AsymmetricDipDetector(FAST).run(comet_lc)[0]
        second = AsymmetricDipDetector(FAST).run(comet_lc)[0]
        assert first.score.significance == pytest.approx(second.score.significance)


class TestOutputRow:
    def test_record_flattens_to_a_row(self, comet_lc) -> None:
        row = AsymmetricDipDetector(FAST).run(comet_lc)[0].to_row()

        assert row["target_id"] == comet_lc.target_id
        assert row["a_dur"] > 0.0
        assert row["depth_ppm"] > 0.0
        assert "delta_bic" in row
        assert row["vetting_status"] is None  # not yet vetted at this stage
