"""The core correctness tests: does the asymmetry measurement mean what we claim?

A piecewise-linear injected dip has an analytic answer — the measured duration
asymmetry must equal ``(egress - ingress) / (egress + ingress)`` — so these tests
check a number, not merely a sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from exocomet.detect.asymmetry import measure_event, ols_slope
from exocomet.detect.baseline import local_noise, rolling_baseline
from exocomet.detect.candidates import find_candidate_events

from tests.conftest import expected_a_dur, inject_asymmetric_dip, make_flat_light_curve


def measure_single_event(lc):
    """Detect and measure the one event in a synthetic light curve."""
    baseline = rolling_baseline(lc.flux, 101)
    sigma = local_noise(lc.flux, baseline, 101)
    windows = find_candidate_events(lc.time, lc.flux, baseline, sigma)
    assert len(windows) == 1, f"expected exactly one event, found {len(windows)}"
    return measure_event(lc.time, lc.flux, baseline, windows[0], lc.target_id)


class TestOlsSlope:
    def test_recovers_an_exact_line(self) -> None:
        x = np.linspace(0.0, 1.0, 50)
        slope, slope_err = ols_slope(x, 3.0 + 2.5 * x)
        assert slope == pytest.approx(2.5)
        assert slope_err == pytest.approx(0.0, abs=1.0e-9)

    def test_too_few_points_is_not_a_fit(self) -> None:
        slope, slope_err = ols_slope(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert np.isnan(slope)
        assert np.isinf(slope_err)

    def test_zero_spread_in_x_is_not_a_fit(self) -> None:
        slope, _ = ols_slope(np.zeros(5), np.arange(5.0))
        assert np.isnan(slope)


class TestSymmetricEvents:
    def test_symmetric_dip_has_near_zero_asymmetry(self, symmetric_lc) -> None:
        event = measure_single_event(symmetric_lc)
        assert event.asymmetry is not None
        assert abs(event.asymmetry.a_dur) < 0.15

    def test_symmetric_dip_slopes_are_comparable(self, symmetric_lc) -> None:
        event = measure_single_event(symmetric_lc)
        assert event.asymmetry is not None
        assert abs(event.asymmetry.a_slope) < 0.2


class TestAsymmetricEvents:
    def test_comet_like_dip_is_positive(self, comet_lc) -> None:
        """Steep ingress, slow egress must give a positive duration asymmetry."""
        event = measure_single_event(comet_lc)
        assert event.asymmetry is not None
        assert event.asymmetry.a_dur > 0.3

    def test_reversed_asymmetry_is_negative(self, reversed_comet_lc) -> None:
        event = measure_single_event(reversed_comet_lc)
        assert event.asymmetry is not None
        assert event.asymmetry.a_dur < -0.3

    def test_matches_the_analytic_value(self) -> None:
        """For a linear ramp the measured asymmetry has a closed form."""
        lc = make_flat_light_curve(noise_sigma=1.0e-5, seed=31)
        flux = inject_asymmetric_dip(
            lc.time, lc.flux, t0=20.0, depth=4.0e-3, ingress_duration=0.15, egress_duration=0.45
        )
        event = measure_single_event(lc.with_flux(flux))

        assert event.asymmetry is not None
        assert event.asymmetry.a_dur == pytest.approx(expected_a_dur(0.15, 0.45), abs=0.1)

    def test_both_metrics_agree_in_sign(self, comet_lc, reversed_comet_lc) -> None:
        """The two metrics are defined so that agreement means physical consistency."""
        for lc in (comet_lc, reversed_comet_lc):
            event = measure_single_event(lc)
            assert event.asymmetry is not None
            assert event.asymmetry.signs_agree

    @pytest.mark.parametrize(
        ("ingress", "egress"),
        [(0.1, 0.4), (0.15, 0.3), (0.3, 0.15), (0.4, 0.1)],
    )
    def test_sign_tracks_the_injected_ordering(self, ingress: float, egress: float) -> None:
        lc = make_flat_light_curve(noise_sigma=2.0e-5, seed=32)
        flux = inject_asymmetric_dip(
            lc.time, lc.flux, t0=20.0, depth=4.0e-3, ingress_duration=ingress, egress_duration=egress
        )
        event = measure_single_event(lc.with_flux(flux))

        assert event.asymmetry is not None
        assert np.sign(event.asymmetry.a_dur) == np.sign(egress - ingress)


class TestMinimumLocation:
    def test_recovers_the_injected_epoch(self, comet_lc) -> None:
        event = measure_single_event(comet_lc)
        assert event.t_min == pytest.approx(comet_lc.meta["injected_t0"], abs=0.05)

    def test_recovers_the_injected_depth(self, comet_lc) -> None:
        event = measure_single_event(comet_lc)
        assert event.depth == pytest.approx(2.0e-3, rel=0.2)
