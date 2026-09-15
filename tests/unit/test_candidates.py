"""Candidate event detection: what counts as an event, and what must not."""

from __future__ import annotations

import numpy as np

from exocomet.core.config import CandidateConfig
from exocomet.detect.baseline import local_noise, rolling_baseline
from exocomet.detect.candidates import find_candidate_events

from tests.conftest import inject_asymmetric_dip, inject_symmetric_dip, make_flat_light_curve


def _windows(flux: np.ndarray, lc, config: CandidateConfig | None = None):
    baseline = rolling_baseline(flux, 101)
    sigma = local_noise(flux, baseline, 101)
    return find_candidate_events(lc.time, flux, baseline, sigma, config)


class TestFindCandidateEvents:
    def test_pure_noise_yields_no_events(self, flat_lc) -> None:
        """The single most important negative result: noise alone must be quiet."""
        assert _windows(flat_lc.flux, flat_lc) == []

    def test_finds_a_single_injected_dip(self, comet_lc) -> None:
        windows = _windows(comet_lc.flux, comet_lc)
        assert len(windows) == 1

        t0 = comet_lc.meta["injected_t0"]
        window = windows[0]
        assert window.start_time <= t0 <= window.end_time

    def test_single_cadence_spike_is_rejected(self) -> None:
        """One deviant sample is a cosmic ray, not an event."""
        lc = make_flat_light_curve(seed=21)
        flux = lc.flux.copy()
        flux[1000] -= 5.0e-3

        assert _windows(flux, lc) == []

    def test_two_separated_dips_stay_separate(self) -> None:
        lc = make_flat_light_curve(seed=22)
        flux = inject_symmetric_dip(lc.time, lc.flux, t0=10.0, depth=2.0e-3, duration=0.2)
        flux = inject_symmetric_dip(lc.time, flux, t0=30.0, depth=2.0e-3, duration=0.2)

        windows = _windows(flux, lc)
        assert len(windows) == 2
        assert windows[0].end_time < windows[1].start_time

    def test_windows_never_overlap(self) -> None:
        """Adjacent events must not share a cadence after outward expansion."""
        lc = make_flat_light_curve(seed=23)
        flux = lc.flux.copy()
        for t0 in (10.0, 10.9, 11.8):
            flux = inject_asymmetric_dip(
                lc.time, flux, t0=t0, depth=3.0e-3, ingress_duration=0.1, egress_duration=0.3
            )

        windows = _windows(flux, lc)
        for earlier, later in zip(windows, windows[1:], strict=False):
            assert earlier.end_index < later.start_index

    def test_higher_threshold_suppresses_a_shallow_dip(self) -> None:
        """The sigma threshold is the knob that trades completeness for purity."""
        lc = make_flat_light_curve(seed=24, noise_sigma=2.0e-4)
        flux = inject_symmetric_dip(lc.time, lc.flux, t0=20.0, depth=2.0e-3, duration=0.2)

        assert len(_windows(flux, lc, CandidateConfig(sigma_threshold=4.0))) == 1
        assert _windows(flux, lc, CandidateConfig(sigma_threshold=50.0)) == []

    def test_events_at_the_series_edge_are_discarded(self) -> None:
        """Edge baselines come from truncated windows and are not trusted."""
        lc = make_flat_light_curve(seed=26)
        flux = inject_symmetric_dip(
            lc.time, lc.flux, t0=float(lc.time[10]), depth=5.0e-3, duration=0.15
        )

        assert _windows(flux, lc, CandidateConfig(edge_margin_days=1.0208)) == []
        assert len(_windows(flux, lc, CandidateConfig(edge_margin_days=0.0))) == 1

    def test_window_duration_is_capped(self) -> None:
        """A long instrumental excursion must not grow an unbounded window."""
        lc = make_flat_light_curve(n_points=4000, seed=25)
        flux = lc.flux.copy()
        depressed = (lc.time > 20.0) & (lc.time < 45.0)
        flux[depressed] -= 3.0e-3

        config = CandidateConfig(max_window_days=3.0)
        for window in _windows(flux, lc, config):
            assert window.duration_days <= 3.0 + 1.0e-6
