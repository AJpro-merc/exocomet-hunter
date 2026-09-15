# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline and noise estimation on data with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from exocomet.detect.baseline import local_noise, odd_window, rolling_baseline

from tests.conftest import inject_symmetric_dip, make_flat_light_curve


class TestOddWindow:
    def test_coerces_even_to_odd(self) -> None:
        assert odd_window(100, 1000) == 99

    def test_clamps_to_series_length(self) -> None:
        assert odd_window(5001, 101) == 101

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            odd_window(0, 100)


class TestRollingBaseline:
    def test_flat_curve_recovers_unity(self, flat_lc) -> None:
        baseline = rolling_baseline(flat_lc.flux, 101)
        assert np.allclose(baseline, 1.0, atol=5.0e-5)

    def test_baseline_ignores_a_dip(self) -> None:
        """A running median must not be dragged down by an in-transit minority."""
        lc = make_flat_light_curve(noise_sigma=0.0, seed=7)
        t0 = float(lc.time[lc.n_points // 2])
        flux = inject_symmetric_dip(lc.time, lc.flux, t0=t0, depth=5.0e-3, duration=0.2)

        baseline = rolling_baseline(flux, 101)
        centre = lc.n_points // 2
        assert baseline[centre] == pytest.approx(1.0, abs=1.0e-6)

    def test_tracks_a_slow_trend(self) -> None:
        lc = make_flat_light_curve(noise_sigma=0.0, seed=8)
        trend = 1.0 + 0.01 * np.sin(2.0 * np.pi * lc.time / 20.0)
        baseline = rolling_baseline(trend, 51)
        interior = slice(50, -50)
        assert np.allclose(baseline[interior], trend[interior], atol=2.0e-4)

    def test_edges_lag_a_trend_by_half_a_window(self) -> None:
        """Documents an accepted cost of truncating rather than padding.

        At index 0 a centred window can only look forward, so on a rising trend
        the median reports a later, higher value. The bias is bounded by the
        trend's change over half a window, and it is why candidate detection
        discards events within ``edge_margin_days`` of either end.
        """
        lc = make_flat_light_curve(noise_sigma=0.0, seed=9)
        slope_per_day = 0.01
        trend = 1.0 + slope_per_day * lc.time
        baseline = rolling_baseline(trend, 51)

        half_window_days = 25 * float(np.median(np.diff(lc.time)))
        max_expected_lag = slope_per_day * half_window_days
        assert 0.0 < baseline[0] - trend[0] <= max_expected_lag * 1.05


class TestLocalNoise:
    def test_recovers_injected_scatter(self) -> None:
        lc = make_flat_light_curve(n_points=4000, noise_sigma=3.0e-4, seed=11)
        baseline = rolling_baseline(lc.flux, 101)
        sigma = local_noise(lc.flux, baseline, 101)
        assert float(np.median(sigma)) == pytest.approx(3.0e-4, rel=0.2)

    def test_is_strictly_positive_on_noiseless_data(self) -> None:
        """A zero denominator downstream would be a division-by-zero bug."""
        lc = make_flat_light_curve(noise_sigma=0.0, seed=12)
        baseline = rolling_baseline(lc.flux, 101)
        sigma = local_noise(lc.flux, baseline, 101)
        assert np.all(sigma > 0.0)
