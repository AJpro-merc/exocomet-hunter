# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Network-call bounding in :mod:`exocomet.io.download`.

Regression cover for the 2026-09-14 hang: a fetch for KIC 3542116 sat at zero
CPU for nine minutes with no timeout anywhere in the code path. These tests
prove the budgets now trip, in bounded time, and that a hung target no longer
holds an entire batch — all with a fake ``lightkurve`` module, so no network
access and no MAST traffic whatsoever.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from exocomet.io import download
from exocomet.io.download import MastTimeoutError, fetch_light_curve, iter_light_curves

#: Every timeout under test is sub-second, so the whole module runs fast; a
#: real hang would blow past this and fail the test rather than block the suite.
TINY_TIMEOUT = 0.2


class _FakeLightCurve:
    """Minimal stand-in for a stitched lightkurve object."""

    def __init__(self, n: int = 50) -> None:
        values = np.linspace(0.0, 1.0, n)
        self.time = types.SimpleNamespace(value=values)
        self.flux = types.SimpleNamespace(value=np.ones(n))
        self.flux_err = types.SimpleNamespace(value=np.full(n, 1e-3))


class _FakeCollection:
    def stitch(self) -> _FakeLightCurve:
        return _FakeLightCurve()


class _FakeSearchResult:
    """Search result whose ``download_all`` behaviour the test dictates."""

    def __init__(self, download_all_impl=None) -> None:
        self.table = types.SimpleNamespace(colnames=["sequence_number"])
        self._impl = download_all_impl

    def __len__(self) -> int:
        return 1

    def download_all(self, **_kwargs) -> _FakeCollection:
        if self._impl is not None:
            return self._impl()
        return _FakeCollection()


def _install_fake_lightkurve(monkeypatch, search_impl) -> None:
    """Replace the lazily imported ``lightkurve`` module for one test."""
    module = types.ModuleType("lightkurve")
    module.search_lightcurve = search_impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightkurve", module)
    # astroquery tuning touches the network stack's config only, but there is
    # no reason to import it at all in a no-network test.
    monkeypatch.setattr(download, "_configure_astroquery", lambda *a, **k: None)


def _blocker(stop: threading.Event):
    """Return a callable that blocks until the test releases it."""

    def _hang(*_args, **_kwargs):
        stop.wait(30.0)
        raise AssertionError("blocker was never released")

    return _hang


def test_search_hang_raises_timeout_in_bounded_time(monkeypatch, tmp_path):
    stop = threading.Event()
    _install_fake_lightkurve(monkeypatch, _blocker(stop))

    started = time.monotonic()
    try:
        with pytest.raises(MastTimeoutError, match="search_lightcurve"):
            fetch_light_curve(
                "KIC 3542116",
                cache_dir=tmp_path,
                search_timeout=TINY_TIMEOUT,
                max_attempts=2,
                retry_backoff=0.01,
            )
        elapsed = time.monotonic() - started
    finally:
        stop.set()

    # Two attempts plus a 0.01s backoff; anything near the old behaviour
    # (unbounded) would never get here at all.
    assert elapsed < 5.0


def test_download_hang_raises_timeout_in_bounded_time(monkeypatch, tmp_path):
    stop = threading.Event()
    result = _FakeSearchResult(download_all_impl=_blocker(stop))
    _install_fake_lightkurve(monkeypatch, lambda *a, **k: result)

    started = time.monotonic()
    try:
        with pytest.raises(MastTimeoutError, match="download_all"):
            fetch_light_curve(
                "KIC 3542116",
                cache_dir=tmp_path,
                search_timeout=5.0,
                download_timeout=TINY_TIMEOUT,
                max_attempts=1,
            )
        elapsed = time.monotonic() - started
    finally:
        stop.set()

    assert elapsed < 5.0


def test_retry_is_bounded_and_recovers_from_a_transient_failure(monkeypatch, tmp_path):
    calls: list[int] = []

    def flaky(*_args, **_kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("connection reset by peer")
        return _FakeSearchResult()

    _install_fake_lightkurve(monkeypatch, flaky)

    data = fetch_light_curve(
        "KIC 3542116",
        cache_dir=tmp_path,
        max_attempts=3,
        retry_backoff=0.01,
    )
    assert len(calls) == 3
    assert data.target_id == "KIC 3542116"


def test_retry_gives_up_rather_than_looping_forever(monkeypatch, tmp_path):
    calls: list[int] = []

    def always_fails(*_args, **_kwargs):
        calls.append(1)
        raise OSError("connection reset by peer")

    _install_fake_lightkurve(monkeypatch, always_fails)

    with pytest.raises(OSError, match="connection reset"):
        fetch_light_curve(
            "KIC 3542116",
            cache_dir=tmp_path,
            max_attempts=3,
            retry_backoff=0.01,
        )
    assert len(calls) == 3


def test_iter_light_curves_moves_past_a_hung_target(monkeypatch, tmp_path):
    """The batch-level guarantee: one hung star must not stall the rest."""
    stop = threading.Event()

    def per_target(target_id, **_kwargs):
        if target_id == "KIC 3542116":
            stop.wait(30.0)
            raise AssertionError("blocker was never released")
        return _FakeSearchResult()

    _install_fake_lightkurve(monkeypatch, per_target)

    started = time.monotonic()
    try:
        yielded = list(
            iter_light_curves(
                ["KIC 3542116", "KIC 11084727"],
                cache_dir=tmp_path,
                per_target_timeout=TINY_TIMEOUT,
                search_timeout=5.0,
                max_attempts=1,
            )
        )
        elapsed = time.monotonic() - started
    finally:
        stop.set()

    assert [lc.target_id for lc in yielded] == ["KIC 11084727"]
    assert elapsed < 5.0


def test_iter_light_curves_reraises_hang_when_skip_errors_is_false(monkeypatch, tmp_path):
    stop = threading.Event()
    _install_fake_lightkurve(monkeypatch, _blocker(stop))

    try:
        with pytest.raises(MastTimeoutError):
            list(
                iter_light_curves(
                    ["KIC 3542116"],
                    cache_dir=tmp_path,
                    per_target_timeout=TINY_TIMEOUT,
                    skip_errors=False,
                    max_attempts=1,
                )
            )
    finally:
        stop.set()
