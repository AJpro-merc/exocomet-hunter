# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip correctness and atomic-write guarantees for :mod:`exocomet.io.extract`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from exocomet.core.types import LightCurveData
from exocomet.io.extract import extract_path, load_extract, sanitize_target_id, save_extract


def _make_light_curve(target_id: str = "KIC 3542116", mission: str = "Kepler") -> LightCurveData:
    n = 200
    time = np.linspace(100.0, 120.0, n)
    rng = np.random.default_rng(0)
    flux = 1.0 + 1e-4 * rng.standard_normal(n)
    flux_err = np.full(n, 1e-4)
    meta = {
        "author": "Kepler",
        "n_products": 17,
        "sectors_or_quarters": [1, 2, 3],
        "median_raw_flux": 12345.6789,
    }
    return LightCurveData(
        target_id=target_id, mission=mission, time=time, flux=flux, flux_err=flux_err, meta=meta
    )


class TestSanitizeTargetId:
    def test_replaces_space(self) -> None:
        assert sanitize_target_id("KIC 3542116") == "KIC_3542116"

    def test_replaces_slash(self) -> None:
        assert sanitize_target_id("TIC/260128333") == "TIC_260128333"

    def test_leaves_clean_id_untouched(self) -> None:
        assert sanitize_target_id("TIC260128333") == "TIC260128333"


class TestExtractPath:
    def test_standard_layout(self, tmp_path: Path) -> None:
        path = extract_path(tmp_path, "Kepler", "KIC 3542116")
        assert path == tmp_path / "extract" / "Kepler" / "KIC_3542116.parquet"


class TestRoundTrip:
    def test_arrays_round_trip_exactly(self, tmp_path: Path) -> None:
        data = _make_light_curve()
        path = tmp_path / "extract.parquet"
        save_extract(data, path)
        loaded = load_extract(path)

        assert np.array_equal(data.time, loaded.time)
        assert np.array_equal(data.flux, loaded.flux)
        assert np.array_equal(data.flux_err, loaded.flux_err)

    def test_scalar_fields_round_trip(self, tmp_path: Path) -> None:
        data = _make_light_curve()
        path = tmp_path / "extract.parquet"
        save_extract(data, path)
        loaded = load_extract(path)

        assert loaded.target_id == data.target_id
        assert loaded.mission == data.mission
        assert loaded.meta == data.meta

    def test_loaded_object_passes_validation_and_derived_properties(
        self, tmp_path: Path
    ) -> None:
        data = _make_light_curve()
        path = tmp_path / "extract.parquet"
        save_extract(data, path)
        loaded = load_extract(path)

        assert loaded.n_points == data.n_points
        assert loaded.baseline_days == pytest.approx(data.baseline_days)

    def test_target_id_with_space_round_trips_through_sanitized_path(
        self, tmp_path: Path
    ) -> None:
        data = _make_light_curve(target_id="KIC 3542116")
        path = extract_path(tmp_path, data.mission, data.target_id)
        save_extract(data, path)
        loaded = load_extract(path)

        # The filename is sanitized, but the metadata inside the file is not --
        # the original, space-containing target_id must come back exactly.
        assert loaded.target_id == "KIC 3542116"

    def test_empty_meta_round_trips(self, tmp_path: Path) -> None:
        data = LightCurveData(
            target_id="TIC 1",
            mission="TESS",
            time=np.array([1.0, 2.0, 3.0]),
            flux=np.array([1.0, 1.0, 1.0]),
            flux_err=np.array([0.01, 0.01, 0.01]),
        )
        path = tmp_path / "extract.parquet"
        save_extract(data, path)
        loaded = load_extract(path)
        assert loaded.meta == {}


class TestAtomicWrite:
    def test_successful_save_leaves_no_tmp_file(self, tmp_path: Path) -> None:
        data = _make_light_curve()
        path = tmp_path / "extract.parquet"
        save_extract(data, path)

        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []
        assert path.exists()

    def test_interrupted_write_leaves_no_corrupt_file_at_final_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _make_light_curve()
        path = tmp_path / "extract.parquet"

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(pq, "write_table", _boom)

        with pytest.raises(RuntimeError, match="simulated crash mid-write"):
            save_extract(data, path)

        assert not path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failed_replace_leaves_no_corrupt_file_and_no_tmp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _make_light_curve()
        path = tmp_path / "extract.parquet"

        import os as os_module

        real_replace = os_module.replace

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("simulated crash before rename completes")

        monkeypatch.setattr(os_module, "replace", _boom)

        with pytest.raises(OSError, match="simulated crash before rename completes"):
            save_extract(data, path)

        assert not path.exists()
        assert list(tmp_path.glob("*.tmp")) == []
        monkeypatch.setattr(os_module, "replace", real_replace)

    def test_a_previously_saved_file_is_untouched_by_a_failed_resave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash while overwriting an existing, valid extract must not corrupt it."""
        original = _make_light_curve()
        path = tmp_path / "extract.parquet"
        save_extract(original, path)
        original_bytes = path.read_bytes()

        updated = _make_light_curve()
        updated = LightCurveData(
            target_id=updated.target_id,
            mission=updated.mission,
            time=updated.time,
            flux=updated.flux * 2.0,
            flux_err=updated.flux_err,
            meta=updated.meta,
        )

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(pq, "write_table", _boom)
        with pytest.raises(RuntimeError):
            save_extract(updated, path)

        # The original file at `path` was never touched by the failed write.
        assert path.read_bytes() == original_bytes
        assert list(tmp_path.glob("*.tmp")) == []
