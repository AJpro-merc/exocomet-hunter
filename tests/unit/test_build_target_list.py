# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Logic tests for :mod:`scripts.build_target_list` -- no network calls.

Every test here exercises the matching/filtering/exclusion functions with
hand-built fixture data (plain dicts, small astropy Tables, or a monkeypatched
``_load_cache``/network-call function), never a real VizieR/MAST/lightkurve
round-trip -- see the module docstring's "Design requirements" for why.

The single most important test in this file is
``test_enforce_hold_out_exclusion_rejects_hold_out_and_known_bad_kic``: it
proves that a target list containing either validation hold-out star, or the
known-bad host, is rejected rather than silently written out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from astropy.table import MaskedColumn, Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import build_target_list as btl

# -- kic_number / is_forbidden_kic -----------------------------------------------


class TestKicNumber:
    def test_parses_kic_string_with_space(self) -> None:
        assert btl.kic_number("KIC 3542116") == 3542116

    def test_parses_kic_string_no_space_and_lowercase(self) -> None:
        assert btl.kic_number("kic3542116") == 3542116

    def test_accepts_bare_int(self) -> None:
        assert btl.kic_number(1026032) == 1026032

    def test_none_in_none_out(self) -> None:
        assert btl.kic_number(None) is None

    def test_tic_string_is_not_a_kic_number(self) -> None:
        """A TIC id string must never parse as a KIC number, even with matching digits.

        This is the fix for a real bug caught during development: an earlier
        version grabbed any trailing digit run from any string, so a TIC id
        that happened to numerically coincide with a forbidden KIC number
        would have been wrongly excluded (safe direction, but wrong, and a
        sign the parser was not actually checking identity).
        """
        assert btl.kic_number("TIC 3542116") is None

    def test_star_name_with_no_digits_returns_none(self) -> None:
        assert btl.kic_number("beta Pic") is None


class TestIsForbiddenKic:
    def test_hold_out_targets_are_forbidden(self) -> None:
        assert btl.is_forbidden_kic("KIC 3542116") is True
        assert btl.is_forbidden_kic("KIC 11084727") is True

    def test_known_bad_host_is_forbidden(self) -> None:
        assert btl.is_forbidden_kic("KIC 1026032") is True
        assert btl.is_forbidden_kic(1026032) is True

    def test_ordinary_kic_is_not_forbidden(self) -> None:
        assert btl.is_forbidden_kic("KIC 1161345") is False

    def test_tic_string_never_forbidden_even_with_colliding_digits(self) -> None:
        assert btl.is_forbidden_kic("TIC 3542116") is False


# -- enforce_hold_out_exclusion: the most important test -------------------------


def _row(target_id: str, mission: str = "Kepler", role: str = "disc") -> btl.TargetRow:
    return btl.TargetRow(
        target_id=target_id,
        mission=mission,
        role=role,
        pair_id="test-pair",
        host_name="Test Host",
        catalog_ref="TEST/CAT/1",
    )


class TestEnforceHoldOutExclusion:
    def test_clean_list_passes_through_unchanged(self) -> None:
        rows = [_row("KIC 1161345"), _row("TIC 27436029", mission="TESS")]
        out = btl.enforce_hold_out_exclusion(rows)
        assert out == rows

    def test_rejects_first_validation_hold_out_target(self) -> None:
        rows = [_row("KIC 1161345"), _row("KIC 3542116")]
        out = btl.enforce_hold_out_exclusion(rows)
        assert [r.target_id for r in out] == ["KIC 1161345"]

    def test_rejects_second_validation_hold_out_target(self) -> None:
        rows = [_row("KIC 11084727"), _row("KIC 1161345")]
        out = btl.enforce_hold_out_exclusion(rows)
        assert [r.target_id for r in out] == ["KIC 1161345"]

    def test_rejects_known_bad_host(self) -> None:
        rows = [_row("KIC 1026032"), _row("KIC 1161345")]
        out = btl.enforce_hold_out_exclusion(rows)
        assert [r.target_id for r in out] == ["KIC 1161345"]

    def test_rejects_hold_out_target_in_control_role_too(self) -> None:
        """A hold-out star must never leak in even as someone else's control."""
        rows = [_row("KIC 1161345", role="disc"), _row("KIC 3542116", role="control")]
        out = btl.enforce_hold_out_exclusion(rows)
        assert [r.target_id for r in out] == ["KIC 1161345"]

    def test_never_returns_a_forbidden_row_even_if_called_directly_bypassing_upstream_checks(
        self,
    ) -> None:
        """The defence-in-depth case: prove the assert itself is load-bearing.

        Even if every upstream filter were skipped (a future bug, a new
        call-site that forgets to check), this function alone must still
        strip every hold-out/known-bad row and never assert-fail on its own
        output -- i.e. the function is self-correcting, not just
        self-checking.
        """
        rows = [_row("KIC 3542116"), _row("KIC 11084727"), _row("KIC 1026032"), _row("KIC 42")]
        out = btl.enforce_hold_out_exclusion(rows)
        assert {r.target_id for r in out} == {"KIC 42"}
        for row in out:
            assert not btl.is_forbidden_kic(row.target_id)


# -- is_forbidden_target: catches a star reached via TIC->KIC cross-match --------


class TestIsForbiddenTarget:
    def test_disc_star_resolving_to_hold_out_kic_is_forbidden(self) -> None:
        """A disc-catalogue star whose TIC cross-match reveals a hold-out KIC.

        This is the scenario the task is most worried about: the debris-disc
        catalogue names a star some other way (a HD number, a common name --
        anything but "KIC 3542116"), and it is only the TIC cross-match's KIC
        cross-id column that reveals it is physically the same star as one of
        the validation hold-outs.
        """
        entry = btl.DiscCatalogEntry(name="Some Unrelated Name", ra_deg=290.0, dec_deg=45.0)
        tic = btl.TicMatch(tic_id=999888777, ra_deg=290.0, dec_deg=45.0, kic_id=3542116)
        resolved = btl.ResolvedTarget(disc_entry=entry, tic=tic)
        assert btl.is_forbidden_target(resolved) is True

    def test_disc_star_with_no_kic_crossmatch_is_not_forbidden(self) -> None:
        entry = btl.DiscCatalogEntry(name="Random Star", ra_deg=10.0, dec_deg=-5.0)
        tic = btl.TicMatch(tic_id=1, ra_deg=10.0, dec_deg=-5.0, kic_id=None)
        resolved = btl.ResolvedTarget(disc_entry=entry, tic=tic)
        assert btl.is_forbidden_target(resolved) is False

    def test_ordinary_kic_crossmatch_is_not_forbidden(self) -> None:
        entry = btl.DiscCatalogEntry(name="Random Star", ra_deg=10.0, dec_deg=-5.0)
        tic = btl.TicMatch(tic_id=1, ra_deg=10.0, dec_deg=-5.0, kic_id=1161345)
        resolved = btl.ResolvedTarget(disc_entry=entry, tic=tic)
        assert btl.is_forbidden_target(resolved) is False


# -- select_control_star -----------------------------------------------------------


class TestSelectControlStar:
    def _target(self, tmag: float = 8.0, plx: float | None = 20.0) -> btl.TicMatch:
        return btl.TicMatch(tic_id=100, ra_deg=0.0, dec_deg=0.0, tmag=tmag, plx_mas=plx)

    def test_picks_closest_magnitude_match_within_tolerance(self) -> None:
        target = self._target(tmag=8.0)
        candidates = [
            btl.TicMatch(tic_id=200, ra_deg=0.01, dec_deg=0.0, tmag=8.4, separation_arcsec=30.0),
            btl.TicMatch(tic_id=201, ra_deg=0.02, dec_deg=0.0, tmag=8.05, separation_arcsec=60.0),
            btl.TicMatch(tic_id=202, ra_deg=0.03, dec_deg=0.0, tmag=9.5, separation_arcsec=10.0),
        ]
        control = btl.select_control_star(target, candidates, mag_tolerance=0.5)
        assert control is not None
        assert control.tic_id == 201

    def test_excludes_candidate_outside_magnitude_tolerance(self) -> None:
        target = self._target(tmag=8.0)
        candidates = [btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=9.0)]
        assert btl.select_control_star(target, candidates, mag_tolerance=0.5) is None

    def test_excludes_candidate_outside_distance_tolerance(self) -> None:
        # plx=20 mas -> 50 pc; a candidate at plx=5 mas -> 200 pc is far outside 30%.
        target = self._target(tmag=8.0, plx=20.0)
        candidates = [
            btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=8.1, plx_mas=5.0),
        ]
        control = btl.select_control_star(target, candidates, dist_tolerance_frac=0.3)
        assert control is None

    def test_never_picks_the_target_itself(self) -> None:
        target = self._target(tmag=8.0)
        candidates = [btl.TicMatch(tic_id=100, ra_deg=0.0, dec_deg=0.0, tmag=8.0)]
        assert btl.select_control_star(target, candidates) is None

    def test_never_picks_a_known_disc_catalogue_member(self) -> None:
        target = self._target(tmag=8.0)
        candidates = [btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=8.0)]
        control = btl.select_control_star(target, candidates, excluded_tic_ids={200})
        assert control is None

    def test_never_picks_a_hold_out_or_known_bad_kic_as_a_control(self) -> None:
        """The control-group path must go through the same hold-out gate."""
        target = self._target(tmag=8.0)
        candidates = [
            btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=8.0, kic_id=3542116),
            btl.TicMatch(tic_id=201, ra_deg=0.0, dec_deg=0.0, tmag=8.0, kic_id=1026032),
        ]
        assert btl.select_control_star(target, candidates) is None

    def test_falls_back_to_a_clean_candidate_once_forbidden_ones_are_excluded(self) -> None:
        target = self._target(tmag=8.0)
        candidates = [
            btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=8.0, kic_id=3542116),
            btl.TicMatch(tic_id=201, ra_deg=0.0, dec_deg=0.0, tmag=8.1, kic_id=42),
        ]
        control = btl.select_control_star(target, candidates)
        assert control is not None
        assert control.tic_id == 201

    def test_candidate_with_no_tmag_is_skipped(self) -> None:
        target = self._target(tmag=8.0)
        candidates = [btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=None)]
        assert btl.select_control_star(target, candidates) is None

    def test_no_distance_known_on_either_side_still_allows_a_match(self) -> None:
        target = self._target(tmag=8.0, plx=None)
        candidates = [btl.TicMatch(tic_id=200, ra_deg=0.0, dec_deg=0.0, tmag=8.1, plx_mas=None)]
        control = btl.select_control_star(target, candidates)
        assert control is not None
        assert control.tic_id == 200


# -- parse_vizier_disc_table ------------------------------------------------------


class TestParseVizierDiscTable:
    def test_parses_standard_column_names(self) -> None:
        table = Table(
            {
                "Name": ["HD 172555", "49 Ceti"],
                "RAJ2000": [278.146, 21.294],
                "DEJ2000": [-64.874, -2.548],
                "SpType": ["A7V", "A1V"],
            }
        )
        entries = btl.parse_vizier_disc_table(table, catalog_ref="TEST/CAT/1")
        assert len(entries) == 2
        assert entries[0].name == "HD 172555"
        assert entries[0].ra_deg == pytest.approx(278.146)
        assert entries[0].sptype == "A7V"

    def test_falls_back_to_alternate_column_names(self) -> None:
        table = Table(
            {
                "MAIN_ID": ["Star A"],
                "RA_ICRS": [10.0],
                "DE_ICRS": [-5.0],
            }
        )
        entries = btl.parse_vizier_disc_table(table, catalog_ref="TEST/CAT/2")
        assert len(entries) == 1
        assert entries[0].name == "Star A"
        assert entries[0].sptype is None

    def test_rows_with_no_position_are_dropped_not_crashed_on(self) -> None:
        table = Table(
            {
                "Name": ["Good Star", "Bad Star"],
                "RAJ2000": [10.0, float("nan")],
                "DEJ2000": [-5.0, -5.0],
            }
        )
        entries = btl.parse_vizier_disc_table(table, catalog_ref="TEST/CAT/3")
        assert [e.name for e in entries] == ["Good Star"]

    def test_parses_sexagesimal_ra_dec_strings(self) -> None:
        """Regression test for a real bug found verifying this script.

        The default catalogue, Cotten & Song 2016 (J/ApJS/225/15), actually
        publishes RAJ2000/DEJ2000 as sexagesimal strings ("00 04 20.33" /
        "-29 16 07.7"), not decimal degrees -- discovered by running a real
        VizieR query, which returned zero usable rows because ``_safe_float``
        silently rejected every one. HR 9102, the catalogue's first row, is
        used verbatim (RA 00h04m20.33s / Dec -29d16m07.7s -> ~1.085,
        ~-29.269 deg).
        """
        table = Table(
            {
                "Name": ["HR 9102", "HD 105"],
                "RAJ2000": ["00 04 20.33", "00 05 52.64"],
                "DEJ2000": ["-29 16 07.7", "-41 45 11.7"],
            }
        )
        entries = btl.parse_vizier_disc_table(table, catalog_ref="TEST/CAT/5")
        assert len(entries) == 2
        assert entries[0].name == "HR 9102"
        assert entries[0].ra_deg == pytest.approx(1.0847, abs=1e-3)
        assert entries[0].dec_deg == pytest.approx(-29.2688, abs=1e-3)

    def test_missing_ra_dec_columns_raises(self) -> None:
        table = Table({"Name": ["Star"], "SomeOtherColumn": [1]})
        with pytest.raises(ValueError, match="RA/Dec"):
            btl.parse_vizier_disc_table(table, catalog_ref="TEST/CAT/4")


# -- _parse_tic_table ---------------------------------------------------------------


class TestParseTicTable:
    def test_parses_standard_tic_columns(self) -> None:
        table = Table(
            {
                "ID": [123456789],
                "ra": [10.5],
                "dec": [-3.2],
                "Tmag": [7.8],
                "Teff": [6000.0],
                "plx": [25.0],
                "KIC": [3312345],
                "dstArcSec": [1.2],
            }
        )
        matches = btl._parse_tic_table(table)
        assert len(matches) == 1
        m = matches[0]
        assert m.tic_id == 123456789
        assert m.tmag == pytest.approx(7.8)
        assert m.kic_id == 3312345
        assert m.distance_pc == pytest.approx(40.0)  # 1000/25

    def test_masked_kic_column_becomes_none(self) -> None:
        table = Table({"ID": [1, 2], "ra": [1.0, 2.0], "dec": [1.0, 2.0]})
        table["KIC"] = MaskedColumn(data=[111, 0], mask=[False, True])
        matches = btl._parse_tic_table(table)
        assert matches[0].kic_id == 111
        assert matches[1].kic_id is None

    def test_row_with_no_id_is_skipped(self) -> None:
        table = Table({"ra": [1.0, 2.0], "dec": [1.0, 2.0]})
        table["ID"] = MaskedColumn(data=[1, 0], mask=[False, True])
        matches = btl._parse_tic_table(table)
        assert len(matches) == 1
        assert matches[0].tic_id == 1


# -- best_tic_match ------------------------------------------------------------------


def test_best_tic_match_picks_smallest_separation() -> None:
    matches = [
        btl.TicMatch(tic_id=1, ra_deg=0.0, dec_deg=0.0, separation_arcsec=5.0),
        btl.TicMatch(tic_id=2, ra_deg=0.0, dec_deg=0.0, separation_arcsec=0.5),
    ]
    assert btl.best_tic_match(matches).tic_id == 2


def test_best_tic_match_empty_list_returns_none() -> None:
    assert btl.best_tic_match([]) is None


# -- cache helpers ------------------------------------------------------------------


class TestCache:
    def test_round_trip_through_disk(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache: dict = {}
        btl._cache_set(cache, "disc_catalog", "key1", {"hello": "world"})
        btl._save_cache(cache_path, cache)

        reloaded = btl._load_cache(cache_path)
        got = btl._cache_get(reloaded, "disc_catalog", "key1", ttl_days=30.0, refresh=False)
        assert got == {"hello": "world"}

    def test_missing_file_returns_empty_cache(self, tmp_path: Path) -> None:
        assert btl._load_cache(tmp_path / "does_not_exist.json") == {}

    def test_corrupt_file_returns_empty_cache_not_a_crash(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert btl._load_cache(path) == {}

    def test_stale_entry_is_not_returned(self) -> None:
        from datetime import UTC, datetime, timedelta

        cache: dict = {
            "disc_catalog": {
                "key1": {
                    "fetched_at": (datetime.now(UTC) - timedelta(days=100)).isoformat(),
                    "data": {"stale": True},
                }
            }
        }
        assert btl._cache_get(cache, "disc_catalog", "key1", ttl_days=30.0, refresh=False) is None

    def test_fresh_entry_is_returned(self) -> None:
        from datetime import UTC, datetime, timedelta

        cache: dict = {
            "disc_catalog": {
                "key1": {
                    "fetched_at": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
                    "data": {"fresh": True},
                }
            }
        }
        got = btl._cache_get(cache, "disc_catalog", "key1", ttl_days=30.0, refresh=False)
        assert got == {"fresh": True}

    def test_refresh_flag_bypasses_a_fresh_entry(self) -> None:
        from datetime import UTC, datetime

        cache: dict = {
            "disc_catalog": {
                "key1": {"fetched_at": datetime.now(UTC).isoformat(), "data": {"fresh": True}}
            }
        }
        assert btl._cache_get(cache, "disc_catalog", "key1", ttl_days=30.0, refresh=True) is None


# -- confirm_photometry: cached, using a monkeypatched network call --------------


class TestConfirmPhotometry:
    def test_caches_a_positive_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "cache.json"
        calls = []

        def fake_query(target_id: str, mission: str) -> int:
            calls.append((target_id, mission))
            return 3

        monkeypatch.setattr(btl, "_query_photometry_available", fake_query)
        cache: dict = {}
        first = btl.confirm_photometry(cache, cache_path, "KIC 1", "Kepler", refresh=False)
        second = btl.confirm_photometry(cache, cache_path, "KIC 1", "Kepler", refresh=False)
        assert first is True
        assert second is True
        assert calls == [("KIC 1", "Kepler")]  # only queried once -- second hit cache

    def test_zero_products_is_false_and_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "cache.json"
        monkeypatch.setattr(btl, "_query_photometry_available", lambda *_: 0)
        cache: dict = {}
        assert btl.confirm_photometry(cache, cache_path, "TIC 1", "TESS", refresh=False) is False

    def test_query_failure_is_treated_as_unavailable_not_a_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "cache.json"

        def boom(*_args: object, **_kwargs: object) -> int:
            raise RuntimeError("archive unreachable")

        monkeypatch.setattr(btl, "_query_photometry_available", boom)
        cache: dict = {}
        assert btl.confirm_photometry(cache, cache_path, "KIC 1", "Kepler", refresh=False) is False


# -- write_target_lists: format + the final defence-in-depth gate ------------------


class TestWriteTargetLists:
    def test_dry_run_writes_nothing(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rows = [_row("KIC 1161345")]
        btl.write_target_lists(rows, tmp_path, dry_run=True)
        assert not (tmp_path / "target_list_kepler.txt").exists()
        assert "would write" in capsys.readouterr().out

    def test_writes_parseable_kepler_and_tess_files(self, tmp_path: Path) -> None:
        rows = [
            _row("KIC 1161345", mission="Kepler", role="disc"),
            _row("TIC 27436029", mission="TESS", role="control"),
        ]
        btl.write_target_lists(rows, tmp_path, dry_run=False)

        # Round-trip through the real repo parser, exactly as
        # generate_training_labels.load_host_list will read these files.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
        import generate_training_labels as gtl

        kepler_ids = gtl.load_host_list(tmp_path / "target_list_kepler.txt")
        tess_ids = gtl.load_host_list(tmp_path / "target_list_tess.txt")
        assert kepler_ids == ("KIC 1161345",)
        assert tess_ids == ("TIC 27436029",)

    def test_writes_metadata_csv_with_one_row_per_target(self, tmp_path: Path) -> None:
        rows = [_row("KIC 1161345"), _row("TIC 1", mission="TESS")]
        btl.write_target_lists(rows, tmp_path, dry_run=False)
        content = (tmp_path / "target_list_metadata.csv").read_text(encoding="utf-8")
        # header + 2 data rows
        assert len(content.strip().splitlines()) == 3
        assert "target_id" in content

    def test_refuses_to_write_a_hold_out_target_even_if_it_reached_this_function(
        self, tmp_path: Path
    ) -> None:
        """write_target_lists re-enforces the exclusion right before writing."""
        rows = [_row("KIC 3542116"), _row("KIC 1161345")]
        btl.write_target_lists(rows, tmp_path, dry_run=False)

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
        import generate_training_labels as gtl

        kepler_ids = gtl.load_host_list(tmp_path / "target_list_kepler.txt")
        assert "KIC 3542116" not in kepler_ids
        assert kepler_ids == ("KIC 1161345",)
