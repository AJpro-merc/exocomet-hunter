# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scripts/ingest.py -- the resumable hourly ingestion loop.

No network in any test here: every test uses ``--offline``/``offline=True``,
which routes through a synthetic light curve generator instead of MAST.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import ingest  # noqa: E402
from build_kepler_host_list import EXCLUDED_HOLD_OUT_TARGETS, KNOWN_BAD_HOSTS  # noqa: E402


def test_get_target_list_excludes_hold_out_by_construction() -> None:
    targets = ingest.get_target_list("Kepler")
    assert not (set(targets) & EXCLUDED_HOLD_OUT_TARGETS)
    assert not (set(targets) & KNOWN_BAD_HOSTS)
    assert len(targets) > 0


def test_get_target_list_guard_fires_on_a_leaked_hold_out_target(monkeypatch) -> None:
    import build_kepler_host_list as bhl

    monkeypatch.setattr(bhl, "KEPLER_HOST_STARS", bhl.KEPLER_HOST_STARS + ("KIC 3542116",))
    with pytest.raises(AssertionError, match="hold-out"):
        ingest.get_target_list("Kepler")


def test_run_ingest_offline_writes_extract_ledger_and_results(tmp_path: Path) -> None:
    from exocomet.io import extract as extract_mod
    from exocomet.io import ledger as ledger_mod

    summary = ingest.run_ingest(
        mission="Kepler",
        data_repo_dir=tmp_path,
        time_budget_minutes=5.0,
        max_attempts=3,
        offline=True,
        max_targets=2,
        config_path=None,
    )

    assert summary["succeeded"] == 2
    assert summary["failed"] == 0

    entries = ledger_mod.read_ledger(tmp_path / "ledger.jsonl")
    assert len(entries) == 2
    assert all(e.status == "success" for e in entries)

    for entry in entries:
        assert entry.extract_path is not None
        assert entry.results_path is not None
        extract_file = tmp_path / entry.extract_path
        results_file = tmp_path / entry.results_path
        assert extract_file.exists()
        assert results_file.exists()

        lc = extract_mod.load_extract(extract_file)
        assert lc.target_id == entry.target_id
        assert lc.n_points > 0


def test_run_ingest_resumes_from_next_unprocessed_target(tmp_path: Path) -> None:
    first = ingest.run_ingest(
        mission="Kepler",
        data_repo_dir=tmp_path,
        time_budget_minutes=5.0,
        max_attempts=3,
        offline=True,
        max_targets=2,
        config_path=None,
    )
    assert first["succeeded"] == 2

    second = ingest.run_ingest(
        mission="Kepler",
        data_repo_dir=tmp_path,
        time_budget_minutes=5.0,
        max_attempts=3,
        offline=True,
        max_targets=3,
        config_path=None,
    )
    assert second["succeeded"] == 3

    from exocomet.io import ledger as ledger_mod

    entries = ledger_mod.read_ledger(tmp_path / "ledger.jsonl")
    processed = {e.target_id for e in entries}
    assert len(processed) == 5, "second run must have processed 3 NEW targets, not repeats"


def test_run_ingest_respects_time_budget(tmp_path: Path) -> None:
    """A budget of ~0 minutes should process nothing and leave everything for next run."""
    all_targets = ingest.get_target_list("Kepler")

    summary = ingest.run_ingest(
        mission="Kepler",
        data_repo_dir=tmp_path,
        time_budget_minutes=0.0,
        max_attempts=3,
        offline=True,
        max_targets=None,
        config_path=None,
    )
    assert summary["succeeded"] == 0
    assert summary["skipped_budget"] == len(all_targets)


def test_a_failed_target_is_recorded_and_retriable(tmp_path: Path, monkeypatch) -> None:
    """A target that raises must get a failed ledger entry, not crash the run."""
    from exocomet.io import ledger as ledger_mod

    call_count = {"n": 0}
    real_process = ingest.process_one_target

    def flaky_process(target_id, mission, config, data_repo_dir, attempt, offline, offline_seed=0):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated MAST failure")
        return real_process(
            target_id, mission, config, data_repo_dir, attempt, offline, offline_seed
        )

    monkeypatch.setattr(ingest, "process_one_target", flaky_process)

    summary = ingest.run_ingest(
        mission="Kepler",
        data_repo_dir=tmp_path,
        time_budget_minutes=5.0,
        max_attempts=3,
        offline=True,
        max_targets=2,
        config_path=None,
    )
    assert summary["failed"] == 1
    assert summary["succeeded"] == 1

    entries = ledger_mod.read_ledger(tmp_path / "ledger.jsonl")
    failed = [e for e in entries if e.status == "failed"]
    assert len(failed) == 1
    assert failed[0].error is not None
    assert "simulated MAST failure" in failed[0].error
    assert failed[0].extract_path is None
    assert failed[0].results_path is None


def test_a_star_with_zero_candidates_still_gets_a_results_file(tmp_path: Path) -> None:
    """A non-detection is retained, not silently dropped -- an empty results file
    is written and the ledger records success, same as a star with candidates."""
    from exocomet.io import ledger as ledger_mod

    ingest.run_ingest(
        mission="Kepler",
        data_repo_dir=tmp_path,
        time_budget_minutes=5.0,
        max_attempts=3,
        offline=True,
        max_targets=1,
        config_path=None,
    )
    entries = ledger_mod.read_ledger(tmp_path / "ledger.jsonl")
    assert len(entries) == 1
    assert entries[0].status == "success"
    results_file = tmp_path / entries[0].results_path
    assert results_file.exists()
    # a flat noise-only synthetic host is expected to yield zero candidates
    assert results_file.read_text(encoding="utf-8").strip() == ""
