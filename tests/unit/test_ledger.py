# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Append-only semantics, corrupt-line recovery and retry logic for :mod:`exocomet.io.ledger`."""

from __future__ import annotations

import json
from pathlib import Path

from exocomet.io.ledger import (
    LedgerEntry,
    append_entry,
    get_status,
    latest_entries,
    next_targets,
    read_ledger,
)


def _entry(
    target_id: str,
    status: str = "success",
    mission: str = "Kepler",
    attempt: int = 1,
    timestamp_utc: str = "2026-09-15T00:00:00+00:00",
    error: str | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        target_id=target_id,
        mission=mission,
        status=status,  # type: ignore[arg-type]
        timestamp_utc=timestamp_utc,
        attempt=attempt,
        error=error,
        extract_path=f"data/extract/{mission}/{target_id}.parquet",
        results_path=None,
    )


class TestAppendAndRead:
    def test_append_then_read_round_trips(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        entry = _entry("KIC 1", status="success")
        append_entry(ledger_path, entry)

        entries = read_ledger(ledger_path)
        assert entries == [entry]

    def test_append_is_true_append_not_rewrite(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        first = _entry("KIC 1", status="failed", attempt=1)
        second = _entry("KIC 1", status="success", attempt=2)
        append_entry(ledger_path, first)
        append_entry(ledger_path, second)

        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["attempt"] == 1
        assert json.loads(lines[1])["attempt"] == 2

    def test_read_missing_ledger_returns_empty(self, tmp_path: Path) -> None:
        assert read_ledger(tmp_path / "does_not_exist.jsonl") == []

    def test_read_ignores_blank_lines(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(
            json.dumps(
                {
                    "target_id": "KIC 1",
                    "mission": "Kepler",
                    "status": "success",
                    "timestamp_utc": "2026-09-15T00:00:00+00:00",
                    "attempt": 1,
                    "error": None,
                    "extract_path": None,
                    "results_path": None,
                }
            )
            + "\n\n",
            encoding="utf-8",
        )
        entries = read_ledger(ledger_path)
        assert len(entries) == 1


class TestCorruptLastLine:
    def test_truncated_last_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        good_entries = [_entry(f"KIC {i}", status="success") for i in range(5)]
        with ledger_path.open("w", encoding="utf-8") as handle:
            for e in good_entries:
                handle.write(json.dumps(e.__dict__) + "\n")
            # Simulate a process killed mid-write of the JSON line itself.
            handle.write('{"target_id": "KIC 99", "mission": "Kepler", "stat')

        entries = read_ledger(ledger_path)
        assert len(entries) == 5
        assert [e.target_id for e in entries] == [f"KIC {i}" for i in range(5)]

    def test_line_missing_required_field_is_skipped(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        good = _entry("KIC 1", status="success")
        with ledger_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(good.__dict__) + "\n")
            # Valid JSON, but missing required fields for LedgerEntry.
            handle.write(json.dumps({"target_id": "KIC 2"}) + "\n")

        entries = read_ledger(ledger_path)
        assert len(entries) == 1
        assert entries[0].target_id == "KIC 1"

    def test_read_recovers_after_next_valid_append(self, tmp_path: Path) -> None:
        """A truncated line followed by a properly appended, valid entry is fine."""
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(tmp_path / "ledger.jsonl", _entry("KIC 1", status="success"))
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write('{"target_id": "KIC 2", "trunc')
        append_entry(ledger_path, _entry("KIC 3", status="success"))

        entries = read_ledger(ledger_path)
        # The truncated write left a broken middle line, but everything
        # around it -- including entries appended afterwards -- still reads.
        assert [e.target_id for e in entries] == ["KIC 1", "KIC 3"]


class TestLatestEntries:
    def test_latest_entry_wins_per_target(self, tmp_path: Path) -> None:
        entries = [
            _entry("KIC 1", status="failed", attempt=1, timestamp_utc="2026-09-15T00:00:00+00:00"),
            _entry("KIC 1", status="success", attempt=2, timestamp_utc="2026-09-15T01:00:00+00:00"),
        ]
        latest = latest_entries(entries)
        assert latest[("KIC 1", "Kepler")].status == "success"
        assert latest[("KIC 1", "Kepler")].attempt == 2

    def test_different_missions_are_distinct_keys(self) -> None:
        entries = [
            _entry("TIC 1", status="success", mission="TESS"),
            _entry("TIC 1", status="failed", mission="Kepler"),
        ]
        latest = latest_entries(entries)
        assert latest[("TIC 1", "TESS")].status == "success"
        assert latest[("TIC 1", "Kepler")].status == "failed"


class TestGetStatus:
    def test_never_attempted_is_pending(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        assert get_status(ledger_path, "KIC 1", "Kepler") == "pending"

    def test_most_recent_success_is_done(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("KIC 1", status="success"))
        assert get_status(ledger_path, "KIC 1", "Kepler") == "done"

    def test_most_recent_failure_after_success_is_pending(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(
            ledger_path,
            _entry("KIC 1", status="success", attempt=1, timestamp_utc="2026-09-15T00:00:00+00:00"),
        )
        append_entry(
            ledger_path,
            _entry("KIC 1", status="failed", attempt=2, timestamp_utc="2026-09-15T01:00:00+00:00"),
        )
        assert get_status(ledger_path, "KIC 1", "Kepler") == "pending"

    def test_only_failures_is_pending(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("KIC 1", status="failed", attempt=1))
        assert get_status(ledger_path, "KIC 1", "Kepler") == "pending"


class TestNextTargets:
    def test_never_attempted_targets_are_included(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        result = next_targets(ledger_path, ["KIC 1", "KIC 2"], mission="Kepler")
        assert result == ["KIC 1", "KIC 2"]

    def test_successful_targets_are_excluded(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("KIC 1", status="success"))
        result = next_targets(ledger_path, ["KIC 1", "KIC 2"], mission="Kepler")
        assert result == ["KIC 2"]

    def test_failed_under_limit_is_retriable(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("KIC 1", status="failed", attempt=1))
        result = next_targets(ledger_path, ["KIC 1"], mission="Kepler", max_attempts=3)
        assert result == ["KIC 1"]

    def test_failed_at_limit_is_excluded(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("KIC 1", status="failed", attempt=3))
        result = next_targets(ledger_path, ["KIC 1"], mission="Kepler", max_attempts=3)
        assert result == []

    def test_never_attempted_targets_come_before_retries(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("KIC 1", status="failed", attempt=1))
        result = next_targets(ledger_path, ["KIC 1", "KIC 2"], mission="Kepler")
        assert result == ["KIC 2", "KIC 1"]

    def test_retries_ordered_oldest_failure_first(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(
            ledger_path,
            _entry("KIC 1", status="failed", attempt=1, timestamp_utc="2026-09-15T02:00:00+00:00"),
        )
        append_entry(
            ledger_path,
            _entry("KIC 2", status="failed", attempt=1, timestamp_utc="2026-09-15T01:00:00+00:00"),
        )
        result = next_targets(ledger_path, ["KIC 1", "KIC 2"], mission="Kepler")
        assert result == ["KIC 2", "KIC 1"]

    def test_missions_are_not_confused(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "ledger.jsonl"
        append_entry(ledger_path, _entry("TIC 1", status="success", mission="TESS"))
        # Same target_id, different mission -- should still need processing.
        result = next_targets(ledger_path, ["TIC 1"], mission="Kepler")
        assert result == ["TIC 1"]


class TestFullRunSimulation:
    def test_realistic_multi_run_sequence(self, tmp_path: Path) -> None:
        """A small synthetic ledger across three simulated runs."""
        ledger_path = tmp_path / "ledger.jsonl"
        targets = ["KIC 1", "KIC 2", "KIC 3"]

        # Run 1: KIC 1 succeeds, KIC 2 fails, KIC 3 never attempted (budget ran out).
        append_entry(
            ledger_path,
            _entry("KIC 1", status="success", attempt=1, timestamp_utc="2026-09-15T00:00:00+00:00"),
        )
        append_entry(
            ledger_path,
            _entry(
                "KIC 2",
                status="failed",
                attempt=1,
                timestamp_utc="2026-09-15T00:01:00+00:00",
                error="MastTimeoutError",
            ),
        )

        pending = next_targets(ledger_path, targets, mission="Kepler")
        assert pending == ["KIC 3", "KIC 2"]

        # Run 2: KIC 3 succeeds, KIC 2 fails again.
        append_entry(
            ledger_path,
            _entry("KIC 3", status="success", attempt=1, timestamp_utc="2026-09-15T01:00:00+00:00"),
        )
        append_entry(
            ledger_path,
            _entry(
                "KIC 2",
                status="failed",
                attempt=2,
                timestamp_utc="2026-09-15T01:01:00+00:00",
                error="MastTimeoutError",
            ),
        )

        pending = next_targets(ledger_path, targets, mission="Kepler", max_attempts=3)
        assert pending == ["KIC 2"]
        assert get_status(ledger_path, "KIC 1", "Kepler") == "done"
        assert get_status(ledger_path, "KIC 2", "Kepler") == "pending"
        assert get_status(ledger_path, "KIC 3", "Kepler") == "done"

        # Run 3: KIC 2 fails a third time and exhausts its retry budget.
        append_entry(
            ledger_path,
            _entry(
                "KIC 2",
                status="failed",
                attempt=3,
                timestamp_utc="2026-09-15T02:00:00+00:00",
                error="MastTimeoutError",
            ),
        )
        pending = next_targets(ledger_path, targets, mission="Kepler", max_attempts=3)
        assert pending == []
