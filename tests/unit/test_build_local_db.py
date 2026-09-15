# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scripts/build_local_db.py -- the disposable SQLite rebuild step.

Builds a small synthetic ledger.jsonl (and matching results files) by hand,
runs the build script against it, then queries the resulting SQLite file
directly with the stdlib ``sqlite3`` module -- no real pipeline run needed.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import build_local_db

from exocomet.io.ledger import LedgerEntry, append_entry


def _make_repo(tmp_path: Path) -> Path:
    """Build a synthetic data-repo-dir covering every case the task calls out.

    - KIC 1 (Kepler): succeeded, 2 candidates (1 flagged) -- normal case.
    - KIC 2 (Kepler): succeeded, 0 candidates -- empty-but-processed results file.
    - KIC 3 (Kepler): failed, no results file at all -- failed target.
    - KIC 4 (Kepler): failed then retried and succeeded -- two ledger lines,
      only the latest ("success", attempt 2) should win in ``targets``, but
      both must appear in ``attempts``.
    - KIC 5 (Kepler): failed once, attempt 1 of max_attempts=3 -- has retries left.
    """
    repo = tmp_path / "data-repo"
    ledger_path = repo / "ledger.jsonl"

    def results_path_for(target: str) -> Path:
        return repo / "results" / "Kepler" / f"{target}.jsonl"

    def write_results(target: str, rows: list[dict]) -> str:
        path = results_path_for(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return str(path.relative_to(repo))

    # KIC 1: 2 candidates, 1 flagged.
    rp1 = write_results(
        "KIC_1",
        [
            {"target_id": "KIC 1", "flagged": True, "significance": 5.0},
            {"target_id": "KIC 1", "flagged": False, "significance": 1.0},
        ],
    )
    append_entry(
        ledger_path,
        LedgerEntry(
            target_id="KIC 1",
            mission="Kepler",
            status="success",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            attempt=1,
            extract_path="extract/Kepler/KIC_1.parquet",
            results_path=rp1,
        ),
    )

    # KIC 2: processed, zero candidates (empty file, still written).
    rp2 = write_results("KIC_2", [])
    append_entry(
        ledger_path,
        LedgerEntry(
            target_id="KIC 2",
            mission="Kepler",
            status="success",
            timestamp_utc="2026-01-01T00:01:00+00:00",
            attempt=1,
            extract_path="extract/Kepler/KIC_2.parquet",
            results_path=rp2,
        ),
    )

    # KIC 3: failed, exhausted retries, no results file at all.
    append_entry(
        ledger_path,
        LedgerEntry(
            target_id="KIC 3",
            mission="Kepler",
            status="failed",
            timestamp_utc="2026-01-01T00:02:00+00:00",
            attempt=3,
            error="download timeout",
            extract_path=None,
            results_path=None,
        ),
    )

    # KIC 4: failed, then retried and succeeded.
    append_entry(
        ledger_path,
        LedgerEntry(
            target_id="KIC 4",
            mission="Kepler",
            status="failed",
            timestamp_utc="2026-01-01T00:03:00+00:00",
            attempt=1,
            error="transient network error",
            extract_path=None,
            results_path=None,
        ),
    )
    rp4 = write_results("KIC_4", [{"target_id": "KIC 4", "flagged": False, "significance": 2.0}])
    append_entry(
        ledger_path,
        LedgerEntry(
            target_id="KIC 4",
            mission="Kepler",
            status="success",
            timestamp_utc="2026-01-01T00:04:00+00:00",
            attempt=2,
            extract_path="extract/Kepler/KIC_4.parquet",
            results_path=rp4,
        ),
    )

    # KIC 5: failed once, retries left under max_attempts=3.
    append_entry(
        ledger_path,
        LedgerEntry(
            target_id="KIC 5",
            mission="Kepler",
            status="failed",
            timestamp_utc="2026-01-01T00:05:00+00:00",
            attempt=1,
            error="flaky",
            extract_path=None,
            results_path=None,
        ),
    )

    return repo


def test_build_local_db_targets_and_attempts(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = tmp_path / "local.db"

    summary = build_local_db.build_local_db(repo, out, max_attempts=3)

    assert summary["attempts"] == 6  # KIC 4 contributes 2 lines
    assert summary["targets"] == 5
    assert summary["candidates"] == 3  # 2 (KIC 1) + 0 (KIC 2) + 1 (KIC 4)

    conn = sqlite3.connect(out)
    try:
        cur = conn.cursor()

        # attempts: full history, KIC 4 appears twice.
        cur.execute("SELECT COUNT(*) FROM attempts WHERE target_id = 'KIC 4'")
        assert cur.fetchone()[0] == 2

        # targets: latest-status view, KIC 4 shows only its winning attempt.
        cur.execute("SELECT status, attempt, n_candidates FROM targets WHERE target_id = 'KIC 4'")
        row = cur.fetchone()
        assert row == ("success", 2, 1)

        # KIC 1: normal case, candidate + flagged counts.
        cur.execute("SELECT n_candidates, n_flagged FROM targets WHERE target_id = 'KIC 1'")
        assert cur.fetchone() == (2, 1)

        # KIC 2: processed but zero candidates -- must be 0, not NULL.
        cur.execute("SELECT status, n_candidates, n_flagged FROM targets WHERE target_id = 'KIC 2'")
        row = cur.fetchone()
        assert row == ("success", 0, 0)

        # KIC 3: failed, no results file at all -- must be NULL, not 0.
        cur.execute(
            "SELECT status, results_path, n_candidates, n_flagged, retries_left "
            "FROM targets WHERE target_id = 'KIC 3'"
        )
        row = cur.fetchone()
        assert row == ("failed", None, None, None, 0)  # attempt 3 of max_attempts=3: exhausted

        # KIC 5: failed once, retries left.
        cur.execute("SELECT status, attempt, retries_left FROM targets WHERE target_id = 'KIC 5'")
        assert cur.fetchone() == ("failed", 1, 2)

        # candidates table: every flagged candidate across all stars, in one query.
        cur.execute("SELECT target_id, mission, significance FROM candidates WHERE flagged = 1")
        flagged_rows = cur.fetchall()
        assert flagged_rows == [("KIC 1", "Kepler", 5.0)]

        # aggregate: total candidates found, total flagged.
        cur.execute("SELECT COUNT(*), SUM(flagged) FROM candidates")
        assert cur.fetchone() == (3, 1)

        # aggregate: total processed (success) per mission.
        cur.execute(
            "SELECT mission, COUNT(*) FROM targets WHERE status = 'success' GROUP BY mission"
        )
        assert cur.fetchall() == [("Kepler", 3)]
    finally:
        conn.close()


def test_build_local_db_overwrites_existing_file(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = tmp_path / "local.db"

    build_local_db.build_local_db(repo, out, max_attempts=3)
    first_mtime = out.stat().st_mtime_ns

    # Rebuild again -- must not error on an existing file, and must fully
    # replace it (a stray leftover row would mean a rebuild silently merged
    # instead of overwriting).
    summary = build_local_db.build_local_db(repo, out, max_attempts=3)
    assert summary["targets"] == 5
    assert out.stat().st_mtime_ns >= first_mtime

    conn = sqlite3.connect(out)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM targets")
        assert cur.fetchone()[0] == 5
    finally:
        conn.close()


def test_build_local_db_missing_ledger_produces_empty_db(tmp_path: Path) -> None:
    repo = tmp_path / "data-repo"  # never created
    out = tmp_path / "local.db"

    summary = build_local_db.build_local_db(repo, out, max_attempts=3)
    assert summary == {"attempts": 0, "targets": 0, "candidates": 0}

    conn = sqlite3.connect(out)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM targets")
        assert cur.fetchone()[0] == 0
    finally:
        conn.close()
