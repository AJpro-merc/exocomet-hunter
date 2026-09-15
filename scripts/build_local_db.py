# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild a disposable local SQLite view of the ledger, for ad-hoc queries.

The ledger (``<data-repo-dir>/ledger.jsonl``, see ``exocomet.io.ledger``) is
the source of truth. This script is *not* a second source of truth -- it is
a queryable view, rebuilt from scratch on demand, never committed to git,
never written by CI. Every run deletes any existing ``--out`` file and
recreates it from the ledger (and, for candidate counts, from the results
files the ledger points at). The ledger is one line per attempt across at
most a few hundred targets, so a full rebuild is cheap (well under a second)
and this avoids an entire class of incremental-sync bugs -- there is no
"did the DB miss an update" question to ask, because the DB never survives
between runs.

Schema
------
``attempts`` -- one row per raw ledger line (the full history, including
every retry and failure): ``target_id``, ``mission``, ``status``,
``timestamp_utc``, ``attempt``, ``error``, ``extract_path``,
``results_path``. Nothing is deduplicated here; this table answers
"show me the retry history for star X".

``targets`` -- one row per ``(target_id, mission)``, the latest-status view
(built from :func:`exocomet.io.ledger.latest_entries`, i.e. the last ledger
line seen for that key): ``status``, ``timestamp_utc``, ``attempt``,
``error``, ``extract_path``, ``results_path``, ``retries_left`` (computed
against ``--max-attempts``, only meaningful when ``status='failed'``),
``n_candidates``, ``n_flagged``. The last two are ``NULL`` when
``results_path`` is unset or the file does not exist (a failed or
never-attempted target -- there is nothing to count), and ``0`` when the
results file exists but is empty (a star that was fully processed and
found nothing). That NULL-vs-zero distinction is deliberate and is the
whole reason this table reads the results files at all rather than trusting
the ledger alone: "never produced a results file" and "produced one with
zero rows" are different facts about a star, and collapsing them would lose
exactly the distinction ``scripts/ingest.py``'s ``_write_results_jsonl``
docstring says the empty file exists to preserve.

``candidates`` -- one row per detected event (one line of a results file),
joined to its ``(target_id, mission)``. Built because it is cheap (the
results files are already being read once each for the ``targets`` table's
counts, so parsing every row into ``candidates`` too is a few extra lines,
not a second pass) and it is what turns "show me every flagged candidate
across all stars" from "open N files by hand" into one SQL query -- exactly
the kind of question this tool exists to make easy for someone who is not a
SQL expert. Columns mirror ``CandidateRecord.to_row()`` in
``exocomet.core.types`` (target_id, t_min, depth_ppm, duration_days,
a_dur_sigma, significance, delta_bic, tau_over_sigma, signs_agree, flagged,
periodic, period_days, vetting_status, vetting_reasons), plus ``mission``
(not in ``to_row()`` itself, but needed here since ``target_id`` alone is
not a unique key) and a foreign key point back at ``targets``.

``--out`` default: ``local.db`` in the current directory, not
``<data-repo-dir>/local.db``. The data-repo-dir tree is meant to be a git
clone of the separate ``exocomet-hunter-data`` repository (see
``scripts/ingest.py``'s module docstring); dropping a disposable, frequently
regenerated binary file inside it risks it getting swept up by a stray
``git add -A`` in that repo. A human who wants it to travel with the data
anyway can just pass ``--out data-repo/local.db`` explicitly.

Usage
-----
    python scripts/build_local_db.py
    python scripts/build_local_db.py --data-repo-dir data-repo --out local.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exocomet.io.ledger import latest_entries, read_ledger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_local_db")

__all__ = ["build_local_db", "build_parser", "main"]

_SCHEMA = """
CREATE TABLE attempts (
    target_id     TEXT NOT NULL,
    mission       TEXT NOT NULL,
    status        TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    attempt       INTEGER NOT NULL,
    error         TEXT,
    extract_path  TEXT,
    results_path  TEXT
);

CREATE TABLE targets (
    target_id     TEXT NOT NULL,
    mission       TEXT NOT NULL,
    status        TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    attempt       INTEGER NOT NULL,
    error         TEXT,
    extract_path  TEXT,
    results_path  TEXT,
    retries_left  INTEGER,
    n_candidates  INTEGER,
    n_flagged     INTEGER,
    PRIMARY KEY (target_id, mission)
);

CREATE TABLE candidates (
    target_id         TEXT NOT NULL,
    mission           TEXT NOT NULL,
    t_min             REAL,
    depth_ppm         REAL,
    duration_days     REAL,
    a_dur_sigma       REAL,
    significance      REAL,
    delta_bic         REAL,
    tau_over_sigma    REAL,
    signs_agree       INTEGER,
    flagged           INTEGER,
    periodic          INTEGER,
    period_days       REAL,
    vetting_status    TEXT,
    vetting_reasons   TEXT,
    FOREIGN KEY (target_id, mission) REFERENCES targets (target_id, mission)
);

CREATE INDEX idx_attempts_target ON attempts (target_id, mission);
CREATE INDEX idx_candidates_target ON candidates (target_id, mission);
CREATE INDEX idx_candidates_flagged ON candidates (flagged);
"""


def _read_results_rows(data_repo_dir: Path, results_path: str | None) -> list[dict] | None:
    """Read a results JSONL file's rows, or ``None`` if there is no such file.

    ``None`` (distinct from ``[]``) means "not processed / nothing to read"
    -- either the ledger never recorded a results path (failed/never
    attempted) or the path it recorded does not exist on disk. ``[]`` means
    the file exists and is genuinely empty (processed, zero candidates).
    Malformed lines are skipped and logged, mirroring
    :func:`exocomet.io.ledger.read_ledger`'s tolerance of a torn trailing
    write -- one bad line should not hide an otherwise-good results file.
    """
    if not results_path:
        return None
    path = data_repo_dir / results_path
    if not path.exists():
        return None

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("skipping corrupt results line %d in %s: %s", lineno, path, exc)
    return rows


def build_local_db(data_repo_dir: Path, out_path: Path, max_attempts: int = 3) -> dict[str, int]:
    """Rebuild ``out_path`` from ``<data_repo_dir>/ledger.jsonl``.

    Deletes any existing file at ``out_path`` first, so this is always a
    clean rebuild, never an incremental merge. Returns a small summary dict
    (row counts per table) for callers/tests to assert on.
    """
    ledger_path = data_repo_dir / "ledger.jsonl"
    entries = read_ledger(ledger_path)
    latest = latest_entries(entries)

    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(out_path)
    try:
        conn.executescript(_SCHEMA)

        conn.executemany(
            """
            INSERT INTO attempts
                (target_id, mission, status, timestamp_utc, attempt, error,
                 extract_path, results_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.target_id,
                    e.mission,
                    e.status,
                    e.timestamp_utc,
                    e.attempt,
                    e.error,
                    e.extract_path,
                    e.results_path,
                )
                for e in entries
            ],
        )

        n_candidate_rows = 0
        for (target_id, mission), entry in latest.items():
            rows = _read_results_rows(data_repo_dir, entry.results_path)
            n_candidates = len(rows) if rows is not None else None
            n_flagged = sum(1 for r in rows if r.get("flagged")) if rows is not None else None

            retries_left: int | None = None
            if entry.status == "failed":
                retries_left = max(max_attempts - entry.attempt, 0)

            conn.execute(
                """
                INSERT INTO targets
                    (target_id, mission, status, timestamp_utc, attempt, error,
                     extract_path, results_path, retries_left, n_candidates, n_flagged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    mission,
                    entry.status,
                    entry.timestamp_utc,
                    entry.attempt,
                    entry.error,
                    entry.extract_path,
                    entry.results_path,
                    retries_left,
                    n_candidates,
                    n_flagged,
                ),
            )

            for row in rows or []:
                conn.execute(
                    """
                    INSERT INTO candidates
                        (target_id, mission, t_min, depth_ppm, duration_days, a_dur_sigma,
                         significance, delta_bic, tau_over_sigma, signs_agree, flagged,
                         periodic, period_days, vetting_status, vetting_reasons)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_id,
                        mission,
                        row.get("t_min"),
                        row.get("depth_ppm"),
                        row.get("duration_days"),
                        row.get("a_dur_sigma"),
                        row.get("significance"),
                        row.get("delta_bic"),
                        row.get("tau_over_sigma"),
                        row.get("signs_agree"),
                        row.get("flagged"),
                        row.get("periodic"),
                        row.get("period_days"),
                        row.get("vetting_status"),
                        row.get("vetting_reasons"),
                    ),
                )
                n_candidate_rows += 1

        conn.commit()
    finally:
        conn.close()

    summary = {
        "attempts": len(entries),
        "targets": len(latest),
        "candidates": n_candidate_rows,
    }
    logger.info(
        "%s: %d attempt(s), %d target(s), %d candidate(s)",
        out_path,
        summary["attempts"],
        summary["targets"],
        summary["candidates"],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-repo-dir",
        type=Path,
        default=Path("data-repo"),
        help="directory holding ledger.jsonl, extract/, results/ (matches scripts/ingest.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("local.db"),
        help=(
            "SQLite file to (re)build (default: local.db in the current directory -- "
            "pass e.g. --out data-repo/local.db to keep the view alongside the data it describes)"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="must match scripts/ingest.py's --max-attempts, so retries_left is accurate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.data_repo_dir.exists():
        print(
            f"No data repo directory found at {args.data_repo_dir} -- nothing to build yet. "
            "Run scripts/ingest.py first, or pass --data-repo-dir to point at the right place."
        )
        return 0
    if not (args.data_repo_dir / "ledger.jsonl").exists():
        print(
            f"{args.data_repo_dir} exists but has no ledger.jsonl yet -- nothing has been "
            "ingested. Run scripts/ingest.py first."
        )
        return 0

    summary = build_local_db(args.data_repo_dir, args.out, max_attempts=args.max_attempts)
    print(
        f"Built {args.out}: {summary['targets']} target(s), {summary['attempts']} attempt(s), "
        f"{summary['candidates']} candidate(s). Open it with the sqlite3 CLI or DB Browser for SQLite."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
