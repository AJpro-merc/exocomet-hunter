# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Dashboard-only SQLite storage: run history, per-target status, candidates.

Deliberately a separate database from ``scripts/build_local_db.py``'s
``local.db`` -- that one is a disposable view over the *real* automation's
ledger; this one is the dashboard's own run history (real, demo, and
injected runs alike), and the two must never mix. A dashboard run, however
real its data, never writes to the production ledger/data-repo.

Schema
------
``runs`` -- one row per dashboard run: id, mode (normal/research), mission,
started/ended timestamps, status (running/completed/stopped/force_stopped),
injected flag, the config actually used (as JSON, for provenance), and the
target list actually used (as JSON).

``run_targets`` -- one row per target attempted within a run: status
(pending/running/success/failed), timing, candidate/flagged counts, error.

``candidates`` -- one row per detected event, mirrors
``CandidateRecord.to_row()`` (see ``core/types.py``) plus ``run_id`` and
``injected`` -- the field nothing in the pipeline itself carries, added here
so an injected result can never be displayed indistinguishably from a real
one.

``coordinate_cache`` -- resolved RA/Dec per (target_id, mission), so the same
star's sky image is only searched-for once across the dashboard's lifetime.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dashboard import DASHBOARD_VAR_DIR

DB_PATH = DASHBOARD_VAR_DIR / "dashboard.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    mission TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    injected INTEGER NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL,
    target_list_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    mission TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    n_candidates INTEGER,
    n_flagged INTEGER,
    error TEXT,
    injected INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    mission TEXT NOT NULL,
    t_min REAL,
    depth_ppm REAL,
    duration_days REAL,
    a_dur_sigma REAL,
    significance REAL,
    delta_bic REAL,
    tau_over_sigma REAL,
    signs_agree INTEGER,
    flagged INTEGER,
    periodic INTEGER,
    period_days REAL,
    vetting_status TEXT,
    vetting_reasons TEXT,
    injected INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS coordinate_cache (
    target_id TEXT NOT NULL,
    mission TEXT NOT NULL,
    ra REAL,
    dec REAL,
    resolved_at TEXT NOT NULL,
    PRIMARY KEY (target_id, mission)
);

CREATE INDEX IF NOT EXISTS idx_run_targets_run ON run_targets(run_id);
CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_candidates_flagged ON candidates(flagged);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DASHBOARD_VAR_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def create_run(
    run_id: str,
    mode: str,
    mission: str,
    injected: bool,
    config: dict[str, Any],
    target_list: list[str],
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, mode, mission, started_at, status, injected, "
            "config_json, target_list_json) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)",
            (
                run_id,
                mode,
                mission,
                _now(),
                int(injected),
                json.dumps(config),
                json.dumps(target_list),
            ),
        )


def set_run_status(run_id: str, status: str, ended: bool = False) -> None:
    with _connect() as conn:
        if ended:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?",
                (status, _now(), run_id),
            )
        else:
            conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))


def add_run_target(
    run_id: str,
    target_id: str,
    mission: str,
    status: str,
    n_candidates: int | None,
    n_flagged: int | None,
    error: str | None,
    injected: bool,
    started_at: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO run_targets (run_id, target_id, mission, status, started_at, "
            "ended_at, n_candidates, n_flagged, error, injected) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                target_id,
                mission,
                status,
                started_at,
                _now(),
                n_candidates,
                n_flagged,
                error,
                int(injected),
            ),
        )


def add_candidates(
    run_id: str, target_id: str, mission: str, rows: list[dict[str, Any]], injected: bool
) -> None:
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO candidates (run_id, target_id, mission, t_min, depth_ppm, "
            "duration_days, a_dur_sigma, significance, delta_bic, tau_over_sigma, "
            "signs_agree, flagged, periodic, period_days, vetting_status, "
            "vetting_reasons, injected) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    target_id,
                    mission,
                    r.get("t_min"),
                    r.get("depth_ppm"),
                    r.get("duration_days"),
                    r.get("a_dur_sigma"),
                    r.get("significance"),
                    r.get("delta_bic"),
                    r.get("tau_over_sigma"),
                    int(bool(r.get("signs_agree"))),
                    int(bool(r.get("flagged"))),
                    int(bool(r.get("periodic"))),
                    r.get("period_days"),
                    r.get("vetting_status"),
                    r.get("vetting_reasons"),
                    int(injected),
                )
                for r in rows
            ],
        )


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def get_run_targets(run_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM run_targets WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_candidates(run_id: str, target_id: str | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if target_id:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE run_id = ? AND target_id = ? ORDER BY id",
                (run_id, target_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_cached_coordinates(target_id: str, mission: str) -> tuple[float, float] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT ra, dec FROM coordinate_cache WHERE target_id = ? AND mission = ?",
            (target_id, mission),
        ).fetchone()
        if row and row["ra"] is not None and row["dec"] is not None:
            return float(row["ra"]), float(row["dec"])
        return None


def cache_coordinates(target_id: str, mission: str, ra: float | None, dec: float | None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO coordinate_cache (target_id, mission, ra, dec, resolved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (target_id, mission, ra, dec, _now()),
        )
